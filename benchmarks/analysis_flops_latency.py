# -*- coding: utf-8 -*-
"""RAGFAS efficiency analysis — FLOPs/MACs + stage-wise latency breakdown.

Self-contained single file for Google Colab (no dependency on the training
repo; the architecture is inlined 1:1).  Paste the whole file into one cell
or run ``python analysis_flops_latency.py [--image face.jpg]``.

What it reports
---------------
1. FLOPs / MACs per image, per stage, via ``torch.utils.flop_counter``
   (exact for conv / mm / bmm; PyTorch counts 1 MAC = 2 FLOPs) plus the
   analytic sparse-propagation term of the Chebyshev graph layers, and a
   per-module breakdown table (great for the paper appendix).
2. Median latency of the three pipeline stages, timed separately with CUDA
   events:
     [1] MediaPipe FaceMesh + patch cutting  (CPU, fp32 — the solutions API
         has no GPU backend, so MediaPipe always runs on CPU)
     [2] D121-LowFE CNN on the single [468,3,40,40] patch tensor (GPU)
     [3] RAGFAS graph forward: GFG + classifier + discriminator + RPN (GPU)
   ...for fp32 and fp16, plus a pipelined end-to-end loop (no per-stage sync)
   which is the real deployment throughput (FPS) including MediaPipe.
3. Parameter counts per stage.

Usage in Colab::

    !python analysis_flops_latency.py --image face.jpg --n-iter 200
"""

from __future__ import annotations

import argparse
import statistics
import time
import warnings

import numpy as np
import torch
import torch.nn as nn

# ============================== configuration ==============================
N_NODES = 468          # MediaPipe landmarks
PATCH = 40             # 40x40 patch per landmark (kernel_size=20)
FACE = 256             # aligned crop size
NODE_FEATURES = 640    # D121-LowFE output (512 + 128)
HIDDEN = 128
CHEB_K = 2
N_DOMAINS = 2          # limited-source MI -> C setting
N_REGION_SAMPLES = 64
N_WARMUP = 30
N_ITER = 200


def _ensure_import(module_name: str, pip_name: str | None = None) -> None:
    """Import a module, pip-installing it first if missing (Colab-friendly)."""
    try:
        __import__(module_name)
    except ImportError:
        import subprocess
        import sys
        pkg = pip_name or module_name
        print(f"[setup] installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
        __import__(module_name)


# ======================= inline model (1:1 with the repo) ==================
class RandomNodeDropout(nn.Module):
    def __init__(self, drop_prob=0.3):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, enabled=True):
        if not self.training or not enabled:
            return x
        mask = torch.rand(x.size(0), device=x.device) > self.drop_prob
        return x * mask.float().unsqueeze(1).expand_as(x)


class _GRLFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coeff):
        ctx.coeff = coeff
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g.neg() * ctx.coeff, None


class GRL(nn.Module):
    """GRL — identity at inference (coeff 0); kept for honest compute parity."""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return _GRLFn.apply(x, 0.0)


def global_mean(x, batch):
    from torch_geometric.nn import global_mean_pool
    return global_mean_pool(x, batch)


class GraphFeatureGenerator(nn.Module):
    """8x ChebConv(2) + BatchNorm + ReLU + DropNode, skip concat before conv5."""

    def __init__(self, in_ch=NODE_FEATURES, hidden=HIDDEN, k=CHEB_K):
        super().__init__()
        torch.manual_seed(12345)
        from torch_geometric.nn import ChebConv
        from torch_geometric.nn.norm import BatchNorm as GBatchNorm
        self.conv1 = ChebConv(in_ch, hidden, k)
        self.conv2 = ChebConv(hidden, hidden, k)
        self.conv3 = ChebConv(hidden, hidden, k)
        self.conv4 = ChebConv(hidden, hidden, k)
        self.conv5 = ChebConv(hidden * 2, hidden, k)
        self.conv6 = ChebConv(hidden, hidden, k)
        self.conv7 = ChebConv(hidden, hidden, k)
        self.conv8 = ChebConv(hidden, hidden, k)
        self.b1 = GBatchNorm(hidden)
        self.b2 = GBatchNorm(hidden)
        self.b3 = GBatchNorm(hidden)
        self.b4 = GBatchNorm(hidden)
        self.b5 = GBatchNorm(hidden)
        self.b6 = GBatchNorm(hidden)
        self.b7 = GBatchNorm(hidden)
        self.drop = RandomNodeDropout(0.3)

    def forward(self, x, edge_index, batch, drop=False):
        x = self.drop(self.b1(self.conv1(x, edge_index)).relu(), drop)
        x = self.drop(self.b2(self.conv2(x, edge_index)).relu(), drop)
        z1 = x
        h1 = global_mean(x, batch)
        x = self.drop(self.b3(self.conv3(x, edge_index)).relu(), drop)
        h3 = global_mean(x, batch)
        x = self.drop(self.b4(self.conv4(x, edge_index)).relu(), drop)
        x = torch.cat((x, z1), dim=1)
        x = self.drop(self.b5(self.conv5(x, edge_index)).relu(), drop)
        h5 = global_mean(x, batch)
        x = self.drop(self.b6(self.conv6(x, edge_index)).relu(), drop)
        x = self.b7(self.conv7(x, edge_index)).relu()
        x = self.conv8(x, edge_index)
        return {"z1": z1, "h1": h1, "h3": h3, "h5": h5,
                "h_final": global_mean(x, batch)}


class MultiLevelTokenClassifier(nn.Module):
    def __init__(self, tokens=HIDDEN, levels=4, embed=8, layers=5, heads=8):
        super().__init__()
        self.project = nn.Linear(levels, embed)
        enc = nn.TransformerEncoderLayer(d_model=embed, nhead=heads)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(tokens * embed, 1)

    def forward(self, h):
        x = self.encoder(self.project(h))
        return self.head(x.reshape(x.size(0), -1))


class DomainDiscriminator(nn.Module):
    """Node-level discriminator (paper Eq. 6): GraphSAGE x4 + per-node fc + softmax."""

    def __init__(self, hidden=HIDDEN, domains=N_DOMAINS):
        super().__init__()
        torch.manual_seed(12345)
        from torch_geometric.nn import GraphSAGE
        self.conv1 = GraphSAGE(hidden, hidden, 2, dropout=0.5)
        self.conv2 = GraphSAGE(hidden, hidden, 2, dropout=0.5)
        self.conv3 = GraphSAGE(hidden, hidden, 2, dropout=0.5)
        self.conv4 = GraphSAGE(hidden, hidden, 2, dropout=0.5)
        self.drop = RandomNodeDropout(0.3)
        self.fc2 = nn.Linear(hidden, domains)
        self.grl = GRL()

    def forward(self, z1, edge_index, drop=False):
        x = self.grl(z1)
        x = self.drop(self.conv1(x, edge_index).relu(), drop)
        x = self.drop(self.conv2(x, edge_index).relu(), drop)
        x = self.conv3(x, edge_index).relu()
        x = self.conv4(x, edge_index).relu()
        return self.fc2(x).softmax(dim=-1)


class RegionPredictionNetwork(nn.Module):
    """RPN (paper Sec. 3.5): ChebConv x2 + transformer (embed 64, 10 layers, 8 heads)."""

    def __init__(self, hidden=HIDDEN, embed=64, layers=10, heads=8, classes=7):
        super().__init__()
        torch.manual_seed(12345)
        from torch_geometric.nn import ChebConv
        self.cheb1 = ChebConv(hidden, hidden, CHEB_K)
        self.cheb2 = ChebConv(hidden, hidden, CHEB_K)
        self.project = nn.Linear(1, embed)
        enc = nn.TransformerEncoderLayer(d_model=embed, nhead=heads)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.head = nn.Linear(hidden * embed, classes)

    def forward(self, z1, edge_index, idx):
        h = self.cheb2(self.cheb1(z1, edge_index).relu(), edge_index).relu()
        h = h[idx].unsqueeze(2)
        x = self.encoder(self.project(h))
        return self.head(x.reshape(x.size(0), -1))


class RAGFAS(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = GraphFeatureGenerator()
        self.classifier = MultiLevelTokenClassifier()
        self.discriminator = DomainDiscriminator()
        self.region_head = RegionPredictionNetwork()

    def forward(self, x, edge_index, batch, drop=False, idx=None):
        t = self.generator(x, edge_index, batch, drop)
        stacked = torch.stack([t["h_final"], t["h1"], t["h5"], t["h3"]], dim=2)
        stacked = nn.functional.dropout(stacked, p=0.5, training=self.training)
        logit = self.classifier(stacked)
        domain = self.discriminator(t["z1"], edge_index, drop)
        region = self.region_head(t["z1"], edge_index, idx)
        return logit, domain, region


class DenseNet121LowFE(nn.Module):
    """D121-LowFE (paper Sec. 3.1): DenseNet-121 first two blocks, GAP+concat [512|128]."""

    def __init__(self, pretrained=False):
        super().__init__()
        import torchvision
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        features = torchvision.models.densenet121(weights=weights).features
        self.stem = nn.Sequential(features.conv0, features.norm0, features.relu0, features.pool0)
        self.block1 = features.denseblock1.layers[:2]
        self.transition1 = features.transition1
        self.block2 = features.denseblock2

    def forward(self, patches):
        x = (patches / 127.5) - 1.0
        x = self.stem(x)
        for layer in self.block1:
            x = layer(x)
        low = x
        x = self.block2(self.transition1(x))
        return torch.cat([x.mean(dim=(2, 3)), low.mean(dim=(2, 3))], dim=1)


# ============================== data helpers ===============================
def facemesh_edge_index() -> torch.Tensor:
    import mediapipe as mp
    edges = []
    for a, b in mp.solutions.face_mesh.FACEMESH_TESSELATION:
        edges += [[a, b], [b, a]]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def cut_patches(image_rgb: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """Zero-pad the 256x256 face and cut the 468 40x40 patches (stage-1 tail)."""
    pts = np.clip(landmarks, 0, FACE) + 20
    padded = np.pad(image_rgb, ((20, 20), (20, 20), (0, 0)), mode="constant")
    pts = np.round(pts).astype(int)
    out = np.empty((len(pts), PATCH, PATCH, 3), dtype=np.float32)
    for i, (x, y) in enumerate(pts):
        out[i] = padded[y - 20:y + 20, x - 20:x + 20]
    return out


def make_stage1_fn(face_mesh, image_rgb, synthetic_landmarks):
    """Callable for stage [1]: FaceMesh detection + 468-patch cutting."""
    def stage1():
        results = face_mesh.process(image_rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            landmarks = np.array([(p.x, p.y) for p in lm]) * FACE
        else:
            landmarks = synthetic_landmarks  # keeps the pipeline running
        return cut_patches(image_rgb, landmarks)
    return stage1


# ================================ FLOPs ====================================
def flops_of(fn, *args, **kwargs) -> int:
    """Exact FLOPs of ``fn(*args)`` via torch.utils.flop_counter (torch>=2.0).

    Counts every conv / mm / bmm (1 MAC = 2 FLOPs).  Element-wise ops and
    sparse graph *propagation* are not cataloged by PyTorch; the caller adds
    the propagation term analytically.
    """
    from torch.utils.flop_counter import FlopCounterMode
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # uncataloged ops would spam
        mode = FlopCounterMode(display=False)
        with mode:
            fn(*args, **kwargs)
        return mode.get_total_flops()


def sparse_propagation_macs() -> int:
    """Analytic MACs of the ChebConv Laplacian propagation (invisible to the counter).

    ChebConv(K=2) performs 2 sparse matmuls of nnz x C_in per layer
    (T_1 = L X and T_2 = 2 L T_1); nnz ~= E + N (symmetric normalisation with
    self-loops).  GraphSAGE aggregation is a scatter (no matmul) -> 0.
    """
    cheb_cins = [NODE_FEATURES] + [HIDDEN] * 6 + [HIDDEN * 2] + [HIDDEN] * 2
    nnz = facemesh_edge_index().size(1) + N_NODES
    return sum(2 * nnz * c for c in cheb_cins)


def print_flops(lowfe, model, patches, edge_index, batch, idx):
    print("\n" + "=" * 76)
    print("FLOPs / MACs per image (single forward, N=468 nodes)")
    print("=" * 76)

    feats = lowfe(patches)
    f_cnn = flops_of(lowfe, patches)

    def gnn():
        model(feats, edge_index, batch, False, idx)

    f_gnn = flops_of(gnn)
    f_prop = sparse_propagation_macs() * 2          # MACs -> FLOPs
    total = f_cnn + f_gnn + f_prop

    print(f"  [2] D121-LowFE CNN          : {f_cnn / 1e9:.3f} GFLOPs  ({f_cnn / 2e9:.3f} GMACs)")
    print(f"  [3] GFG + heads (graph)     : {f_gnn / 1e9:.3f} GFLOPs  ({f_gnn / 2e9:.3f} GMACs)")
    print(f"      + sparse propagation    : {f_prop / 1e9:.3f} GFLOPs  (analytic, ChebConv L@X)")
    print(f"  TOTAL torch pipeline        : {total / 1e9:.3f} GFLOPs  ({total / 2e9:.3f} GMACs)")
    print("  [1] MediaPipe FaceMesh      : not a torch graph (C++ inference on CPU)")

    print("\n  per-module FLOPs table (torch.utils.flop_counter):")
    from torch.utils.flop_counter import FlopCounterMode
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mode = FlopCounterMode(display=True)
        with mode:
            lowfe(patches)
            model(feats, edge_index, batch, False, idx)


def profile_generic(fn, *args) -> dict:
    """Reusable helper: profile ANY torch callable (drop baseline models in here)."""
    fn(*args)                                   # warm-up (cudnn autotune, caches)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return {"ms": (time.perf_counter() - t0) * 1e3,
            "gflops": flops_of(fn, *args) / 1e9}


# ================================ latency ==================================
def time_gpu_median(fn, n_warmup, n_iter) -> float:
    """Median GPU time of fn() measured with CUDA events (ms)."""
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(n_iter):
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        samples.append(e0.elapsed_time(e1))
    return statistics.median(samples)


def time_cpu_median(fn, n_warmup, n_iter) -> float:
    for _ in range(n_warmup):
        fn()
    samples = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description="RAGFAS FLOPs + stage latency breakdown")
    parser.add_argument("--image", default=None, help="photo with a face (recommended)")
    parser.add_argument("--n-warmup", type=int, default=N_WARMUP)
    parser.add_argument("--n-iter", type=int, default=N_ITER)
    parser.add_argument("--pretrained", action="store_true",
                        help="load ImageNet weights into D121-LowFE (compute-identical)")
    parser.add_argument("--no-fp16", action="store_true", help="skip the fp16 run")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the GPU stages (experimental)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)                 # pure-inference benchmark
    dev_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"

    print("=" * 76)
    print(f"RAGFAS efficiency analysis | device: {dev_name} | torch {torch.__version__}")
    print("=" * 76)

    # lazily pip-install whatever the Colab image is missing
    _ensure_import("torch_geometric")
    _ensure_import("mediapipe")

    # ------------------------- MediaPipe + image -------------------------
    import mediapipe as mp
    if args.image:
        import cv2
        img = cv2.imread(args.image)
        assert img is not None, f"cannot read {args.image}"
        image_rgb = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (FACE, FACE))
    else:
        image_rgb = np.random.randint(0, 255, (FACE, FACE, 3), dtype=np.uint8)
        print("NOTE: no --image given; MediaPipe will likely miss the synthetic face "
              "(the stage is still timed). Pass a real face photo for realistic numbers.")

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
    linspace = np.linspace(0, FACE, N_NODES)
    synthetic = np.stack([linspace, linspace], axis=1).astype(np.float64)
    stage1 = make_stage1_fn(face_mesh, image_rgb, synthetic)

    # ----------------------------- models --------------------------------
    torch.manual_seed(0)
    lowfe = DenseNet121LowFE(pretrained=args.pretrained).to(device).eval()
    model = RAGFAS().to(device).eval()
    edge_index = facemesh_edge_index().to(device)
    batch = torch.zeros(N_NODES, dtype=torch.long, device=device)
    idx = torch.randperm(N_NODES)[:N_REGION_SAMPLES].to(device)

    params = {"D121-LowFE CNN": lowfe, "GFG (Cheb GCN)": model.generator,
              "Classifier": model.classifier, "Discriminator": model.discriminator,
              "RPN": model.region_head}
    params = {n: sum(p.numel() for p in m.parameters()) for n, m in params.items()}
    print("\nparameters per stage:")
    for n, v in params.items():
        print(f"  {n:<22}: {v / 1e6:.2f}M")
    print(f"  {'TOTAL':<22}: {sum(params.values()) / 1e6:.2f}M")
    print(f"  graph: {N_NODES} nodes, {edge_index.size(1)} directed edges (FACEMESH_TESSELATION)")

    # ------------------------------ FLOPs --------------------------------
    patches = torch.from_numpy(
        stage1().transpose(0, 3, 1, 2).copy()).to(device)   # [468, 3, 40, 40]
    print_flops(lowfe, model, patches, edge_index, batch, idx)

    # ----------------------------- latency -------------------------------
    print("\n" + "=" * 76)
    print(f"median latency per image ({args.n_iter} iterations, batch = 1)")
    print("=" * 76)

    for precision, dt in (("fp32", torch.float32), ("fp16", torch.float16)):
        if precision == "fp16" and (args.no_fp16 or device.type != "cuda"):
            continue
        if precision == "fp16":
            lowfe, model = lowfe.half(), model.half()
        patches_t = patches.to(dt)
        feats = lowfe(patches_t)

        def cnn():
            lowfe(patches_t)

        def gnn():
            model(feats, edge_index, batch, False, idx)

        if args.compile:
            cnn, gnn = torch.compile(cnn), torch.compile(gnn)

        t_mp = time_cpu_median(stage1, args.n_warmup, args.n_iter)
        t_cnn = time_gpu_median(cnn, args.n_warmup, args.n_iter)
        t_gnn = time_gpu_median(gnn, args.n_warmup, args.n_iter)

        # pipelined end-to-end (no per-stage sync: CPU stage overlaps GPU work)
        def full():
            p = stage1()
            nchw = np.ascontiguousarray(p.transpose(0, 3, 1, 2))   # NHWC -> NCHW
            f = lowfe(torch.from_numpy(nchw).to(device=device, dtype=dt,
                                                 non_blocking=True))
            model(f, edge_index, batch, False, idx)

        for _ in range(args.n_warmup):
            full()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.n_iter):
            full()
        torch.cuda.synchronize()
        t_e2e = (time.perf_counter() - t0) / args.n_iter * 1e3

        print(f"\n--- {precision} ---")
        print(f"  [1] MediaPipe + patches (CPU) : {t_mp:7.2f} ms")
        print(f"  [2] D121-LowFE CNN (GPU)      : {t_cnn:7.2f} ms")
        print(f"  [3] GNN + heads (GPU)         : {t_gnn:7.2f} ms")
        print(f"  stage sum (sequential)        : {t_mp + t_cnn + t_gnn:7.2f} ms"
              f"  -> {1e3 / (t_mp + t_cnn + t_gnn):6.1f} FPS")
        print(f"  pipelined end-to-end          : {t_e2e:7.2f} ms"
              f"  -> {1e3 / t_e2e:6.1f} FPS   (paper metric, MediaPipe included)")

    print("\nMediaPipe note: the Python solutions API executes on CPU (no GPU "
          "delegate), so its cost is precision-independent.  In the pipelined "
          "loop the CPU stage overlaps the GPU stages, which is why the "
          "end-to-end FPS can exceed 1/(stage sum).")
    face_mesh.close()


if __name__ == "__main__":
    main()