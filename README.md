# RAGFAS — Region-Aware Generalized Face Anti-Spoofing

Official PyTorch implementation of the paper
**"Region-Aware Generalized Face Anti-Spoofing"**
(H. Yousefzade, A. Akhavanpour, M. Farajian, M. Sabokrou).

RAGFAS is a graph neural network (GNN) framework for domain-generalized face
anti-spoofing. Each graph node is a local face patch anchored at a MediaPipe
landmark and the edges follow the canonical face-mesh topology. Two
node-level objectives augment the spoof classifier: **adversarial learning**
that aligns each facial region across domains, and a **region-based
self-supervised task** that groups anatomically similar landmarks. On the
standard four-dataset leave-one-out protocol RAGFAS reaches an average
**HTER 4.21% / AUC 98.63%** at **198 FPS** with only **5.5M parameters**.

---

## Repository layout

```
ragfas/
├── configs/
│   ├── default.yaml            # full default configuration
│   └── protocols/              # one YAML per paper protocol (Table 1 / Table 2)
├── ragfas/                     # core package (mirrors the paper sections)
│   ├── data/
│   │   ├── dataset.py          # graph list datasets + 64-node region subsampling
│   │   ├── protocols.py        # OCI->M, OMI->C, OCM->I, ICM->O, MI->C, MI->O
│   │   ├── regions.py          # 7 anatomical region targets (Sec. 3.5)
│   │   └── topology.py         # canonical FACEMESH_TESSELATION edges (Sec. 3.1)
│   ├── models/
│   │   ├── lowfe.py            # D121-LowFE patch encoder (Sec. 3.1, FGRE)
│   │   ├── generator.py        # GFG: 8x ChebConv(2) + BN + DropNode (Sec. 3.2)
│   │   ├── classifier.py       # token transformer + BCE (Sec. 3.3, Eq. 3-4)
│   │   ├── discriminator.py    # node-level adversarial head + GRL (Sec. 3.4, Eq. 6-7)
│   │   ├── region_head.py      # RPN: 7-region classification (Sec. 3.5, Eq. 8)
│   │   └── ragfas.py           # full multi-task model (Fig. 2)
│   ├── losses.py               # focal adversarial + BCE + region CE (total loss)
│   ├── metrics.py              # frame-level HTER/AUC protocol (Sec. 4.1)
│   ├── engine/                 # Trainer / Evaluator
│   └── utils/                  # config, logging, seeding, path helpers
├── tools/
│   ├── crop_faces.py           # stage 1: align + crop + MediaPipe landmarks
│   ├── build_graphs.py         # stage 2: 40x40 patches -> D121-LowFE -> .pt graphs
│   ├── make_lists.py           # build train/test list files per protocol
│   └── benchmark_fps.py        # end-to-end inference speed (Sec. 4.5)
├── train.py                    # training entry point
├── evaluate.py                 # inference-only evaluation of a checkpoint
└── assets/landmark_colors.npy  # landmark -> region assignment (468 entries)
```

## Installation

```bash
conda create -n ragfas python=3.10 -y
conda activate ragfas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install -r requirements.txt
```

## Data preparation

### 1. Face cropping + landmarks (FGRE stage 1)

```bash
python tools/crop_faces.py --input <dataset>/frames --output <dataset>/preposess --stage crop
```

Produces, per frame: `cropped_XXXX.jpg` (256x256 aligned crop) and
`cropped_XXXX.npy` (468 scaled MediaPipe landmarks).

### 2. Graph construction (FGRE stage 2)

```bash
python tools/build_graphs.py --input <dataset>/preposess \
    --list-out data/lists/graph_<dataset>.txt
```

Produces `cropped_XXXX.jpg.pt` graph files with
`x = [468, 640]` node features (D121-LowFE) and the canonical FACEMESH edges.
Labels are parsed from the video folder name (`live`/`real` -> 1,
`spoof`/`attack` -> 0).

### 3. Protocol list files

```bash
python tools/make_lists.py --data-root /data --protocol OMI_to_C --out-dir data/lists
```

Builds `graph_train_OMI.txt` (sources) and `graph_test_C.txt` (target) for
any of the six protocols: `OCI_to_M`, `OMI_to_C`, `OCM_to_I`, `ICM_to_O`,
`MI_to_C`, `MI_to_O`.

## Training

```bash
# Leave-one-out (paper Table 1)
python train.py --config configs/protocols/OMI_to_C.yaml \
    --set data.train_list=data/lists/graph_train_OMI.txt \
          data.test_list=data/lists/graph_test_C.txt

# Limited-source (paper Table 2)
python train.py --config configs/protocols/MI_to_C.yaml \
    --set data.train_list=data/lists/graph_train_MI.txt \
          data.test_list=data/lists/graph_test_C.txt
```

Training logs, score files (`Scores/`) and per-epoch metrics CSVs are written
to `runs/<protocol>/`. The best checkpoint is selected by test person-level
AUC (configurable via `train.selection_metric`).

## Evaluation

```bash
python evaluate.py --config configs/protocols/MI_to_C.yaml \
    --checkpoint runs/MI_to_C/checkpoints/MI_to_C_best.pth
```

Reports frame- and person-level ACC/HTER/AUC and (when the domain folder
names are recoverable from the paths) per-domain frame metrics.

## Inference speed

```bash
python tools/benchmark_fps.py --config configs/protocols/MI_to_C.yaml \
    --checkpoint runs/MI_to_C/checkpoints/MI_to_C_best.pth --image face.jpg
```

Times the full end-to-end pipeline **including MediaPipe landmark detection**:
all 468 patches are stacked into a single `[468, 3, 40, 40]` tensor and
processed in one parallel D121-LowFE forward pass.

## Configuration notes (`--legacy`)

The refactor keeps two behaviours switchable from the config:

| Setting | Paper (default) | `--legacy` (original scripts) |
|---|---|---|
| `model.discriminator.pooling` | `node` — per-node dense layer + softmax (Eq. 6-7) | `graph` — pooled before the dense layer |
| `model.dropnode.first_two_blocks` | `false` — DropNode zero in the first two blocks | `true` — DropNode also in blocks 1-2 |
| `model.region_head.conv` | `cheb` — ChebConv layers in the RPN | `sage` — GraphSAGE layers |

Both variants share the same parameter names (except the RPN convolutions),
so checkpoints remain loadable across modes; use `--legacy` to continue
training from the originally released checkpoints bit-exactly.

## Citation


