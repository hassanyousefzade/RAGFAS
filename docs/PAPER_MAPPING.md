# Paper ↔ Code Mapping

Every module of this repository corresponds to a specific section/equation of
the paper *"Region-Aware Generalized Face Anti-Spoofing"* (ICASSP 2027).

| Paper element | Section / Eq. | Code |
|---|---|---|
| Facial Graph Representation Extractor (FGRE): MediaPipe alignment, 256×256 crop, 40×40 landmark patches | Sec. 3.1 | `tools/crop_faces.py`, `tools/build_graphs.py`, `ragfas/models/lowfe.py` |
| D121-LowFE: ImageNet DenseNet-121, first two blocks, GAP + concat → node features `X ∈ R^{N×K}`, K = 640 | Sec. 3.1, Eq. 1 | `ragfas/models/lowfe.py` (`DenseNet121LowFE`) |
| Canonical FACEMESH_TESSELATION topology (undirected, fixed geometric prior) | Sec. 3.1, Table 4 | `ragfas/data/topology.py` (`facemesh_edge_index`) |
| Graph Feature Generator (GFG): 8× ChebConv(order 2) + BatchNorm + ReLU + DropNode, skip concat of blocks 2 & 4 before block 5 | Sec. 3.2 | `ragfas/models/generator.py` (`GraphFeatureGenerator`) |
| Four tapped representations `{Z1, Z2, Z3, Z4}` after blocks 2, 3, 5 and the final readout | Sec. 3.2, Eq. 2 | `GraphFeatureGenerator.forward` returns `z1, h1, h3, h5, h_final` |
| Classifier: 128 tokens × 4 levels → embed 8, transformer (8 heads) → single logit + sigmoid | Sec. 3.3, Eq. 3 | `ragfas/models/classifier.py` (`MultiLevelTokenClassifier`) |
| Binary cross-entropy spoof loss `L_Cls` | Sec. 3.3, Eq. 4 | `ragfas/losses.py` (`RAGFASCriterion.cls_criterion`) |
| Node-level adversarial learning: domain discriminator `DG(Z1, A)` predicting the domain of every node, GraphSAGE layers + per-node dense + softmax | Sec. 3.4, Eq. 5-6 | `ragfas/models/discriminator.py` (`DomainDiscriminator`, `pooling="node"`) |
| Focal adversarial loss `L_Adv = 1/N Σ (1-p_y)^γ (-log p_y)`, γ = 2 | Sec. 3.4, Eq. 7 | `ragfas/losses.py` (`FocalLoss`) |
| Gradient Reversal Layer, schedule `λ = 2/(1+exp(-α i/N)) - 1`, α = 10 | Sec. 3.4 | `ragfas/models/layers.py` (`GradientReversal`) |
| Facial region classification (RPN): 7 regions — Nose (106), Lips (96), Eyes (94), Right cheek/eyebrow (72), Left cheek/eyebrow (34), Jaw/contour (51), Pupils/forehead (15) | Sec. 3.5 | `ragfas/data/regions.py`, `ragfas/models/region_head.py` |
| Region self-supervised loss `L_Self` (cross-entropy over 64 sampled nodes) | Sec. 3.5, Eq. 8 | `ragfas/losses.py` (`RAGFASCriterion.region_criterion`), `ragfas/data/dataset.py` (`sample_node_indices`) |
| Total loss `L_Total = L_Cls + L_Adv + L_Self` (unweighted) | Sec. 3.5 | `ragfas/losses.py` (`RAGFASCriterion`, `LossWeights` all 1.0) |
| Optimisation: Adam lr 1e-4, batch 64, DropNode p = 0.3, 500 epochs, 64-of-468 node sampling | Sec. 4.1 | `configs/default.yaml`, `ragfas/engine/trainer.py` |
| LOO + limited-source protocols (O/C/M/I) | Sec. 4.1, Tables 1-2 | `ragfas/data/protocols.py`, `tools/make_lists.py` |
| Frame-level HTER/AUC; person-level averaging | Sec. 4.1 | `ragfas/metrics.py`, `ragfas/engine/evaluator.py` |
| "Global" adversarial ablation (mean-pool + GRL + MLP) — −7.02 pp HTER vs node-level | Table 3 | `ragfas/models/discriminator.py` (`pooling="global_mlp"`) |
| ViT / TCN / K-NN graph ablations | Table 4 | legacy scripts (`ablition_transformer_TCN.py`, `knn-grpah.py` in the research tree) |
| End-to-end 198 FPS incl. MediaPipe, single `[468, 3, 40, 40]` LowFE pass | Sec. 4.5 | `tools/benchmark_fps.py` |

## Deliberate deviations of the refactored code

The original research scripts and the paper text differ in three details.
The refactored code defaults to the **paper** and keeps the original
behaviour behind the `--legacy` flag (see README):

1. **Discriminator granularity** — the paper (Eq. 6-7) predicts the domain of
   every node; the original script pooled node features before the dense
   layer. Default: per-node (`pooling: node`). Legacy: `pooling: graph`.
2. **DropNode in the first two blocks** — the paper states DropNode is zero
   in the first two blocks; the original script applied it there on ~50% of
   batches (Bernoulli gate). Default: `first_two_blocks: false`.
3. **RPN convolutions** — the paper describes ChebConv layers; the original
   script used GraphSAGE. Default: `conv: cheb`. Legacy: `conv: sage`
   (required to strictly load the released checkpoints).
