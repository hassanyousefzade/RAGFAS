# -*- coding: utf-8 -*-
"""
RAGFAS — Region-Aware Generalized Face Anti-Spoofing.

Official implementation of:

    H. Yousefzade, A. Akhavanpour, M. Farajian, and M. Sabokrou,
    "Region-Aware Generalized Face Anti-Spoofing", ICASSP 2027.

Package layout mirrors the paper sections:

    ragfas.models.lowfe        -> Sec. 3.1  FGRE (D121-LowFE patch encoder)
    ragfas.data.topology       -> Sec. 3.1 (canonical FACEMESH topology)
    ragfas.models.generator    -> Sec. 3.2 (GFG, Chebyshev graph generator)
    ragfas.models.classifier   -> Sec. 3.3 (multi-level token transformer + BCE)
    ragfas.models.discriminator-> Sec. 3.4 (node-level adversarial learning + GRL)
    ragfas.models.region_head  -> Sec. 3.5 (facial region classification, RPN)
    ragfas.losses              -> Sec. 3.4/3.5 (focal adversarial + region CE + total loss)
    ragfas.metrics             -> Sec. 4.1 (frame-level HTER/AUC protocol)
"""

__version__ = "1.0.0"