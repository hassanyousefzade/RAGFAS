# -*- coding: utf-8 -*-
"""Frame-level evaluation metrics (paper Sec. 4.1).

``HTER`` and ``AUC`` are computed at the frame level on the target-domain
test set; the score-file format is ``<score> <label>`` per line with
label 1 = live and 0 = spoof.  This is a 1:1 port of the original
``performance.py`` used for every reported number.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


def performances_val(score_filename: str) -> Tuple[float, float, float, float, float, float, float]:
    """ACC / FPR / FRR / HTER / AUC / EER(=err) / TPR from a score file.

    The operating threshold is the one minimising HTER on the given scores
    (standard DG-FAS protocol).
    """
    with open(score_filename, "r") as f:
        lines = f.readlines()

    val_scores, val_labels = [], []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        val_scores.append(float(parts[0]))
        val_labels.append(int(float(parts[1])))

    val_scores = np.array(val_scores)
    val_labels = np.array(val_labels).astype(int)

    # label 1 = live (positive class in this project)
    fpr_arr, tpr_arr, thresholds = roc_curve(val_labels, val_scores, pos_label=1)
    auc_test = auc(fpr_arr, tpr_arr)

    fnr_arr = 1.0 - tpr_arr
    eer_index = np.nanargmin(np.abs(fnr_arr - fpr_arr))
    eer = (fpr_arr[eer_index] + fnr_arr[eer_index]) / 2.0

    hter_arr = (fpr_arr + fnr_arr) / 2.0
    best_idx = np.nanargmin(hter_arr)

    fpr = fpr_arr[best_idx]
    frr = fnr_arr[best_idx]
    hter = (fpr + frr) / 2.0
    test_err = hter

    preds = (val_scores >= thresholds[best_idx]).astype(int)
    val_acc = metrics.accuracy_score(val_labels, preds)
    tpr = tpr_arr[best_idx]

    return val_acc, fpr, frr, hter, auc_test, test_err, tpr


__all__ = ["performances_val"]