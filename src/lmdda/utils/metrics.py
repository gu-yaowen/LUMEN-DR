from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        "auroc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float("nan"),
        "aupr": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    return out

