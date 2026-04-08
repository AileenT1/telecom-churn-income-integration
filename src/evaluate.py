"""
Binary classification metrics: ROC AUC, PR AUC, log loss, F1 at best threshold; model comparison table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    log_loss,
    roc_auc_score,
)


def f1_at_best_threshold(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Max F1 over a grid of classification thresholds on predicted probabilities."""
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores).ravel()
    thresholds = np.linspace(0.01, 0.99, 99)
    best = 0.0
    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        f = f1_score(y_true, y_pred, zero_division=0)
        if f > best:
            best = f
    return float(best)


def evaluate_binary_classifier(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    name: str,
) -> dict[str, Any]:
    """
    ROC AUC, PR AUC (average precision), log loss, F1 at best threshold on ``X_test``, ``y_test``.
    """
    y_test = np.asarray(y_test).ravel()
    y_proba = model.predict_proba(X_test)[:, 1]
    y_proba = np.clip(y_proba, 1e-15, 1.0 - 1e-15)

    out = {
        "name": name,
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "log_loss": float(log_loss(y_test, y_proba, labels=[0, 1])),
        "f1_best_threshold": f1_at_best_threshold(y_test, y_proba),
    }
    return out


def compare_models(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build comparison DataFrame from list of evaluate_binary_classifier outputs."""
    return pd.DataFrame(results).set_index("name")


def save_model_comparison(df: pd.DataFrame, path: Path | None = None) -> Path:
    from src.config import OUTPUTS_TABLES

    out = path or (OUTPUTS_TABLES / "model_comparison.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index(names="name").to_csv(out, index=False)
    return out
