"""evaluate.py: metrics on synthetic binary problem."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluate import (
    compare_models,
    evaluate_binary_classifier,
    f1_at_best_threshold,
)


def test_f1_at_best_threshold_perfect():
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.8, 0.9])
    assert f1_at_best_threshold(y, s) == 1.0


def test_evaluate_binary_classifier_smoke():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.standard_normal(80)})
    y = (X["a"] + rng.standard_normal(80) * 0.5 > 0).astype(int)
    m = LogisticRegression(max_iter=500).fit(X, y)
    out = evaluate_binary_classifier(m, X, y, "test")
    assert "roc_auc" in out and "pr_auc" in out and "log_loss" in out
    assert 0 <= out["roc_auc"] <= 1


def test_compare_models():
    rows = [
        {"name": "m1", "roc_auc": 0.8, "pr_auc": 0.4, "log_loss": 0.5, "f1_best_threshold": 0.6},
        {"name": "m2", "roc_auc": 0.7, "pr_auc": 0.5, "log_loss": 0.6, "f1_best_threshold": 0.55},
    ]
    df = compare_models(rows)
    assert len(df) == 2
    assert "m1" in df.index
