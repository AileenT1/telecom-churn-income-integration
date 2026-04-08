"""interpret.py: preprocessed permutation importance shape and CSV helpers."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.interpret import (
    perm_importance_to_dataframe,
    permutation_importance_preprocessed,
    transform_preprocessed,
)
from src.train_models import build_preprocessor


@pytest.fixture
def tiny_pipeline():
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            "num_a": rng.standard_normal(n),
            "cat_x": rng.choice(["p", "q"], size=n),
            "cat_y": rng.choice(["u", "v", "w"], size=n),
        }
    )
    y = ((df["num_a"] + (df["cat_x"] == "p").astype(int) * 0.5 + rng.standard_normal(n) * 0.3) > 0).astype(
        int
    )
    pre = build_preprocessor(df)
    pipe = Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                HistGradientBoostingClassifier(
                    max_depth=3,
                    max_iter=30,
                    random_state=0,
                ),
            ),
        ]
    )
    pipe.fit(df, y)
    X_test = df.iloc[:40].copy()
    y_test = y[:40]
    return pipe, X_test, y_test


def test_transform_preprocessed_dense_and_names(tiny_pipeline):
    pipe, X_test, _ = tiny_pipeline
    Xt, names = transform_preprocessed(pipe, X_test)
    assert Xt.ndim == 2
    assert not hasattr(Xt, "toarray")
    assert len(names) == Xt.shape[1]


def test_permutation_importance_preprocessed_smoke(tiny_pipeline):
    pipe, X_test, y_test = tiny_pipeline
    result, names = permutation_importance_preprocessed(
        pipe,
        X_test,
        y_test,
        n_repeats=3,
        random_state=1,
        n_jobs=1,
    )
    df = perm_importance_to_dataframe(result, names)
    assert len(df) == len(names)
    assert "importance_mean" in df.columns


def test_perm_importance_to_dataframe_sorted(tiny_pipeline):
    pipe, X_test, y_test = tiny_pipeline
    result, names = permutation_importance_preprocessed(
        pipe, X_test, y_test, n_repeats=2, random_state=2, n_jobs=1
    )
    df = perm_importance_to_dataframe(result, names)
    assert df["importance_mean"].is_monotonic_decreasing
