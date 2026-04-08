"""train_ab_models on real featurized parquet if present."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TELCO_WITH_FEATURES_PATH
from src.train_models import MODEL_A_EXCLUDE, _feature_columns_model_a, save_ab_models, train_ab_models


def test_model_a_excludes_income_cols():
    if not TELCO_WITH_FEATURES_PATH.exists():
        pytest.skip("telco_with_features.parquet missing")
    df = pd.read_parquet(TELCO_WITH_FEATURES_PATH)
    ca = _feature_columns_model_a(df)
    for c in MODEL_A_EXCLUDE:
        assert c not in ca


def test_train_ab_models_smoke():
    if not TELCO_WITH_FEATURES_PATH.exists():
        pytest.skip("telco_with_features.parquet missing")
    df = pd.read_parquet(TELCO_WITH_FEATURES_PATH)
    out = train_ab_models(df, test_size=0.3)
    assert set(out["models"].keys()) == {"A_logreg", "A_hgb", "B_logreg", "B_hgb"}
    assert len(out["y_test"]) == len(out["X_test_A"])
    assert "median_household_income" in out["columns_B"]
    assert "median_household_income" not in out["columns_A"]


def test_save_ab_models_joblib(tmp_path: Path):
    rng = np.random.default_rng(42)
    n = 400
    df = pd.DataFrame(
        {
            "customerid": np.arange(n),
            "churn": rng.integers(0, 2, size=n),
            "num1": rng.standard_normal(n),
            "cat1": rng.choice(["a", "b", "c"], size=n),
        }
    )
    bundle = train_ab_models(df, test_size=0.25)
    paths = save_ab_models(bundle, out_dir=tmp_path)
    assert set(paths.keys()) == {"A_logreg", "A_hgb", "B_logreg", "B_hgb"}
    for p in paths.values():
        assert p.exists() and p.stat().st_size > 0
