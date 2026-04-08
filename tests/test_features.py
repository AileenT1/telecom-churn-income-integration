"""Feature engineering: row count preserved, expected columns."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TELCO_WITH_FEATURES_PATH, TELCO_WITH_INCOME_PATH
from src.features import (
    add_income_quartile,
    add_price_sensitivity,
    add_tenure_bucket,
    featurize,
)


def test_add_tenure_bucket():
    df = pd.DataFrame({"tenure_months": [0, 7, 13, 30, 60, 80]})
    out = add_tenure_bucket(df)
    assert "tenure_bucket" in out.columns
    assert out["tenure_bucket"].tolist() == ["0-6", "6-12", "12-24", "24-48", "48-72", "72+"]


def test_add_price_sensitivity():
    df = pd.DataFrame({"monthly_charges": [50.0], "tenure_months": [9]})
    out = add_price_sensitivity(df)
    assert np.isclose(out["price_sensitivity_proxy"].iloc[0], 50.0 / 10.0)


def test_add_income_quartile_constant():
    df = pd.DataFrame({"median_household_income": [80000.0] * 10})
    out = add_income_quartile(df)
    assert (out["income_quartile"] == "Q1").all()


def test_featurize_preserves_rows():
    df = pd.DataFrame(
        {
            "tenure_months": [12, 24],
            "median_household_income": [70_000.0, 90_000.0],
            "monthly_charges": [50.0, 60.0],
        }
    )
    out = featurize(df)
    assert len(out) == 2
    assert {"tenure_bucket", "income_quartile", "price_sensitivity_proxy"} <= set(out.columns)


@pytest.fixture(scope="module")
def features_df():
    if not TELCO_WITH_FEATURES_PATH.exists():
        pytest.skip(f"Run: python src/features.py — missing {TELCO_WITH_FEATURES_PATH}")
    return pd.read_parquet(TELCO_WITH_FEATURES_PATH)


def test_features_parquet_schema(features_df):
    need = {"tenure_bucket", "income_quartile", "price_sensitivity_proxy", "churn"}
    assert need <= set(features_df.columns)


def test_features_same_length_as_merged(features_df):
    if not TELCO_WITH_INCOME_PATH.exists():
        pytest.skip("merged parquet missing")
    n_m = len(pd.read_parquet(TELCO_WITH_INCOME_PATH))
    assert len(features_df) == n_m
