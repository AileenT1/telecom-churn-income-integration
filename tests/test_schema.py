"""
Schema and target validity for cleaned telco data.
Run after Step 4 (clean_telco) has produced telco_clean.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Project root on path when running pytest from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import TELCO_CLEAN_PATH, TARGET_COL


@pytest.fixture(scope="module")
def cleaned_df():
    """Load cleaned telco data; skip if not yet built."""
    if not TELCO_CLEAN_PATH.exists():
        pytest.skip(f"Cleaned data not found: {TELCO_CLEAN_PATH}. Run clean pipeline first.")
    import pandas as pd
    return pd.read_parquet(TELCO_CLEAN_PATH)


def test_churn_binary(cleaned_df):
    """Churn column must be 0 or 1."""
    assert TARGET_COL in cleaned_df.columns, f"Missing target column {TARGET_COL}"
    unique = cleaned_df[TARGET_COL].dropna().unique()
    assert set(unique).issubset({0, 1}), f"Churn should be binary 0/1, got {unique}"


def test_row_count_sanity(cleaned_df):
    """Sanity check: enough rows for modeling."""
    assert cleaned_df.shape[0] > 1000, "Expected > 1000 rows in cleaned data"
