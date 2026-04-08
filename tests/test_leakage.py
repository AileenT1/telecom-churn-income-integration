"""
Assert no leakage columns in cleaned telco data.
Run after Step 4 (clean_telco) has produced telco_clean.parquet.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clean_telco import LEAK_KEYWORDS
from src.config import TELCO_CLEAN_PATH


@pytest.fixture(scope="module")
def cleaned_df():
    """Load cleaned telco data; skip if not yet built."""
    if not TELCO_CLEAN_PATH.exists():
        pytest.skip(f"Cleaned data not found: {TELCO_CLEAN_PATH}. Run clean pipeline first.")
    import pandas as pd
    return pd.read_parquet(TELCO_CLEAN_PATH)


def test_no_leakage_columns(cleaned_df):
    """Cleaned data must not contain leakage columns (churn reason, category, score, etc.)."""
    cols_lower = [c.lower().replace(" ", "_") for c in cleaned_df.columns]
    found = [
        cleaned_df.columns[i]
        for i, c in enumerate(cols_lower)
        if any(k in c for k in LEAK_KEYWORDS)
    ]
    assert not found, f"Leakage columns must be dropped: {found}"
