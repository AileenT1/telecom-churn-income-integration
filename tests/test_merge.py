"""
Merge telco + income: many-to-one join, row count preserved, low missing income rate.
Requires data_processed/telco_with_income.parquet from merge_income (or skip).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clean_income import clean_income, load_income_raw_csv
from src.clean_telco import clean_telco
from src.config import (
    ID_COL,
    INCOME_RAW_CSV,
    STATE_COL,
    TARGET_COL,
    TELCO_CLEAN_PATH,
    TELCO_WITH_INCOME_PATH,
)
from src.ingest_kaggle import load_telco_raw
from src.merge_income import merge_telco_income


def test_merge_telco_income_synthetic():
    """Small dataframes: m:1 join, no missing income."""
    df_t = pd.DataFrame(
        {
            "customerid": ["a", "b", "c"],
            "state": ["California", "California", "Texas"],
            TARGET_COL: [0, 1, 0],
        }
    )
    df_i = pd.DataFrame(
        {
            "state_fips": ["06", "48"],
            "state_name": ["California", "Texas"],
            "state_abbrev": ["CA", "TX"],
            "median_household_income": [90000.0, 75000.0],
        }
    )
    out = merge_telco_income(df_t, df_i, state_col="state", max_missing_income_rate=0.05)
    assert len(out) == 3
    assert out[ID_COL if ID_COL in out.columns else "customerid"].nunique() == 3
    assert out["median_household_income"].isna().sum() == 0
    assert (out["state"] == df_t["state"]).all()


def test_merge_end_to_end_from_raw():
    """Full clean + merge on real files if raw data exists."""
    if not TELCO_CLEAN_PATH.exists() or not INCOME_RAW_CSV.exists():
        pytest.skip("Need telco_clean.parquet and ACS raw CSV")

    df_t = clean_telco(load_telco_raw())
    df_i = clean_income(load_income_raw_csv(INCOME_RAW_CSV))
    out = merge_telco_income(df_t, df_i, state_col=STATE_COL)
    assert len(out) == len(df_t)
    assert out["median_household_income"].isna().mean() < 0.05


@pytest.fixture(scope="module")
def merged_df():
    if not TELCO_WITH_INCOME_PATH.exists():
        pytest.skip(
            f"Merged file not found: {TELCO_WITH_INCOME_PATH}. Run: python src/merge_income.py"
        )
    return pd.read_parquet(TELCO_WITH_INCOME_PATH)


def test_merged_row_count_matches_telco(merged_df):
    if not TELCO_CLEAN_PATH.exists():
        pytest.skip("telco_clean.parquet missing")
    n_telco = len(pd.read_parquet(TELCO_CLEAN_PATH))
    assert len(merged_df) == n_telco


def test_no_duplicate_customer_ids(merged_df):
    assert merged_df[ID_COL].nunique() == len(merged_df)


def test_missing_income_rate_low(merged_df):
    miss = merged_df["median_household_income"].isna().mean()
    assert miss < 0.05, f"Missing income rate {miss:.2%} >= 5%"


def test_target_present(merged_df):
    assert TARGET_COL in merged_df.columns
    assert set(merged_df[TARGET_COL].dropna().unique()).issubset({0, 1})
