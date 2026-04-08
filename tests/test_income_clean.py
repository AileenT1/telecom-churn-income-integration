import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.clean_income import clean_income, normalize_state_to_abbrev
from src.config import INCOME_CLEAN_PATH, INCOME_RAW_CSV


@pytest.mark.parametrize(
    "name,expected",
    [
        ("California", "CA"),
        ("california", "CA"),
        ("CA", "CA"),
        ("District of Columbia", "DC"),
        ("Puerto Rico", "PR"),
    ],
)
def test_normalize_state_to_abbrev(name, expected):
    assert normalize_state_to_abbrev(name) == expected


def test_clean_income_columns_and_shape():
    df_raw = pd.DataFrame(
        {
            "state_name": ["California", "Texas"],
            "median_household_income": [90000, 70000],
            "state_fips": ["06", "48"],
        }
    )
    out = clean_income(df_raw)
    assert list(out.columns) == [
        "state_fips",
        "state_name",
        "state_abbrev",
        "median_household_income",
    ]
    assert len(out) == 2
    assert out["state_abbrev"].tolist() == ["CA", "TX"]


@pytest.fixture(scope="module")
def income_clean_df():
    if not INCOME_CLEAN_PATH.exists():
        pytest.skip(
            f"Income clean not found: {INCOME_CLEAN_PATH}. Run clean_income first."
        )
    return pd.read_parquet(INCOME_CLEAN_PATH)


def test_income_parquet_schema(income_clean_df):
    assert set(income_clean_df.columns) == {
        "state_fips",
        "state_name",
        "state_abbrev",
        "median_household_income",
    }
    assert income_clean_df["state_abbrev"].nunique() == len(income_clean_df)
    assert income_clean_df["median_household_income"].notna().all()


def test_raw_csv_exists_for_integration():
    """Integration path: raw ACS file exists when user has run ingest."""
    if not INCOME_RAW_CSV.exists():
        pytest.skip("No raw ACS CSV; run ingest_acs_income.py first.")
