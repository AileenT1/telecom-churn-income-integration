"""
Discover and load the main customer-level telco churn CSV from data_raw/kaggle_telco/.
Internal/Kaggle data lives in data_raw/kaggle_telco/; this module documents the single customer-level source.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Preferred main file: single customer-level table with churn label and state
MAIN_TELCO_FILENAME = "Telco_customer_churn.csv"


def get_telco_raw_path(data_raw_dir: Path | None = None) -> Path:
    """
    Return the path to the main customer-level telco CSV in data_raw/.

    Uses MAIN_TELCO_FILENAME if present; otherwise raise error
    """
    from src.config import DATA_RAW_TELCO

    root = data_raw_dir or DATA_RAW_TELCO
    if not root.exists():
        raise FileNotFoundError(f"Raw data directory not found: {root}")

    preferred = root / MAIN_TELCO_FILENAME
    if preferred.exists():
        return preferred


    raise FileNotFoundError(
        f"No main telco CSV found in {root}. Expected {MAIN_TELCO_FILENAME} or a CSV with Churn and Customer columns."
    )


def load_telco_raw(data_raw_dir: Path | None = None) -> pd.DataFrame:
    """Load the main customer-level telco DataFrame from data_raw/kaggle_telco/."""
    path = get_telco_raw_path(data_raw_dir)
    return pd.read_csv(path)


def main() -> None:
    """Print the path to the main telco CSV and confirm it loads."""
    path = get_telco_raw_path()
    print(f"Main telco source: {path}")
    df = load_telco_raw()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:15]}..." if len(df.columns) > 15 else f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    import sys
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    main()
