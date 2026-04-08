"""
Feature engineering on merged telco + income: tenure buckets, income quartiles,
optional price-sensitivity proxy. Writes telco_with_features.parquet for modeling.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Tenure bucket edges (months): (0,6], (6,12], ... (72, inf)
TENURE_BIN_EDGES = [-0.001, 6, 12, 24, 48, 72, np.inf]
TENURE_BIN_LABELS = ["0-6", "6-12", "12-24", "24-48", "48-72", "72+"]


def add_tenure_bucket(df: pd.DataFrame, tenure_col: str = "tenure_months") -> pd.DataFrame:
    """Ordinal tenure buckets for EDA and modeling."""
    df = df.copy()
    if tenure_col not in df.columns:
        raise KeyError(f"Missing {tenure_col}")
    df["tenure_bucket"] = pd.cut(
        df[tenure_col],
        bins=TENURE_BIN_EDGES,
        labels=TENURE_BIN_LABELS,
        right=True,
    ).astype(str)
    return df


def add_income_quartile(df: pd.DataFrame, income_col: str = "median_household_income") -> pd.DataFrame:
    """
    Q1–Q4 from state median income on this dataset.
    If all values are identical (e.g. single-state sample), assigns a single label.
    """
    df = df.copy()
    if income_col not in df.columns:
        raise KeyError(f"Missing {income_col}")
    s = df[income_col]
    nuniq = s.nunique(dropna=True)
    if nuniq <= 1:
        df["income_quartile"] = "Q1"
    else:
        df["income_quartile"] = pd.qcut(
            s,
            q=4,
            labels=["Q1", "Q2", "Q3", "Q4"],
            duplicates="drop",
        ).astype(str)
    return df


def add_price_sensitivity(
    df: pd.DataFrame,
    charges_col: str = "monthly_charges",
    tenure_col: str = "tenure_months",
) -> pd.DataFrame:
    """Simple proxy: monthly spend relative to tenure (avoid div by zero)."""
    df = df.copy()
    if charges_col not in df.columns or tenure_col not in df.columns:
        raise KeyError("Need monthly_charges and tenure_months")
    df["price_sensitivity_proxy"] = df[charges_col] / (df[tenure_col].astype(float) + 1.0)
    return df


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature steps (idempotent on column names)."""
    df = add_tenure_bucket(df)
    df = add_income_quartile(df)
    df = add_price_sensitivity(df)
    return df


def write_telco_with_features(
    merged_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Load merged parquet, featurize, write telco_with_features.parquet."""
    from src.config import TELCO_WITH_FEATURES_PATH, TELCO_WITH_INCOME_PATH

    src = merged_path or TELCO_WITH_INCOME_PATH
    dst = out_path or TELCO_WITH_FEATURES_PATH

    if not src.exists():
        raise FileNotFoundError(f"Merged data not found: {src}. Run merge_income first.")

    df = pd.read_parquet(src)
    df = featurize(df)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dst, index=False)
    return dst


def main() -> None:
    from src.config import TELCO_WITH_FEATURES_PATH

    p = argparse.ArgumentParser(description="Featurize merged telco+income → telco_with_features.parquet")
    p.add_argument("--in", dest="in_path", type=Path, default=None, help="Input telco_with_income.parquet")
    p.add_argument("--out", type=Path, default=None, help=f"Output (default: {TELCO_WITH_FEATURES_PATH})")
    args = p.parse_args()
    out = write_telco_with_features(merged_path=args.in_path, out_path=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    main()
