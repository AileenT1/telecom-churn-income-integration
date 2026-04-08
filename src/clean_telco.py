"""
Clean the main telco churn table: standardize names, build binary churn target,
drop leakage columns, coerce key numerics. Writes telco_clean.parquet for tests and modeling.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Must stay in sync with tests/test_leakage.py
LEAK_KEYWORDS = ("churn_reason", "churn_category", "churn_score", "customer_status")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def coerce_yes_no_to_int(s: pd.Series) -> pd.Series:
    s2 = s.astype(str).str.strip().str.lower()
    return (s2 == "yes").astype(int)


def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns whose names indicate target leakage; never drop target `churn`."""
    df = df.copy()
    to_drop: list[str] = []
    for c in df.columns:
        if c == "churn":
            continue
        cl = c.lower().replace(" ", "_")
        if any(k in cl for k in LEAK_KEYWORDS):
            to_drop.append(c)
    return df.drop(columns=to_drop, errors="ignore")


def _build_churn_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary `churn` (0/1) from Churn Value or Churn Label; drop redundant raw churn cols."""
    df = df.copy()
    if "churn_value" in df.columns:
        df["churn"] = pd.to_numeric(df["churn_value"], errors="coerce").fillna(0).astype(int)
        df["churn"] = df["churn"].clip(0, 1)
        drop_extra = [c for c in ("churn_label", "churn_value") if c in df.columns]
        df = df.drop(columns=drop_extra, errors="ignore")
    elif "churn_label" in df.columns:
        df["churn"] = coerce_yes_no_to_int(df["churn_label"])
        df = df.drop(columns=["churn_label"], errors="ignore")
    elif "churn" in df.columns:
        if df["churn"].dtype == object:
            df["churn"] = coerce_yes_no_to_int(df["churn"])
        else:
            df["churn"] = df["churn"].astype(int)
    else:
        raise ValueError(
            "Expected churn column: churn_value, churn_label, or churn after standardizing names."
        )
    return df


def _coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ("total_charges", "monthly_charges", "tenure_months"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_telco(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for the main customer-level telco CSV.

    Order: standardize → binary target `churn` → drop leakage → numeric coercion.
    """
    df = standardize_columns(df_raw)
    df = _build_churn_target(df)
    df = drop_leakage_columns(df)
    df = _coerce_numeric_columns(df)
    return df


def write_telco_clean(out_path: Path | None = None) -> Path:
    """Load raw telco, clean, write parquet. Returns path written."""
    from src.config import TELCO_CLEAN_PATH
    from src.ingest_kaggle import load_telco_raw

    path = out_path or TELCO_CLEAN_PATH
    df_raw = load_telco_raw()
    df_clean = clean_telco(df_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(path, index=False)
    return path


def main() -> None:
    p = argparse.ArgumentParser(description="Clean telco churn CSV → telco_clean.parquet")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet path (default: data_processed/telco_clean.parquet)",
    )
    args = p.parse_args()
    out = write_telco_clean(out_path=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    main()
