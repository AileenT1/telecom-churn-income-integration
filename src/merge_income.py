"""
Left-join cleaned telco rows to state-level income on state_abbrev (many-to-one).
Validates merge type and caps missing income rate.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import pandas as pd

# Default max share of rows with missing income after left join (plan §4)
DEFAULT_MAX_MISSING_INCOME_RATE = 0.05


def merge_telco_income(
    df_telco: pd.DataFrame,
    df_income: pd.DataFrame,
    state_col: str = "state",
    max_missing_income_rate: float = DEFAULT_MAX_MISSING_INCOME_RATE,
) -> pd.DataFrame:
    """
    Add state_abbrev from telco state names, left-merge income on state_abbrev.

    Uses ``validate='many_to_one'``: many telco rows per state, one income row per state_abbrev.
    Raises if missing ``median_household_income`` rate exceeds ``max_missing_income_rate``.
    """
    from src.clean_income import normalize_state_to_abbrev

    df = df_telco.copy()
    if state_col not in df.columns:
        raise KeyError(f"Telco state column {state_col!r} not found. Columns: {list(df.columns)}")

    req_inc = {"state_abbrev", "median_household_income"}
    missing_inc = req_inc - set(df_income.columns)
    if missing_inc:
        raise ValueError(f"Income table missing columns: {missing_inc}")

    inc = df_income.drop_duplicates(subset=["state_abbrev"], keep="first")
    dup_n = len(df_income) - len(inc)
    if dup_n:
        warnings.warn(f"Income rows duplicated on state_abbrev: dropped {dup_n} duplicate row(s).")

    df["state_abbrev"] = df[state_col].apply(normalize_state_to_abbrev)
    n_unmapped = df["state_abbrev"].isna().sum()
    if n_unmapped:
        warnings.warn(
            f"Telco rows with unmapped state ({state_col}) → state_abbrev: {n_unmapped} "
            "(income will be NaN for these rows)."
        )

    out = df.merge(
        inc,
        on="state_abbrev",
        how="left",
        validate="many_to_one",
    )

    if len(out) != len(df_telco):
        raise ValueError(
            f"Row count changed after merge: telco {len(df_telco)} vs merged {len(out)} "
            "(expected left join to preserve row count)."
        )

    miss_rate = out["median_household_income"].isna().mean()
    if miss_rate > max_missing_income_rate:
        bad = (
            out.loc[out["median_household_income"].isna(), [state_col, "state_abbrev"]]
            .drop_duplicates()
            .head(15)
        )
        raise ValueError(
            f"Missing median_household_income rate {miss_rate:.2%} exceeds "
            f"{max_missing_income_rate:.2%}. Sample unmatched: {bad.to_dict('records')}"
        )

    return out


def write_telco_with_income(
    telco_path: Path | None = None,
    income_path: Path | None = None,
    out_path: Path | None = None,
    max_missing_income_rate: float = DEFAULT_MAX_MISSING_INCOME_RATE,
) -> Path:
    """Load cleaned parquets, merge, write ``telco_with_income.parquet``."""
    from src.config import (
        INCOME_CLEAN_PATH,
        STATE_COL,
        TELCO_CLEAN_PATH,
        TELCO_WITH_INCOME_PATH,
    )

    t_path = telco_path or TELCO_CLEAN_PATH
    i_path = income_path or INCOME_CLEAN_PATH
    o_path = out_path or TELCO_WITH_INCOME_PATH

    if not t_path.exists():
        raise FileNotFoundError(f"Telco clean not found: {t_path}. Run: python src/clean_telco.py")
    if not i_path.exists():
        raise FileNotFoundError(f"Income clean not found: {i_path}. Run: python src/clean_income.py")

    df_t = pd.read_parquet(t_path)
    df_i = pd.read_parquet(i_path)
    merged = merge_telco_income(
        df_t,
        df_i,
        state_col=STATE_COL,
        max_missing_income_rate=max_missing_income_rate,
    )
    o_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(o_path, index=False)
    return o_path


def main() -> None:
    from src.config import TELCO_WITH_INCOME_PATH

    p = argparse.ArgumentParser(description="Merge telco_clean + income_clean → telco_with_income.parquet")
    p.add_argument("--telco", type=Path, default=None, help="telco_clean.parquet path")
    p.add_argument("--income", type=Path, default=None, help="income_clean.parquet path")
    p.add_argument("--out", type=Path, default=None, help=f"output path (default: {TELCO_WITH_INCOME_PATH})")
    p.add_argument(
        "--max-missing-rate",
        type=float,
        default=DEFAULT_MAX_MISSING_INCOME_RATE,
        help="Max allowed fraction of rows with NaN median_household_income after merge",
    )
    args = p.parse_args()
    out = write_telco_with_income(
        telco_path=args.telco,
        income_path=args.income,
        out_path=args.out,
        max_missing_income_rate=args.max_missing_rate,
    )
    print(f"Wrote {out}")


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    main()
