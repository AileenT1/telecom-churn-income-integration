"""
Clean ACS state income CSV: map state names to USPS abbreviations for joining with telco data.
Writes income_clean.parquet (state_fips, state_name, state_abbrev, median_household_income).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Full name (lower) -> USPS code. Includes DC and Puerto Rico (ACS state:* includes PR).
STATE_ABBR: dict[str, str] = {
    "alabama": "AL",
    "alaska": "AK",
    "arizona": "AZ",
    "arkansas": "AR",
    "california": "CA",
    "colorado": "CO",
    "connecticut": "CT",
    "delaware": "DE",
    "district of columbia": "DC",
    "florida": "FL",
    "georgia": "GA",
    "hawaii": "HI",
    "idaho": "ID",
    "illinois": "IL",
    "indiana": "IN",
    "iowa": "IA",
    "kansas": "KS",
    "kentucky": "KY",
    "louisiana": "LA",
    "maine": "ME",
    "maryland": "MD",
    "massachusetts": "MA",
    "michigan": "MI",
    "minnesota": "MN",
    "mississippi": "MS",
    "missouri": "MO",
    "montana": "MT",
    "nebraska": "NE",
    "nevada": "NV",
    "new hampshire": "NH",
    "new jersey": "NJ",
    "new mexico": "NM",
    "new york": "NY",
    "north carolina": "NC",
    "north dakota": "ND",
    "ohio": "OH",
    "oklahoma": "OK",
    "oregon": "OR",
    "pennsylvania": "PA",
    "rhode island": "RI",
    "south carolina": "SC",
    "south dakota": "SD",
    "tennessee": "TN",
    "texas": "TX",
    "utah": "UT",
    "vermont": "VT",
    "virginia": "VA",
    "washington": "WA",
    "west virginia": "WV",
    "wisconsin": "WI",
    "wyoming": "WY",
    "puerto rico": "PR",
}


def normalize_state_to_abbrev(x) -> str | None:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if len(s) == 2 and s.isalpha():
        return s.upper()
    s_low = s.lower()
    return STATE_ABBR.get(s_low)


def clean_income(df_income_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize ACS income CSV from ingest_acs_income: add state_abbrev, coerce income, drop unmappable rows.
    """
    df = df_income_raw.copy()

    # Expected columns from ingest_acs_income.py
    required = {"state_name", "median_household_income", "state_fips"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Income CSV missing columns: {missing}")

    df["state_abbrev"] = df["state_name"].apply(normalize_state_to_abbrev)
    df["median_household_income"] = pd.to_numeric(
        df["median_household_income"], errors="coerce"
    )
    df["state_fips"] = df["state_fips"].astype(str).str.zfill(2)

    df = df.dropna(subset=["state_abbrev"])

    out = df[["state_fips", "state_name", "state_abbrev", "median_household_income"]].copy()
    return out


def load_income_raw_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def write_income_clean(
    raw_csv: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Load raw ACS income CSV, clean, write parquet."""
    from src.config import INCOME_CLEAN_PATH, INCOME_RAW_CSV

    path_in = raw_csv or INCOME_RAW_CSV
    path_out = out_path or INCOME_CLEAN_PATH

    if not path_in.exists():
        raise FileNotFoundError(f"Income raw CSV not found: {path_in}")

    df_raw = load_income_raw_csv(path_in)
    df_clean = clean_income(df_raw)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(path_out, index=False)
    return path_out


def main() -> None:
    from src.config import INCOME_RAW_CSV

    p = argparse.ArgumentParser(
        description="Clean ACS state income CSV → income_clean.parquet"
    )
    p.add_argument(
        "--in",
        dest="in_path",
        type=Path,
        default=None,
        help=f"Input CSV (default: {INCOME_RAW_CSV})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output parquet (default: data_processed/income_clean.parquet)",
    )
    args = p.parse_args()
    out = write_income_clean(raw_csv=args.in_path, out_path=args.out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    main()
