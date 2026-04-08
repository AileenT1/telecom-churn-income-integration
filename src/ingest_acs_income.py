"""
Pull ACS 5-year median household income by state from the Census API.
Option 2 (Census ACS API) — reproducible and scriptable.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import requests

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def fetch_acs_income_by_state(
    year: int,
    out_csv: Path,
    api_key: str | None = None,
) -> None:
    """
    Pull ACS 5-year median household income by state using B19013_001E.

    Uses Census API endpoint: /data/{year}/acs/acs5.
    Joins later will require state name -> abbreviation normalization (see clean_income).
    """
    base_url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME,B19013_001E",
        "for": "state:*",
    }
    if api_key:
        params["key"] = api_key

    r = requests.get(base_url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    df = pd.DataFrame(data[1:], columns=data[0])
    df = df.rename(columns={
        "NAME": "state_name",
        "B19013_001E": "median_household_income",
        "state": "state_fips",
    })

    df["median_household_income"] = pd.to_numeric(
        df["median_household_income"], errors="coerce"
    )
    df["state_fips"] = df["state_fips"].astype(str).str.zfill(2)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)


def main() -> None:
    if load_dotenv is not None:
        load_dotenv()

    from src.config import DATA_RAW_EXTERNAL_INCOME

    p = argparse.ArgumentParser(
        description="Fetch ACS 5-year median household income by state (B19013_001E)."
    )
    p.add_argument("--year", type=int, default=2023)
    p.add_argument(
        "--out",
        type=Path,
        default=DATA_RAW_EXTERNAL_INCOME / "acs_income_state_2023.csv",
    )
    p.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Census API key (or set CENSUS_API_KEY in .env)",
    )
    args = p.parse_args()

    api_key = args.api_key or os.environ.get("CENSUS_API_KEY")
    fetch_acs_income_by_state(
        year=args.year,
        out_csv=args.out,
        api_key=api_key,
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    import sys
    _root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_root))
    main()
