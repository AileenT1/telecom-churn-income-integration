"""
Project configuration: paths, random seed, and key column names.
Update column names after profiling in 01_ingest_and_profile.ipynb.
"""
from pathlib import Path

# Repo root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data paths (raw data lives in data_raw/; do not commit)
DATA_RAW = PROJECT_ROOT / "data_raw"
DATA_RAW_TELCO = DATA_RAW / "kaggle_telco"   # internal/Kaggle telco CSVs
DATA_RAW_EXTERNAL_INCOME = DATA_RAW / "external_income"
# Raw ACS income CSV (from ingest_acs_income.py default)
INCOME_RAW_CSV = DATA_RAW_EXTERNAL_INCOME / "acs_income_state_2023.csv"
DATA_PROCESSED = PROJECT_ROOT / "data_processed"

# Processed artifacts
TELCO_CLEAN_PATH = DATA_PROCESSED / "telco_clean.parquet"
INCOME_CLEAN_PATH = DATA_PROCESSED / "income_clean.parquet"
TELCO_WITH_INCOME_PATH = DATA_PROCESSED / "telco_with_income.parquet"
# After feature engineering (tenure buckets, income quartile, etc.)
TELCO_WITH_FEATURES_PATH = DATA_PROCESSED / "telco_with_features.parquet"

# Outputs
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUTS_FIGURES = OUTPUTS / "figures"
OUTPUTS_TABLES = OUTPUTS / "tables"
OUTPUTS_MODELS = OUTPUTS / "models"
OUTPUTS_REPORTS = OUTPUTS / "reports"

# Reproducibility
RANDOM_SEED = 42

# Key columns (set after profiling; names match raw telco schema)
TARGET_COL = "churn"           # binary 0/1 after cleaning (raw: "Churn Label" or "Churn Value")
STATE_COL = "state"           # raw: "State" (full name, e.g. California)
ID_COL = "customerid"         # raw: "CustomerID" (optional for modeling)
