"""
A/B training: Model A (internal features) vs Model B (+ income features).
Logistic Regression and HistGradientBoostingClassifier; shared stratified split.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from src.config import ID_COL, OUTPUTS_MODELS, RANDOM_SEED, TARGET_COL

# Filenames aligned with plan / deep-research-report (outputs/models/)
MODEL_JOBLIB_NAMES: dict[str, str] = {
    "A_logreg": "modelA_logreg.joblib",
    "A_hgb": "modelA_hgb.joblib",
    "B_logreg": "modelB_logreg.joblib",
    "B_hgb": "modelB_hgb.joblib",
}

# Model A drops income-integration columns (plan §6)
MODEL_A_EXCLUDE: tuple[str, ...] = (
    "median_household_income",
    "income_quartile",
    "state_fips",
    "state_name",
    "state_abbrev",
)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Numeric: median impute + StandardScaler.
    Categorical (object/category): most_frequent impute + OneHotEncoder(handle_unknown='ignore').
    """
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_pipe, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0,  # dense matrix for HistGradientBoostingClassifier
    )
    return pre


def _feature_columns_model_a(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    return [c for c in cols if c not in MODEL_A_EXCLUDE]


def _feature_columns_model_b(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (ID_COL, TARGET_COL)]


def train_ab_models(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    id_col: str = ID_COL,
    test_size: float = 0.25,
    random_state: int = RANDOM_SEED,
) -> dict[str, Any]:
    """
    Stratified train/test split once on indices; train 4 pipelines:
    A_logreg, A_hgb, B_logreg, B_hgb.

    Returns dict with y_test, split indices, column lists, and fitted ``Pipeline`` objects.
    """
    if target_col not in df.columns:
        raise KeyError(f"Missing target {target_col}")
    y = df[target_col].astype(int).values

    idx = np.arange(len(df))
    idx_train, idx_test, y_train, y_test = train_test_split(
        idx,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    df_train = df.iloc[idx_train].reset_index(drop=True)
    df_test = df.iloc[idx_test].reset_index(drop=True)
    y_train_s = df_train[target_col].astype(int).values
    y_test_s = df_test[target_col].astype(int).values

    cols_a = _feature_columns_model_a(df)
    cols_b = _feature_columns_model_b(df)

    X_train_a = df_train[cols_a]
    X_test_a = df_test[cols_a]
    X_train_b = df_train[cols_b]
    X_test_b = df_test[cols_b]

    models: dict[str, Pipeline] = {}

    for tag, X_tr, X_te in (
        ("A", X_train_a, X_test_a),
        ("B", X_train_b, X_test_b),
    ):
        pre = build_preprocessor(X_tr)
        pre.fit(X_tr)

        pipe_lr = Pipeline(
            steps=[
                ("pre", pre),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
        pipe_lr.fit(X_tr, y_train_s)
        models[f"{tag}_logreg"] = pipe_lr

        pre_h = build_preprocessor(X_tr)
        pipe_hgb = Pipeline(
            steps=[
                ("pre", pre_h),
                (
                    "clf",
                    HistGradientBoostingClassifier(
                        random_state=random_state,
                        max_depth=6,
                        max_iter=200,
                    ),
                ),
            ]
        )
        pipe_hgb.fit(X_tr, y_train_s)
        models[f"{tag}_hgb"] = pipe_hgb

    return {
        "y_test": y_test_s,
        "X_test_A": X_test_a,
        "X_test_B": X_test_b,
        "columns_A": cols_a,
        "columns_B": cols_b,
        "models": models,
    }


def save_ab_models(
    bundle: dict[str, Any],
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """
    Persist fitted A/B pipelines with joblib under ``outputs/models/`` (or ``out_dir``).
    Keys match ``train_ab_models``; filenames follow ``MODEL_JOBLIB_NAMES``.
    """
    out = Path(out_dir or OUTPUTS_MODELS)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, pipe in bundle["models"].items():
        fname = MODEL_JOBLIB_NAMES.get(key, f"{key}.joblib")
        dest = out / fname
        joblib.dump(pipe, dest)
        paths[key] = dest
    return paths


def main() -> None:
    from src.config import TELCO_WITH_FEATURES_PATH

    if not TELCO_WITH_FEATURES_PATH.exists():
        raise SystemExit(f"Missing {TELCO_WITH_FEATURES_PATH}. Run: python src/features.py")
    df = pd.read_parquet(TELCO_WITH_FEATURES_PATH)
    out = train_ab_models(df)
    saved = save_ab_models(out)
    print("Trained:", list(out["models"].keys()))
    print("Saved:", {k: str(v) for k, v in saved.items()})


if __name__ == "__main__":
    main()
