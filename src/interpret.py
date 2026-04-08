"""
Interpretability: permutation importance on preprocessed features; optional SHAP for HGB.
Writes tables under outputs/tables/ and figures under outputs/figures/.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

from src.config import OUTPUTS_FIGURES, OUTPUTS_TABLES, RANDOM_SEED


def _subsample_rows(
    X_test: pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    max_samples: int | None,
    random_state: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Optional row subsample for faster permutation importance (same indices for X and y)."""
    y = np.asarray(y_test).ravel()
    n = len(X_test)
    if max_samples is None or n <= max_samples:
        return X_test, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(n, size=max_samples, replace=False)
    return X_test.iloc[idx].reset_index(drop=True), y[idx]


def transform_preprocessed(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply fitted ``pipeline``'s preprocessor to ``X``; return dense array and
    ``get_feature_names_out()`` names for permutation importance / SHAP.
    """
    pre = pipeline.named_steps["pre"]
    Xt = pre.transform(X)
    if hasattr(Xt, "toarray"):
        Xt = Xt.toarray()
    Xt = np.asarray(Xt, dtype=np.float64)
    names = pre.get_feature_names_out()
    return Xt, names


def permutation_importance_preprocessed(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    *,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = RANDOM_SEED,
    n_jobs: int | None = -1,
    max_samples: int | None = None,
) -> tuple[Any, np.ndarray]:
    """
    Permutation importance on the **classifier** using **preprocessed** test matrix,
    so feature indices align with ``pre.get_feature_names_out()``.

    ``max_samples``: if set, draw a fixed-size random subset of test rows (faster on
    wide one-hot data); scores are slightly noisier but rankings are usually stable.
    """
    X_sub, y_sub = _subsample_rows(X_test, y_test, max_samples, random_state)
    Xt, names = transform_preprocessed(pipeline, X_sub)
    clf = pipeline.named_steps["clf"]
    y = np.asarray(y_sub).ravel()
    result = permutation_importance(
        clf,
        Xt,
        y,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    return result, names


def perm_importance_to_dataframe(result: Any, feature_names: np.ndarray) -> pd.DataFrame:
    """Sort by mean importance descending."""
    df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return df.sort_values("importance_mean", ascending=False).reset_index(drop=True)


def plot_permutation_importance(
    df: pd.DataFrame,
    out_png: Path,
    *,
    top_n: int = 20,
    title: str | None = None,
) -> Path:
    """Horizontal bar chart of top-N features with error bars (importance_std)."""
    sub = df.head(top_n).iloc[::-1]
    fig_h = max(4.0, top_n * 0.28)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    y_pos = np.arange(len(sub))
    ax.barh(
        y_pos,
        sub["importance_mean"],
        xerr=sub["importance_std"],
        color="steelblue",
        edgecolor="black",
        linewidth=0.4,
        capsize=2,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sub["feature"], fontsize=9)
    ax.set_xlabel("Mean decrease in ROC AUC (permutation)")
    ax.set_title(title or f"Top {top_n} permutation importance")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_png


def save_permutation_importance_artifacts(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: np.ndarray | pd.Series,
    csv_path: Path,
    png_path: Path,
    *,
    n_repeats: int = 10,
    top_n_plot: int = 20,
    title: str | None = None,
    random_state: int = RANDOM_SEED,
    max_samples: int | None = 5000,
    n_jobs: int | None = -1,
) -> tuple[pd.DataFrame, Path, Path]:
    """Compute permutation importance, save CSV and bar plot."""
    result, names = permutation_importance_preprocessed(
        pipeline,
        X_test,
        y_test,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
        max_samples=max_samples,
    )
    df = perm_importance_to_dataframe(result, names)
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    plot_permutation_importance(df, png_path, top_n=top_n_plot, title=title)
    return df, csv_path, Path(png_path)


def shap_summary_hgb(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    out_png: Path,
    *,
    max_samples: int = 500,
    max_display: int = 20,
    random_state: int = RANDOM_SEED,
) -> Path | None:
    """
    SHAP ``TreeExplainer`` on Model B ``HistGradientBoostingClassifier`` using a
    random sample of **preprocessed** test rows. Saves summary beeswarm-style plot.

    Returns path if successful; ``None`` if SHAP fails (optional dependency edge cases).
    """
    try:
        import shap
    except ImportError:
        return None

    clf = pipeline.named_steps["clf"]
    Xt, names = transform_preprocessed(pipeline, X_test)
    rng = np.random.default_rng(random_state)
    n = min(max_samples, len(Xt))
    idx = rng.choice(len(Xt), size=n, replace=False)
    Xs = Xt[idx]

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    try:
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list):
            sv = np.asarray(sv[1])
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            sv,
            Xs,
            feature_names=list(names),
            show=False,
            max_display=max_display,
        )
        plt.tight_layout()
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        plt.close("all")
        return None

    return out_png


def run_interpretation(
    df: pd.DataFrame | None = None,
    *,
    n_repeats: int = 10,
    top_n_plot: int = 20,
    shap_max_samples: int = 500,
    perm_max_samples: int | None = 5000,
    include_model_a: bool = True,
    perm_n_jobs: int | None = -1,
) -> dict[str, Any]:
    """
    Train A/B bundle, then write permutation importance (B required, A optional)
    and SHAP summary for B HGB.
    """
    from src.config import TELCO_WITH_FEATURES_PATH
    from src.train_models import save_ab_models, train_ab_models

    if df is None:
        if not TELCO_WITH_FEATURES_PATH.exists():
            raise FileNotFoundError(
                f"Missing {TELCO_WITH_FEATURES_PATH}. Run features pipeline or pass df."
            )
        df = pd.read_parquet(TELCO_WITH_FEATURES_PATH)

    bundle = train_ab_models(df)
    y_test = bundle["y_test"]
    models = bundle["models"]
    paths: dict[str, Path] = {}

    for name, mp in save_ab_models(bundle).items():
        paths[f"model_{name}_joblib"] = mp

    df_b, csv_b, png_b = save_permutation_importance_artifacts(
        models["B_hgb"],
        bundle["X_test_B"],
        y_test,
        OUTPUTS_TABLES / "perm_importance_modelB.csv",
        OUTPUTS_FIGURES / "perm_importance_modelB.png",
        n_repeats=n_repeats,
        top_n_plot=top_n_plot,
        title="Model B (HGB): permutation importance (ROC AUC)",
        random_state=RANDOM_SEED,
        max_samples=perm_max_samples,
        n_jobs=perm_n_jobs,
    )
    paths["perm_importance_modelB_csv"] = csv_b
    paths["perm_importance_modelB_png"] = png_b

    if include_model_a:
        _, csv_a, png_a = save_permutation_importance_artifacts(
            models["A_hgb"],
            bundle["X_test_A"],
            y_test,
            OUTPUTS_TABLES / "perm_importance_modelA.csv",
            OUTPUTS_FIGURES / "perm_importance_modelA.png",
            n_repeats=n_repeats,
            top_n_plot=top_n_plot,
            title="Model A (HGB): permutation importance (ROC AUC)",
            random_state=RANDOM_SEED,
            max_samples=perm_max_samples,
            n_jobs=perm_n_jobs,
        )
        paths["perm_importance_modelA_csv"] = csv_a
        paths["perm_importance_modelA_png"] = png_a

    shap_path = shap_summary_hgb(
        models["B_hgb"],
        bundle["X_test_B"],
        OUTPUTS_FIGURES / "shap_summary_modelB_hgb.png",
        max_samples=shap_max_samples,
        random_state=RANDOM_SEED,
    )
    if shap_path is not None:
        paths["shap_summary_modelB_hgb_png"] = shap_path

    return {
        "bundle": bundle,
        "perm_importance_modelB": df_b,
        "paths": paths,
    }


def main() -> None:
    out = run_interpretation()
    print("Wrote:", {k: str(v) for k, v in out["paths"].items()})


if __name__ == "__main__":
    main()
