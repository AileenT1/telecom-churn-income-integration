"""
Thin plotting helpers for EDA: churn rate by group, saved to PNG.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import OUTPUTS_FIGURES, TARGET_COL


def plot_churn_rate(
    df: pd.DataFrame,
    group_col: str,
    out_png: Path,
    *,
    target_col: str = TARGET_COL,
    title: str | None = None,
) -> None:
    """
    Bar chart of churn rate (%) by ``group_col``; saves to ``out_png``.
    """
    if target_col not in df.columns:
        raise KeyError(f"Missing target column {target_col}")
    if group_col not in df.columns:
        raise KeyError(f"Missing group column {group_col}")

    sub = df[[group_col, target_col]].dropna(subset=[group_col])
    rates = sub.groupby(group_col, observed=True)[target_col].mean() * 100.0
    rates = rates.sort_index()

    fig, ax = plt.subplots(figsize=(9, 4.5))
    rates.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Churn rate (%)")
    ax.set_xlabel(group_col)
    ax.set_title(title or f"Churn rate by {group_col}")
    ax.tick_params(axis="x", rotation=45)
    plt.setp(ax.xaxis.get_majorticklabels(), ha="right")
    fig.tight_layout()
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison_bar(
    comp: pd.DataFrame,
    out_png: Path | None = None,
    *,
    title: str | None = None,
) -> Path:
    """
    Grouped bar chart of **ROC AUC** and **PR AUC** per model row (from ``compare_models``).
    Default path: ``outputs/figures/model_comparison_bar.png``.
    """
    df = comp.copy()
    if "name" in df.columns:
        df = df.set_index("name")
    for col in ("roc_auc", "pr_auc"):
        if col not in df.columns:
            raise KeyError(f"comparison DataFrame must include '{col}'")

    names = df.index.tolist()
    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        df["roc_auc"],
        width,
        label="ROC AUC",
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.bar(
        x + width / 2,
        df["pr_auc"],
        width,
        label="PR AUC",
        color="coral",
        edgecolor="black",
        linewidth=0.5,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.set_title(title or "Model comparison (test set)")
    fig.tight_layout()
    out = Path(out_png or (OUTPUTS_FIGURES / "model_comparison_bar.png"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out
