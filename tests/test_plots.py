"""plots.py: model comparison bar chart."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plots import plot_model_comparison_bar


def test_plot_model_comparison_bar_writes_png(tmp_path: Path):
    comp = pd.DataFrame(
        [
            {"name": "A_logreg", "roc_auc": 0.8, "pr_auc": 0.35, "log_loss": 0.5, "f1_best_threshold": 0.4},
            {"name": "B_hgb", "roc_auc": 0.82, "pr_auc": 0.38, "log_loss": 0.48, "f1_best_threshold": 0.42},
        ]
    ).set_index("name")
    out = tmp_path / "mc.png"
    plot_model_comparison_bar(comp, out)
    assert out.exists() and out.stat().st_size > 100
