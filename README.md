# Telecom Churn Intelligence with Census Income Integration

This project predicts telecom customer churn and tests whether adding external Census income data improves model quality and business insight.

## Business goal

Identify high-risk churn segments and compare:
- **Model A:** internal telecom features only
- **Model B:** telecom features + income features

## What I built

- End-to-end data pipeline from raw data to final report
- Leakage-safe cleaning (drops churn post-outcome fields)
- State-level data integration using Census ACS (`B19013_001E`)
- Feature engineering (tenure bucket, income quartile, price sensitivity proxy)
- A/B model training with Logistic Regression and HistGradientBoosting
- Evaluation with ROC AUC, PR AUC, log loss, and best-threshold F1
- Interpretability with permutation importance and optional SHAP

## Main result snapshot

- Best tree model: **ROC-AUC 0.8441**, **PR-AUC 0.6487**
- Logistic model: small lift from income integration  
  (**B vs A** ROC-AUC: 0.8381 vs 0.8379, PR-AUC: 0.6377 vs 0.6373)

## Tech stack

Python, pandas, NumPy, scikit-learn, matplotlib, Jupyter, pytest.

## Data sources

| Source | Purpose |
|---|---|
| Telco CSVs (`data_raw/kaggle_telco/`) | Customer features + churn target |
| U.S. Census ACS 5-year API | State median household income (`B19013_001E`) |

## Project structure

```text
data_raw/                 # raw datasets (gitignored)
data_processed/           # cleaned, merged, featurized datasets
src/                      # pipeline code
notebooks/                # analysis + narrative notebooks
outputs/                  # figures, tables, models, reports
tests/                    # automated tests
```

## Quick start

```powershell
cd "path\to\telecom-churn-income-integration"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Optional:

```powershell
pip install -e .
```

Set API key (first-time setup):
1. Copy `.env.example` to `.env`
2. Add `CENSUS_API_KEY=...`  
   [Get key](https://api.census.gov/data/key_signup.html)

## Reproduce pipeline

```powershell
python src/ingest_kaggle.py
python src/ingest_acs_income.py
python src/clean_telco.py
python src/clean_income.py
python src/merge_income.py
python src/features.py
python src/train_models.py
python src/interpret.py
```

## Key deliverables

| Deliverable | Path |
|---|---|
| Model metrics table | `outputs/tables/model_comparison.csv` |
| Model comparison plot | `outputs/figures/model_comparison_bar.png` |
| Feature importance | `outputs/tables/perm_importance_modelB.csv` |
| Final report (HTML) | `outputs/reports/final_story.html` |
| Written conclusion | `outputs/reports/conclusion.md` |
| Resume bullets | `outputs/reports/resume_bullets.md` |

## Tests

```powershell
pytest -v
```

## Notes

- `data_raw/` is intentionally gitignored
- `.env` is not committed

## Contact

**Aileen Tao**  
Email: `aileen060803tyy@gmail.com`  
LinkedIn: [linkedin.com/in/aileen-tao-62157937b](https://www.linkedin.com/in/aileen-tao-62157937b/)
