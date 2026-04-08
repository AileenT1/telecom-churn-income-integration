# Conclusion

## The question

We want to rank churn risk and test whether state-level median household income adds signal beyond internal telecom features.

## What we did

1. Cleaned the customer table, created a binary churn target, and removed leakage columns (for example, churn reason and churn score).  
2. Joined Census ACS median income by state after normalizing state keys.  
3. Engineered tenure buckets, income quartile, and a price-sensitivity proxy.  
4. Trained Model A (internal features) vs Model B (internal + income) on the same stratified split using logistic regression and hist gradient boosting.

## Headline results (held-out test set)

Metrics come from the shared holdout test set (`outputs/tables/model_comparison.csv`).

| Model        | ROC AUC | PR AUC |
|-------------|---------|--------|
| A_logreg    | 0.8379  | 0.6373 |
| B_logreg    | 0.8381  | 0.6377 |
| A_hgb       | 0.8441  | 0.6487 |
| B_hgb       | 0.8441  | 0.6487 |

- Logistic regression: Model B is slightly better than Model A on both ROC AUC and PR AUC.  
- Hist gradient boosting: A and B are tied at this precision on this run.

## What seems to drive churn (Model B HGB, permutation importance)

On preprocessed test rows, permutation importance highlights:

- Price sensitivity proxy  
- Month-to-month contract  
- Dependents  
- Fiber optic internet  
- Electronic check payment

Income-specific columns are not top-ranked in this run, so plan and service signals dominate model ranking. Income still helps with segment context.

## Segments (EDA)

Churn-rate plots by contract, payment method, tenure bucket, and income quartile show where churn concentrates. Pair these with model scores to prioritize outreach.

## So what? (Product-style actions)

1. Run retention tests for month-to-month and electronic-check cohorts.  
2. Improve onboarding for early-tenure customers and promote support/security bundles where relevant.  
3. Use income quartile as segmentation context for messaging and offer strategy.  
4. If stronger income lift is needed, test finer geography or richer affordability features.

---

Numbers above reflect one evaluation run. Re-run training and evaluation if data or code changes.
