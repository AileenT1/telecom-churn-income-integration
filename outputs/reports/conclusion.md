# Conclusion

## The question I wanted to answer

I wanted to rank churn risk and check whether state-level median household income adds useful signal beyond internal telecom features.

## What I did

I cleaned the customer table, created a binary churn target, and removed leakage columns such as churn reason and churn score.  
I joined Census ACS median income by state after normalizing state keys.  
I engineered tenure buckets, income quartile, and a price-sensitivity proxy.  
I trained Model A (internal features) and Model B (internal plus income) on the same stratified split using logistic regression and hist gradient boosting.

## Headline results (held-out test set)

These metrics come from the shared holdout test set in `outputs/tables/model_comparison.csv`.

| Model        | ROC AUC | PR AUC |
|-------------|---------|--------|
| A_logreg    | 0.8379  | 0.6373 |
| B_logreg    | 0.8381  | 0.6377 |
| A_hgb       | 0.8441  | 0.6487 |
| B_hgb       | 0.8441  | 0.6487 |

In logistic regression, Model B is slightly better than Model A on both ROC AUC and PR AUC.  
In hist gradient boosting, Models A and B are tied at this precision for this run.

## What seems to drive churn

Using Model B HGB with permutation importance on preprocessed test rows, the strongest signals are:

price sensitivity proxy, month-to-month contract, dependents, fiber optic internet, and electronic check payment.

Income-specific columns are not top-ranked in this run, so plan and service variables drive most of the ranking signal. Income still helps as segment context.

## Segments (EDA)

Churn-rate plots by contract, payment method, tenure bucket, and income quartile show where churn is concentrated. I use those segment views with model scores to prioritize outreach.

## What I would do next

I would run retention tests for month-to-month and electronic-check cohorts first.  
I would improve onboarding for early-tenure customers and promote support/security bundles where those features are strong.  
I would use income quartile as segmentation context for messaging and offer strategy.  
If I need stronger lift from external data, I would test finer geography or richer affordability features.

---

These numbers reflect one evaluation run. I would rerun training and evaluation whenever data or code changes.
