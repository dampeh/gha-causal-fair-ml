# Causal Fairness Auditing of Student Academic Performance Prediction in Ghana

A causal fairness audit of student outcomes in Ghanaian Basic Schools
using a Regularised-Weighted Causal Forest to estimate the effects of gender, location, and school
type on at-risk status, with a UK public benchmark (OULAD) used as a transferability check.

# What's in this repo

- `GhEduData_OULAD_Causal_Fairness_Pipeline.ipynb` — the full, reproducible pipeline
- `GhEduData_Merged_Anonymized.xlsx` — the anonymized primary dataset (995 students, 14 schools)

# Engineering contribution: RW-Causal Forest

We regularize `min_samples_leaf` in `CausalForestDML` (default 5 → 8) and apply a
quarter-strength class weight to the outcome model to improve causal estimation
precision and prediction quality under class imbalance. This is grounded in the
bias-variance tradeoff described in Wager & Athey (2018).

Compared to an unmodified baseline, the Stabilized Causal Forest improves CATE
confidence interval width, AUC-PR, Macro F1, and At-risk recall across every
treatment on both datasets, with the largest gains on GhEduData, where class
imbalance is most severe (3:1, vs. OULAD's 1.12:1).

# How to run this

1. Open `GhEduData_OULAD_Causal_Fairness_Pipeline.ipynb`.
2. Upload `GhEduData_Merged_Anonymized.xlsx`.
3. Run every cell from top to bottom. OULAD downloads automatically in Section 3.


# Data

GhEduData was collected from 14 Basic Schools in Ghana. Student and school names have been replaced with anonymized IDs
(`ST_001`-`ST_995`, `SCH01`-`SCH14`). OULAD is a public UK Open University dataset,
available under CC BY 4.0 from the UCI Machine Learning Repository.

