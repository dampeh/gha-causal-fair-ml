# Causal Fairness Auditing of Student Academic Performance Prediction in Ghana

A causal fairness audit of student outcomes in Ghanaian Basic Schools
using a Regularised-Weighted Causal Forest to estimate the effects of gender, location, and school
type on at-risk status, with a UK public benchmark (OULAD) used as a transferability check.

# What's in this repo

- `GhEduData_OULAD_Causal_Fairness_Pipeline.ipynb` — the full, reproducible pipeline
- `GhEduData_Merged_Anonymized.xlsx` — the anonymized primary dataset (995 students, 14 schools)
- `scripts/nested_cv_search.py` — nested-CV validation of the hyperparameter search (all four GhEduData contrasts)
- `results/` — outputs of that script

# Engineering contribution: RW-Causal Forest

We regularize `min_samples_leaf` in `CausalForestDML` (default 5 → 12) and apply a
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


# Nested-CV search validation

`scripts/nested_cv_search.py` runs the 30-configuration search (6 leaf sizes × 5 weight fractions)
under nested grouped CV (3 repeats × 5 outer × 3 inner folds) for Gender, Location (Peri-urban vs
Rural, Urban vs Rural) and School Type. Selection uses the inner loop only; only outer-loop
figures are reported.

    GHEDU_PATH=GhEduData_Merged_Anonymized.xlsx OUT_DIR=results python scripts/nested_cv_search.py

It takes about 35 min on 2 CPUs (Python 3.13, econml 0.17.0, scikit-learn 1.6.1). Outputs are in
`results/`: `nested_cv_all.csv` (outer-fold evaluations), `selection_all.csv` (configuration chosen
per outer fold), per-contrast files and `run_log.txt`. The refutation tests are in notebook
sections 7d (Gender) and 7e (School Type). Figures can shift by about a point across
Python/platform builds.

# Data

GhEduData was collected from 14 Basic Schools in Ghana. Student and school names have been replaced with anonymized IDs
(`ST_001`-`ST_995`, `SCH01`-`SCH14`). OULAD is a public UK Open University dataset,
available under CC BY 4.0 from the UCI Machine Learning Repository.

