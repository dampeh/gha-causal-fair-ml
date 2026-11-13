# ============================================================
# scripts/nested_cv_search.py
# Nested-CV hyperparameter-search validation for ALL four
# GhEduData contrasts (Gender, Location x2, School Type), one code
# path. Factorial 30-configuration grid (6 leaf sizes x 5 weight
# fractions), nested outer/inner grouped CV (3 repeats x 5 outer x
# 3 inner). Selection uses the INNER loop only; only OUTER-loop
# figures are evidence (M13). Compared arms per outer fold:
#   A_adopted_leaf12_w025  the configuration used in the pipeline
#   A_baseline_leaf5_w0    unmodified baseline
#   B_nested_selected      what the declared selection rule picks
# Reproduces M13 / Results Table 3 footnote (validated reductions).
# Run:  python scripts/nested_cv_search.py   (about 30-60 min, 2 CPUs)
# Needs: python 3.13, econml 0.17.0, scikit-learn 1.6.1, pandas 2.2.3, numpy 2.1.3
# Output: results/nested_cv_all.csv, results/selection_all.csv (+ per-contrast files)
# ============================================================
import os, sys, time, json
import numpy as np
import pandas as pd
import sklearn, econml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from econml.dml import CausalForestDML

print("PYTHON:", sys.version)
print("scikit-learn:", sklearn.__version__, "| econml:", econml.__version__)
assert sklearn.__version__.startswith("1.6")
assert econml.__version__.startswith("0.17")

GH_PATH = os.environ.get("GHEDU_PATH", "GhEduData_Merged_Anonymized.xlsx")
GH_SHEET = "GhEduData_Merged"
SCHOOL_COL = "School_ID"
AT_RISK_CUT = 21
N_ESTIMATORS = 500
MAX_DEPTH = 8
HONEST = True
MODEL_Y_MAX_ITER = 1000
DECISION_THR = 0.5
SPLIT_SEED = 42
MODEL_SEED = 42
TEST_SIZE = 0.2
TRIM_LO, TRIM_HI = 0.05, 0.95

N_REPEATS = 3
N_OUTER = 5
N_INNER = 3

LEAF_GRID = [5, 8, 10, 12, 15, 20]
WEIGHT_GRID = [0.0, 0.10, 0.20, 0.25, 1.0]
LEAF_BASE = 5
LEAF_ADOPTED, WEIGHT_ADOPTED = 12, 0.25   # the actually-adopted config (M12/M13)
SELECT_MIN_REDUCTION = 0.10

OUT_DIR = os.environ.get("OUT_DIR", "results")
os.makedirs(OUT_DIR, exist_ok=True)

# ---- load + encode (identical to the main pipeline) ----------
df = pd.read_excel(GH_PATH, sheet_name=GH_SHEET)
print("SHAPE AS LOADED:", df.shape)

def clean_text(v):
    return " ".join(str(v).replace("□", "").strip().split())
for c in ["Governance_Type", "Location_Category", "Avg_Class_Size",
          "Pct_Teachers_Degree", "Teacher_Student_Ratio", "Classroom_Condition",
          "English_Textbook_Access", "Math_Textbook_Access",
          "Home_Study_Access", "Socioeconomic_Profile"]:
    df[c] = df[c].apply(clean_text)

def enc_att(v):
    if "Less than 50%" in v: return 0
    if "50" in v and "74" in v: return 1
    if "75" in v and "90" in v: return 2
    if "More than 90%" in v: return 3
    return -1
def enc_hw(v):
    for i, k in enumerate(["Rarely", "Sometimes", "Often", "Almost always"]):
        if k in v: return i
    return -1
def enc_part(v):
    if "Passive" in v: return 0
    if "Very active" in v: return 3
    if "Moderate" in v: return 1
    if "Active" in v: return 2
    return -1

def enc_class(v):
    return {"Fewer than 20": 0, "20\u201335": 1, "36\u201350": 2,
            "51 - 65": 3, "66 - 80": 4, "More than 80": 5}.get(v, -1)
def enc_qual(v):
    return {"Less than 25%": 0, "50\u201375%": 1, "More than 75%": 2}.get(v, -1)
def enc_ratio(v):
    for k, i in (("fewer than 25", 0), ("25\u201335", 1), ("36\u201350", 2),
                 ("51 - 65", 3), ("66 - 80", 4)):
        if k in v: return i
    return -1
def enc_cond(v):
    for k, i in (("Some classrooms are inadequate", 0),
                 ("Mostly adequate", 1), ("All classrooms adequate", 2)):
        if k in v: return i
    return -1
def enc_book(v):
    for k, i in (("3 or more", 0), ("between 2", 1), ("own copy", 2)):
        if k in v: return i
    return -1
def enc_home(v):
    if "Fewer than 25%" in v: return 0
    if "25\u201349%" in v: return 1
    if "More than 75%" in v: return 3
    if "50" in v: return 2
    return -1
def enc_ses(v):
    for k, i in (("subsistence", 0), ("low income", 1),
                 ("middle income", 2), ("high income", 3)):
        if k in v: return i
    return -1

df["at_risk"] = (pd.to_numeric(df["Aggregate"], errors="raise") >= AT_RISK_CUT).astype(int)
df["location_num"] = df["Location_Category"].map({"Rural": 0, "Peri-urban": 1, "Urban": 2}).fillna(-1).astype(int)
df["gender_num"] = df["Gender"].apply(lambda x: 1 if x == "Male" else 0)
df["class_size_num"] = df["Avg_Class_Size"].apply(enc_class)
df["teacher_qual_num"] = df["Pct_Teachers_Degree"].apply(enc_qual)
df["teacher_ratio_num"] = df["Teacher_Student_Ratio"].apply(enc_ratio)
df["classroom_condition_num"] = df["Classroom_Condition"].apply(enc_cond)
df["english_textbook_num"] = df["English_Textbook_Access"].apply(enc_book)
df["math_textbook_num"] = df["Math_Textbook_Access"].apply(enc_book)
df["home_study_num"] = df["Home_Study_Access"].apply(enc_home)
df["socioeconomic_num"] = df["Socioeconomic_Profile"].apply(enc_ses)
df["school_type_num"] = df["Governance_Type"].apply(lambda x: 1 if "Private" in x else 0)
df["attendance_num"] = df["Attendance Rate"].apply(enc_att)
df["homework_num"] = df["Homework Submission Rate"].apply(enc_hw)
df["participation_num"] = df["Classroom Participation"].apply(enc_part)

STUDENT_CONF = ["Age", "attendance_num", "homework_num", "participation_num"]
FULL_CONF = STUDENT_CONF + ["class_size_num", "teacher_qual_num", "teacher_ratio_num",
                            "classroom_condition_num", "english_textbook_num",
                            "math_textbook_num", "home_study_num", "socioeconomic_num"]
for c in FULL_CONF:
    assert int((df[c] == -1).sum()) == 0, f"UNMATCHED ENCODINGS in {c}"
for c in STUDENT_CONF + ["location_num", "school_type_num"]:
    n_bad = int((df[c] == -1).sum()) if c in ["attendance_num","homework_num","participation_num","location_num"] else 0
    assert n_bad == 0, f"UNMATCHED ENCODINGS in {c}: {n_bad}"

def weight_from(y_train, frac):
    if frac == 0.0:
        return None
    n1, n0 = int((y_train == 1).sum()), int((y_train == 0).sum())
    r = max(n1, n0) / min(n1, n0) if min(n1, n0) > 0 else 1.0
    return {0: 1, 1: 1 + (r - 1) * frac}

def build(leaf, cw):
    return CausalForestDML(
        model_y=LogisticRegression(class_weight=cw, max_iter=MODEL_Y_MAX_ITER),
        model_t=RandomForestClassifier(random_state=MODEL_SEED),
        discrete_outcome=True, discrete_treatment=True,
        n_estimators=N_ESTIMATORS, min_samples_leaf=leaf,
        max_depth=MAX_DEPTH, honest=HONEST, random_state=MODEL_SEED)

def fit_and_score(leaf, frac, X_tr, T_tr, Y_tr, X_te, T0, T1):
    m = build(leaf, weight_from(Y_tr, frac))
    m.fit(Y_tr, T_tr, X=X_tr)
    lo, hi = m.effect_interval(X_te, T0=T0, T1=T1, alpha=0.05)
    tau = m.effect(X_te, T0=T0, T1=T1)
    return float((hi - lo).mean()), float(np.std(np.asarray(tau).ravel(), ddof=1))

GRID = [(l, w) for l in LEAF_GRID for w in WEIGHT_GRID]
print("GRID size:", len(GRID))

TREATMENTS = [
    ("Gender",       "gender_num", FULL_CONF, 0, 1, False),
    ("Location_1v0", "location_num", STUDENT_CONF, 0, 1, False),
    ("Location_2v0", "location_num", STUDENT_CONF, 0, 2, False),
    ("SchoolType",   "school_type_num", STUDENT_CONF, 0, 1, True),
]

all_rows, all_sel = [], []
overall_start = time.time()

for tname, tcol, conf, T0, T1, needs_trim in TREATMENTS:
    print("\n" + "=" * 78)
    print(f"TREATMENT: {tname} (T0={T0}, T1={T1}, trim={needs_trim})")
    print("=" * 78)
    t_start = time.time()

    work = df.copy()
    if needs_trim:
        ov = LogisticRegression(max_iter=1000)
        ov.fit(work[conf].values, work[tcol].values)
        pr = ov.predict_proba(work[conf].values)[:, 1]
        # multi-class-safe: school_type_num is binary, fine
        keep = (pr >= TRIM_LO) & (pr <= TRIM_HI)
        n_before = len(work)
        work = work.loc[keep]
        print(f"  FULL-DATA overlap trim | n before {n_before} | n after {len(work)}")

    X_raw_full = work[conf].values
    T_full = work[tcol].values
    Y_full = work["at_risk"].values
    G_full = work[SCHOOL_COL].values

    gsplit = StratifiedGroupKFold(n_splits=int(round(1 / TEST_SIZE)), shuffle=True, random_state=SPLIT_SEED)
    search_idx, confirm_idx = next(gsplit.split(X_raw_full, Y_full, groups=G_full))
    Xs, Ts, Ys, Gs = X_raw_full[search_idx], T_full[search_idx], Y_full[search_idx], G_full[search_idx]
    print(f"  search n = {len(Ys)} (of {len(Y_full)})")

    rows, sel_rows = [], []
    for repeat in range(N_REPEATS):
        outer = StratifiedGroupKFold(n_splits=N_OUTER, shuffle=True, random_state=repeat)
        for ofold, (otr, ote) in enumerate(outer.split(Xs, Ys, groups=Gs), 1):
            Xo_tr_raw, Xo_te_raw = Xs[otr], Xs[ote]
            To_tr, To_te, Yo_tr, Yo_te = Ts[otr], Ts[ote], Ys[otr], Ys[ote]
            Go_tr = Gs[otr]

            # in-fold trim for SchoolType, matching the main protocol
            if needs_trim:
                ov2 = LogisticRegression(max_iter=1000)
                ov2.fit(Xo_tr_raw, To_tr)
                keep_tr = ((ov2.predict_proba(Xo_tr_raw)[:, 1] >= TRIM_LO) &
                           (ov2.predict_proba(Xo_tr_raw)[:, 1] <= TRIM_HI))
                Xo_tr_raw, To_tr, Yo_tr, Go_tr = Xo_tr_raw[keep_tr], To_tr[keep_tr], Yo_tr[keep_tr], Go_tr[keep_tr]
                keep_te = ((ov2.predict_proba(Xo_te_raw)[:, 1] >= TRIM_LO) &
                           (ov2.predict_proba(Xo_te_raw)[:, 1] <= TRIM_HI))
                Xo_te_raw, To_te, Yo_te = Xo_te_raw[keep_te], To_te[keep_te], Yo_te[keep_te]

            sc = StandardScaler()
            Xo_tr, Xo_te = sc.fit_transform(Xo_tr_raw), sc.transform(Xo_te_raw)
            if (T0 not in set(To_tr)) or (T1 not in set(To_tr)) or (T0 not in set(To_te)) or (T1 not in set(To_te)) or len(To_te) < 10:
                sel_rows.append(dict(treatment=tname, repeat=repeat, outer_fold=ofold,
                                      n_qualified=0, chosen_leaf=np.nan, chosen_weight=np.nan,
                                      note="INESTIMABLE: treatment constant or test fold too small"))
                continue

            inner = StratifiedGroupKFold(n_splits=N_INNER, shuffle=True, random_state=repeat)
            inner_scores = {cfg: {"ci": [], "sd": []} for cfg in GRID}
            n_inner_used = 0
            for itr, ite in inner.split(Xo_tr, Yo_tr, groups=Go_tr):
                if ((T0 not in set(To_tr[itr])) or (T1 not in set(To_tr[itr])) or
                        (T0 not in set(To_tr[ite])) or (T1 not in set(To_tr[ite]))):
                    continue
                n_inner_used += 1
                for cfg in GRID:
                    ci, sd = fit_and_score(cfg[0], cfg[1], Xo_tr[itr], To_tr[itr], Yo_tr[itr], Xo_tr[ite], T0, T1)
                    inner_scores[cfg]["ci"].append(ci)
                    inner_scores[cfg]["sd"].append(sd)

            if n_inner_used == 0:
                sel_rows.append(dict(treatment=tname, repeat=repeat, outer_fold=ofold,
                                      n_qualified=0, chosen_leaf=np.nan, chosen_weight=np.nan,
                                      note="INESTIMABLE: no usable inner fold"))
                continue

            grid_df = pd.DataFrame([
                {"leaf": c[0], "weight": c[1],
                 "inner_ci_mean": np.mean(v["ci"]) if v["ci"] else np.nan,
                 "inner_cate_sd": np.mean(v["sd"]) if v["sd"] else np.nan}
                for c, v in inner_scores.items()])
            base_row = grid_df.loc[(grid_df.leaf == LEAF_BASE) & (grid_df.weight == 0.0)]
            if base_row.empty or pd.isna(base_row["inner_ci_mean"].iloc[0]):
                sel_rows.append(dict(treatment=tname, repeat=repeat, outer_fold=ofold,
                                      n_qualified=0, chosen_leaf=np.nan, chosen_weight=np.nan,
                                      note="INESTIMABLE: baseline config failed in inner loop"))
                continue
            base_ci = base_row["inner_ci_mean"].iloc[0]
            base_sd = base_row["inner_cate_sd"].iloc[0]
            grid_df["ci_reduction"] = 1 - grid_df["inner_ci_mean"] / base_ci
            grid_df["sd_retained"] = grid_df["inner_cate_sd"] / base_sd
            grid_df["repeat"], grid_df["outer_fold"] = repeat, ofold
            grid_df["treatment"] = tname

            qualified = grid_df[grid_df["ci_reduction"] > SELECT_MIN_REDUCTION]
            if len(qualified) == 0:
                chosen, why = None, "NO CONFIGURATION MET THE 10% RULE"
            else:
                best = qualified.loc[qualified["sd_retained"].idxmax()]
                chosen, why = (int(best["leaf"]), float(best["weight"])), "rules 1 and 2"
            sel_rows.append(dict(treatment=tname, repeat=repeat, outer_fold=ofold,
                                  n_qualified=len(qualified),
                                  chosen_leaf=chosen[0] if chosen else np.nan,
                                  chosen_weight=chosen[1] if chosen else np.nan, note=why))
            rows.append(grid_df.assign(kind="inner_grid"))

            for label, cfg in (("A_adopted_leaf12_w025", (LEAF_ADOPTED, WEIGHT_ADOPTED)),
                               ("A_baseline_leaf5_w0", (LEAF_BASE, 0.0)),
                               ("B_nested_selected", chosen)):
                if cfg is None:
                    rows.append(pd.DataFrame([dict(kind="outer", treatment=tname, regime=label,
                                                   repeat=repeat, outer_fold=ofold,
                                                   leaf=np.nan, weight=np.nan, outer_ci=np.nan,
                                                   note="no configuration qualified")]))
                    continue
                ci, sd = fit_and_score(cfg[0], cfg[1], Xo_tr, To_tr, Yo_tr, Xo_te, T0, T1)
                rows.append(pd.DataFrame([dict(kind="outer", treatment=tname, regime=label,
                                               repeat=repeat, outer_fold=ofold,
                                               leaf=cfg[0], weight=cfg[1],
                                               outer_ci=ci, outer_cate_sd=sd, note="")]))
            print(f"  [{tname} r{repeat} o{ofold}] qualified {len(qualified)}/{len(GRID)} "
                  f"| selected {chosen} | {time.time()-t_start:.0f}s elapsed "
                  f"| total {time.time()-overall_start:.0f}s")

    if rows:
        out = pd.concat(rows, ignore_index=True)
        out.to_csv(f"{OUT_DIR}/nested_cv_{tname}.csv", index=False)
        all_rows.append(out)
    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(f"{OUT_DIR}/selection_{tname}.csv", index=False)
    all_sel.append(sel_df)
    print(f"TREATMENT {tname} DONE in {time.time()-t_start:.0f}s")

combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
combined.to_csv(f"{OUT_DIR}/nested_cv_all.csv", index=False)
combined_sel = pd.concat(all_sel, ignore_index=True) if all_sel else pd.DataFrame()
combined_sel.to_csv(f"{OUT_DIR}/selection_all.csv", index=False)

print("\n" + "=" * 78)
print("FINAL REPORT - outer-loop CI width by treatment and regime")
print("=" * 78)
o = combined[combined["kind"] == "outer"].groupby(["treatment", "regime"])["outer_ci"].agg(
    mean="mean", sd=lambda s: s.std(ddof=1), n="count")
print(o.to_string())

print("\nSELECTION STABILITY per treatment:")
for tname in combined_sel["treatment"].unique():
    sub = combined_sel[combined_sel["treatment"] == tname]
    n_fail = int(sub["chosen_leaf"].isna().sum())
    distinct = sub.dropna(subset=["chosen_leaf"])[["chosen_leaf", "chosen_weight"]].drop_duplicates().shape[0]
    print(f"  {tname}: outer folds with no qualifying config: {n_fail}/{len(sub)} | distinct selections: {distinct}")

print(f"\nTOTAL RUNTIME: {time.time()-overall_start:.0f}s")
print("DONE")
