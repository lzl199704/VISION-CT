#!/usr/bin/env python3
"""Build statistical_methods.ipynb — documents the exact stats used in the study:
segmentation Dice bootstrap CI + paired tests, and classification AUROC DeLong CI /
PR-AUC bootstrap CI / macro bootstrap, plus DeLong's paired AUROC comparison.
Run:  python build_stats_notebook.py"""
import nbformat as nbf
nb = nbf.v4.new_notebook(); cells = []
def md(s): cells.append(nbf.v4.new_markdown_cell(s))
def code(s): cells.append(nbf.v4.new_code_cell(s))

md("""# Statistical methods — VISION-CT 3D CT study

Reference notebook for the intern: the **exact** statistical tests used in the paper, with runnable
demos on synthetic data. Two settings:

1. **Step-1 organ segmentation** — macro **Dice** + **case-level bootstrap 95% CI**, and **paired tests**
   (Wilcoxon signed-rank / paired *t*-test + Cohen's *d*) comparing two models.
2. **Downstream AUC** — **AUROC** + **DeLong 95% CI** (analytic), **PR-AUC** + **bootstrap 95% CI**,
   **macro** (mean over findings) + **patient-clustered bootstrap CI**, and **DeLong's paired test** for
   comparing two models' AUROCs on the same cases.

### Reproducibility — use the SAME seeds as the experiments
All bootstraps use NumPy `default_rng(seed)` with these fixed seeds (do not change them — results must
match the paper):

| statistic | method | seed |
|---|---|---|
| segmentation Dice CI (per-organ & macro) | case-level bootstrap | **11** |
| per-finding PR-AUC CI | case-level bootstrap | **42** |
| macro AUROC / macro PR-AUC CI | patient-clustered bootstrap | **7** |
| per-finding AUROC CI | DeLong (analytic) | — (deterministic) |

Bootstrap resamples `B=2000` and percentiles `[2.5, 97.5]`.""")

code("""import numpy as np, pandas as pd
from scipy.stats import wilcoxon, ttest_rel, norm
from sklearn.metrics import roc_auc_score, average_precision_score

# fixed experiment seeds
SEED_DICE, SEED_PRAUC, SEED_MACRO = 11, 42, 7
B = 2000                         # bootstrap resamples
ALPHA = 0.95                     # 95% CI
PCT = [(1-ALPHA)/2*100, (1+ALPHA)/2*100]   # [2.5, 97.5]
print('percentiles', PCT)""")

# ============================ Segmentation ============================
md("""## 1. Segmentation (Step 1): macro Dice + 95% CI + paired tests

**Data model.** One row per (scan, organ) with a `dice_score`, filtered to ground-truth-positive organs
(`gt_positive == True`, `dice_score >= 0`). Prostate is excluded for AMOS (as in the study).

- **Per-case macro Dice** = mean Dice over a scan's organs.
- **Macro Dice** (headline) = mean over organs of each organ's mean Dice.
- **95% CI** = **case-level bootstrap**: resample *scans* (patients) with replacement `B` times, recompute
  the metric, take the 2.5/97.5 percentiles. Clustering by scan (not by row) keeps a patient's organs
  together — the correct unit of resampling.""")

code('''# --- exact CI helpers used in compute_ci_stage1.py / gen_stage1_perorgan_stats.py ---
def boot_ci_mean(values, seed=SEED_DICE, B=B):
    """Case-level bootstrap 95% CI of the mean over cases (e.g. per-scan macro Dice)."""
    rng = np.random.default_rng(seed)
    v = np.asarray(values, float)
    if len(v) < 2: return (np.nan, np.nan)
    idx = rng.integers(0, len(v), size=(B, len(v)))
    means = v[idx].mean(axis=1)
    return tuple(np.percentile(means, PCT))

def macro_dice(df):
    """mean over organs of each organ's mean per-scan Dice."""
    return df.groupby('organ').dice_score.mean().mean()

def per_scan_macro(df):
    """per-scan macro Dice (unit of bootstrap resampling)."""
    return df.groupby('patient_id').dice_score.mean()

def dice_ci(df, seed=SEED_DICE):
    """macro Dice + case-level bootstrap 95% CI (resample SCANS with replacement)."""
    pm = per_scan_macro(df)
    rng = np.random.default_rng(seed); pats = pm.index.values
    boots = [pm.iloc[rng.integers(0, len(pats), len(pats))].mean() for _ in range(B)]
    return macro_dice(df), float(np.percentile(boots, PCT[0])), float(np.percentile(boots, PCT[1]))''')

code('''# --- synthetic demo: two models on 30 scans x 7 organs ---
rng = np.random.default_rng(0)
organs = ['spleen','kidney','gallbladder','liver','stomach','aorta','pancreas']
def make(mean):
    rows=[]
    for p in range(30):
        for o in organs:
            rows.append(dict(patient_id=f'case{p:02d}', organ=o, gt_positive=True,
                             dice_score=float(np.clip(rng.normal(mean, 0.08), 0, 1))))
    return pd.DataFrame(rows)
uni = make(0.84); base = make(0.90)     # e.g. VISION-CT vs a baseline
m, lo, hi = dice_ci(uni)
print(f'VISION-CT  macro Dice {m:.4f}  (95% CI {lo:.4f}-{hi:.4f})')
m, lo, hi = dice_ci(base)
print(f'Baseline  macro Dice {m:.4f}  (95% CI {lo:.4f}-{hi:.4f})')''')

md("""### 1b. Paired comparison between two models

Compare two models on the **same scans** using per-scan macro Dice differences:
- **Wilcoxon signed-rank** — rank-based, robust for small *n* (used for the main Table-1 model-vs-baseline
  P-values).
- **Paired *t*-test** — used for the per-organ supplement.
- **Cohen's *d*** (paired) — effect size = mean(diff) / sd(diff).

Both are *paired* over the common scan set.""")

code('''def paired_seg(df_a, df_b):
    a = per_scan_macro(df_a); b = per_scan_macro(df_b)
    j = pd.concat([a, b], axis=1, keys=['a','b']).dropna()
    diff = j.a - j.b
    if np.allclose(diff, 0):
        return dict(n=len(j), wilcoxon_p=np.nan, ttest_p=np.nan, cohen_d=np.nan)
    d = diff.mean()/diff.std(ddof=1) if diff.std(ddof=1) else np.nan
    return dict(n=len(j), mean_diff=round(float(diff.mean()),4),
                wilcoxon_p=float(wilcoxon(j.a, j.b).pvalue),
                ttest_p=float(ttest_rel(j.a, j.b).pvalue),
                cohen_d=round(float(d),3))
print(paired_seg(uni, base))''')

# ============================ Classification AUC ============================
md("""## 2. Downstream classification: AUROC, PR-AUC, macro — 95% CIs

**Data model.** For a finding: binary labels `y` and continuous scores `p` (one per case). Macro = mean
over findings; the study uses `patient_id` to cluster the macro bootstrap.

- **AUROC 95% CI → DeLong (analytic)** — the fast Sun & Xu (2014) implementation; no bootstrap, so it is
  deterministic (no seed).
- **PR-AUC 95% CI → case-level bootstrap** (seed **42**) — DeLong has no PR-AUC analogue.
- **Macro AUROC / PR-AUC 95% CI → patient-clustered bootstrap** (seed **7**).""")

code('''# ===== DeLong (AUROC variance + CI), verbatim from delong_ci.py =====
def _compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, float); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1); i = j
    T2 = np.empty(N, float); T2[J] = T + 1; return T2

def _fast_delong(preds_sorted, m):
    """preds_sorted: (k_models, n) with the m positives first. Returns (aucs[k], cov[k,k])."""
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]; neg = preds_sorted[:, m:]; k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m+n])
    for r in range(k):
        tx[r] = _compute_midrank(pos[r]); ty[r] = _compute_midrank(neg[r]); tz[r] = _compute_midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1)/m/n - (m+1.0)/2.0/n
    v01 = (tz[:, :m] - tx)/n; v10 = 1.0 - (tz[:, m:] - ty)/m
    cov = np.cov(v01)/m + np.cov(v10)/n
    return aucs, np.atleast_2d(cov)

def auc_ci(y_true, y_score, alpha=ALPHA):
    """AUROC + DeLong 95% CI. Returns (auc, lo, hi). Deterministic (no seed)."""
    y = np.asarray(y_true, int); s = np.asarray(y_score, float)
    if len(np.unique(y)) < 2: return np.nan, np.nan, np.nan
    auc = roc_auc_score(y, s)
    order = (-y).argsort(); m = int(y.sum())
    _, cov = _fast_delong(s[np.newaxis, order], m)
    std = np.sqrt(max(float(cov.ravel()[0]), 0.0))
    if std == 0: return float(auc), float(auc), float(auc)
    lo, hi = norm.ppf([(1-alpha)/2, (1+alpha)/2], loc=auc, scale=std)
    return float(auc), float(max(0, lo)), float(min(1, hi))''')

code('''# ===== PR-AUC bootstrap CI (seed 42) & macro patient-clustered bootstrap (seed 7), from delong_ci.py =====
def prauc_ci(y_true, y_score, alpha=ALPHA, B=B, seed=SEED_PRAUC):
    y = np.asarray(y_true, int); s = np.asarray(y_score, float)
    if len(np.unique(y)) < 2: return np.nan, np.nan, np.nan
    pt = average_precision_score(y, s); rng = np.random.default_rng(seed); n = len(y); vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2: continue
        vals.append(average_precision_score(y[idx], s[idx]))
    return float(pt), float(np.percentile(vals, PCT[0])), float(np.percentile(vals, PCT[1]))

def _macro(df, metric):
    fn = roc_auc_score if metric == 'auc' else average_precision_score
    vals = [fn(g.label.values, g.predict.values) for _, g in df.groupby('finding')
            if len(np.unique(g.label.values)) > 1]
    return np.mean(vals) if vals else np.nan

def macro_ci(df, metric='auc', alpha=ALPHA, B=B, seed=SEED_MACRO):
    """Macro over findings + study/patient-clustered bootstrap CI.
    df columns: patient_id, finding, label, predict."""
    rng = np.random.default_rng(seed); df = df.copy(); df.patient_id = df.patient_id.astype(str)
    pats = df.patient_id.unique(); idx = {p: g.index.values for p, g in df.groupby('patient_id')}
    point = _macro(df, metric); vals = []
    for _ in range(B):
        samp = rng.choice(pats, len(pats), replace=True)
        vals.append(_macro(df.loc[np.concatenate([idx[p] for p in samp])], metric))
    return float(point), float(np.nanpercentile(vals, PCT[0])), float(np.nanpercentile(vals, PCT[1]))''')

code('''# --- synthetic demo: one finding (300 cases, ~15% prevalence) ---
rng = np.random.default_rng(1); n = 300
y = (rng.random(n) < 0.15).astype(int)
p = np.clip(0.5 + 0.3*y + rng.normal(0, 0.3, n), 0, 1)   # scores correlated with y
auc, lo, hi = auc_ci(y, p);   print(f'AUROC  {auc:.3f}  (DeLong 95% CI {lo:.3f}-{hi:.3f})')
pr, plo, phi = prauc_ci(y, p); print(f'PR-AUC {pr:.3f}  (bootstrap 95% CI {plo:.3f}-{phi:.3f})  [seed {SEED_PRAUC}]')

# macro over 3 findings, patient-clustered
rows = []
for f in ['finding_a','finding_b','finding_c']:
    yy = (rng.random(n) < 0.15).astype(int); pp = np.clip(0.5 + 0.3*yy + rng.normal(0,0.3,n), 0, 1)
    rows += [dict(patient_id=f'case{i}', finding=f, label=int(yy[i]), predict=float(pp[i])) for i in range(n)]
mdf = pd.DataFrame(rows)
mac, mlo, mhi = macro_ci(mdf, 'auc')
print(f'macro AUROC {mac:.3f}  (patient-clustered bootstrap 95% CI {mlo:.3f}-{mhi:.3f})  [seed {SEED_MACRO}]')''')

md("""### 2d. Comparing two models' AUROCs — DeLong's paired test

To test whether model A's AUROC differs from model B's **on the same cases**, use **DeLong's paired
test** (correlated ROC curves). It reuses the same fast-DeLong covariance: the z-statistic is
`(AUC_A − AUC_B) / sqrt(var_A + var_B − 2·cov_AB)`, two-sided p-value from the normal.
Deterministic (no seed). Use this for the downstream AUC comparison between two encoders/models.""")

code('''def delong_roc_test(y_true, score_a, score_b):
    """Paired DeLong test: H0 AUC_a == AUC_b on the same samples. Returns (auc_a, auc_b, p)."""
    y = np.asarray(y_true, int); order = (-y).argsort(); m = int(y.sum())
    preds = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])[:, order]
    aucs, cov = _fast_delong(preds, m)
    L = np.array([[1.0, -1.0]]); var = float(L @ cov @ L.T)
    if var <= 0: return float(aucs[0]), float(aucs[1]), np.nan
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return float(aucs[0]), float(aucs[1]), float(2*norm.sf(abs(z)))

# demo: model B strictly better-correlated scores on the same cases
pa = np.clip(0.5 + 0.20*y + rng.normal(0,0.3,n), 0, 1)
pb = np.clip(0.5 + 0.35*y + rng.normal(0,0.3,n), 0, 1)
a, b, pval = delong_roc_test(y, pa, pb)
print(f'AUROC_A {a:.3f}  vs  AUROC_B {b:.3f}   DeLong paired p = {pval:.4g}')''')

md("""## Summary — which test where

| Task | Metric | 95% CI | Compare two models |
|---|---|---|---|
| **Step-1 segmentation** | macro Dice | case-level **bootstrap** (seed 11) | **Wilcoxon** (main) / paired ***t*-test** (per-organ) + Cohen's *d* |
| **Downstream AUC (per finding)** | AUROC | **DeLong** (analytic) | **DeLong's paired test** |
| | PR-AUC | **bootstrap** (seed 42) | bootstrap the paired difference |
| **Downstream AUC (macro)** | macro AUROC/PR-AUC | patient-clustered **bootstrap** (seed 7) | — |

Notes: bootstrap resamples the *cluster* (scan / patient), `B=2000`, percentiles `[2.5, 97.5]`. DeLong
is analytic and rank-based (invariant to any monotone transform of the scores, e.g. sigmoid). Always
report the metric matching the task and select on validation / report on test.""")

nb['cells'] = cells
nbf.write(nb, 'statistical_methods.ipynb')
print('wrote statistical_methods.ipynb with', len(cells), 'cells')
