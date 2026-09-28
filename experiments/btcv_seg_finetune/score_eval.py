#!/usr/bin/env python3
"""Pool the 5 held-out fold test predictions -> BTCV 5-fold CV macro Dice (+ per-organ).

Each of the 30 BTCV patients is the held-out test in exactly one fold, so concatenating the
5 fold-test prediction parquets gives one leakage-free Dice per patient over all 30.

Usage:
  # point at the run root that contains fold0/ .. fold4/ with the eval parquets:
  python experiments/btcv_seg_finetune/score_eval.py <run_root> [ckpt_name]
  # or pass the 5 parquet paths explicitly:
  python experiments/btcv_seg_finetune/score_eval.py f0.parquet f1.parquet f2.parquet f3.parquet f4.parquet
"""
import sys, glob, os, pandas as pd, numpy as np

REF = {'zero_shot_visionct': 0.843, 'TotalSeg_v2': 0.891, 'SegResNet': 0.851,
       'Swin UNETR (LEAKED)': 0.900, 'SuPreM (leaked)': 0.907}

def collect(args):
    if len(args) >= 5 and all(a.endswith('.parquet') for a in args):
        return args
    run_root = args[0]; ckpt = args[1] if len(args) > 1 else '*'
    files = sorted(glob.glob(f'{run_root}/fold*/btcv_fold*_{ckpt}.parquet') or
                   glob.glob(f'{run_root}/fold*/*checkpoint*.parquet'))
    return files

def main():
    files = collect(sys.argv[1:])
    assert len(files) >= 1, "no fold eval parquets found"
    print(f'pooling {len(files)} fold files:')
    for f in files: print('  ', f)
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d[(d.dice_score >= 0) & (d.gt_positive == True)]
    n_pat = d.patient_id.nunique()
    per = d.groupby('organ').dice_score.mean(); macro = per.mean()
    rng = np.random.default_rng(0); pm = d.groupby('patient_id').dice_score.mean(); pats = pm.index.values
    b = [pm.loc[rng.choice(pats, len(pats), True)].mean() for _ in range(2000)]
    lo, hi = np.percentile(b, [2.5, 97.5])
    print(f'\nBTCV 5-fold CV  ({n_pat} patients pooled, {d.organ.nunique()} organs)')
    if n_pat != 30:
        print(f'  WARNING: expected 30 patients, got {n_pat} — check that all 5 folds are present and disjoint.')
    print(f'  MACRO Dice = {macro:.4f}   (95% CI {lo:.3f}-{hi:.3f})')
    print('  per-organ:'); print('    ' + per.round(4).to_string().replace(chr(10), chr(10) + '    '))
    print('\n  reference (macro Dice):')
    for k, v in REF.items(): print(f'    {k:22s} {v:.3f}   (delta {macro - v:+.3f})')
    print('  Goal: beat zero-shot VISION-CT (0.843); fair leakage-free target = TotalSeg v2 (0.891).')

if __name__ == '__main__':
    main()
