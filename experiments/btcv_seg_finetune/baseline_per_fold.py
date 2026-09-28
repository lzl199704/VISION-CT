#!/usr/bin/env python3
"""Reference baseline macro Dice on each BTCV fold's held-out test patients.
Lets the intern compare a fold's fine-tuned Uniferum test Dice against the baselines
on the SAME 6 test cases. Macro = mean over 7 organs of per-organ mean Dice (GT-positive).
Writes baseline_reference_per_fold.csv."""
import pandas as pd, numpy as np, os
CV='${DATA_ROOT}/step1/btcv_cv5'; OUT=os.path.dirname(os.path.abspath(__file__))
BASE={'TotalSeg_v2':'${DATA_ROOT}/step1/totalseg_btcv_dice.parquet',
      'SwinUNETR':'${DATA_ROOT}/benchmark_seg/swin_unetr_dice.parquet',
      'SegResNet':'${DATA_ROOT}/benchmark_seg/wholebody_dice.parquet'}
def load(p):
    d=pd.read_parquet(p)
    if 'dataset' in d.columns: d=d[d.dataset=='btcv']
    return d[(d.dice_score>=0)&(d.gt_positive==True)]
def macro(d,pids):
    s=d[d.patient_id.astype(str).isin(pids)]
    return s.groupby('organ').dice_score.mean().mean()
folds={k:sorted(pd.read_parquet(f'{CV}/fold{k}_test.parquet').patient_id.astype(str).unique()) for k in range(5)}
bl={n:load(p) for n,p in BASE.items()}
rows=[]
for k in range(5):
    r={'fold':k,'n_test':len(folds[k])}
    for n,d in bl.items(): r[n]=round(macro(d,folds[k]),4)
    rows.append(r)
# all-30 sanity row
allp=sorted(pd.read_csv(f'{CV}/fold_assignment.csv').patient_id.astype(str))
rows.append({'fold':'ALL30','n_test':30,**{n:round(macro(d,allp),4) for n,d in bl.items()}})
df=pd.DataFrame(rows)
df.to_csv(f'{OUT}/baseline_reference_per_fold.csv',index=False)
print(df.to_string(index=False))
print('\n(Uniferum zero-shot all-30 BTCV macro Dice = 0.843; fine-tune should beat that and approach TotalSeg_v2.)')
