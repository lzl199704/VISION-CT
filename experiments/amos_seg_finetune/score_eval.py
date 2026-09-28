#!/usr/bin/env python3
"""Score an AMOS/BTCV segmentation eval parquet -> macro Dice (+ per-organ).
Usage:  python experiments/amos_seg_finetune/score_eval.py <eval_parquet> [amos|btcv]
Compares against the zero-shot Uniferum and the fair (leakage-free) references."""
import sys, pandas as pd, numpy as np
EXCL={'amos':['prostate'],'btcv':[]}
REF={'amos':{'zero_shot_uniferum':0.839,'TotalSeg_v2':0.905,'SegResNet':0.861,
             'VISTA3D (LEAKED)':0.919,'SuPreM (leaked)':0.910},
     'btcv':{'zero_shot_uniferum':0.843,'TotalSeg_v2':0.891,'SegResNet':0.851,
             'Swin UNETR (LEAKED)':0.900,'SuPreM (leaked)':0.907}}
def main():
    path=sys.argv[1]; ds=sys.argv[2] if len(sys.argv)>2 else ('btcv' if 'btcv' in path.lower() else 'amos')
    d=pd.read_parquet(path); d=d[(d.dice_score>=0)&(d.gt_positive==True)&(~d.organ.isin(EXCL[ds]))]
    per=d.groupby('organ').dice_score.mean()
    macro=per.mean()
    # patient-clustered bootstrap 95% CI
    rng=np.random.default_rng(0); pm=d.groupby('patient_id').dice_score.mean(); pats=pm.index.values
    b=[pm.loc[rng.choice(pats,len(pats),True)].mean() for _ in range(2000)]
    lo,hi=np.percentile(b,[2.5,97.5])
    print(f'\n{ds.upper()}  ({d.patient_id.nunique()} patients, {d.organ.nunique()} organs)')
    print(f'  MACRO Dice = {macro:.4f}   (95% CI {lo:.3f}-{hi:.3f})')
    print('  per-organ:'); print(per.round(4).to_string().replace(chr(10),chr(10)+'    '))
    print('\n  reference (macro Dice):')
    for k,v in REF[ds].items(): print(f'    {k:22s} {v:.3f}   (delta {macro-v:+.3f})')
    print('  Goal: beat the LEAKAGE-FREE references (zero-shot Uniferum, TotalSeg v2, SegResNet).')
if __name__=='__main__': main()
