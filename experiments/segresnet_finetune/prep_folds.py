#!/usr/bin/env python3
"""Build 5-fold CV item lists (image + TRUE GT mask) for AMOS and BTCV SegResNet fine-tuning.

GT masks: use `.../mask/<pid>.npz` (native AMOS/BTCV label scheme, verified), NOT `mask_ts/`
(which are TotalSegmentator pseudo-labels). Writes folds/{amos,btcv}_fold{0-4}.json.

  AMOS GT scheme : spleen=1 kidney=2,3 gallbladder=4 liver=6 stomach=7 aorta=8 pancreas=10 prostate=15
  BTCV GT scheme : spleen=1 kidney=2,3 gallbladder=4 liver=6 stomach=7 aorta=8 pancreas=11

Run: python prep_folds.py
"""
import os, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
S1 = '${DATA_ROOT}/Step1_data'

def items_for(pids, ds):
    root = f'{S1}/{ds}'
    out = []
    for pid in pids:
        img = f'{root}/image/{pid}.npz'; lbl = f'{root}/mask/{pid}.npz'
        if os.path.exists(img) and os.path.exists(lbl):
            out.append({'pid': pid, 'image': img, 'label': lbl})
    return out

# ---------- AMOS: make 5 patient-disjoint folds from the 200-patient train manifest ----------
amos = pd.read_parquet(f'{S1}/amos_finetune_train_manifest.parquet')
pats = sorted(amos.patient_id.unique())
folds = {k: pats[k::5] for k in range(5)}           # deterministic round-robin split
for k in range(5):
    val = folds[k]; train = [p for j in range(5) if j != k for p in folds[j]]
    d = {'dataset': 'amos', 'fold': k, 'train': items_for(train, 'amos'), 'val': items_for(val, 'amos')}
    json.dump(d, open(f'{HERE}/folds/amos_fold{k}.json', 'w'))
    print(f'amos fold{k}: train {len(d["train"])}  val {len(d["val"])}')

# ---------- BTCV: reuse the prebuilt cv5 folds (train/val split already patient-disjoint) ----------
for k in range(5):
    f = pd.read_parquet(f'{S1}/btcv_cv5_fold{k}_trainval_wlabel.parquet')
    tr = sorted(f[f.split == 'train'].patient_id.unique()); va = sorted(f[f.split == 'val'].patient_id.unique())
    # patient_id here is e.g. img0001 -> matches image/mask filenames
    d = {'dataset': 'btcv', 'fold': k, 'train': items_for(tr, 'btcv'), 'val': items_for(va, 'btcv')}
    json.dump(d, open(f'{HERE}/folds/btcv_fold{k}.json', 'w'))
    print(f'btcv fold{k}: train {len(d["train"])}  val {len(d["val"])}')
print('wrote folds/*.json')
