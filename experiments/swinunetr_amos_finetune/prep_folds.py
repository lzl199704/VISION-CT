#!/usr/bin/env python3
"""Build 5-fold AMOS CV item lists (image + TRUE GT mask/) -> folds/amos_fold{0-4}.json.
AMOS GT scheme: spleen=1 kidney=2,3 gallbladder=4 liver=6 stomach=7 aorta=8 pancreas=10 prostate=15."""
import os, json, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); S1 = '${DATA_ROOT}/Step1_data'
def items(pids):
    out=[]
    for pid in pids:
        img=f'{S1}/amos/image/{pid}.npz'; lbl=f'{S1}/amos/mask/{pid}.npz'
        if os.path.exists(img) and os.path.exists(lbl): out.append({'pid':pid,'image':img,'label':lbl})
    return out
pats=sorted(pd.read_parquet(f'{S1}/amos_finetune_train_manifest.parquet').patient_id.unique())
folds={k:pats[k::5] for k in range(5)}
for k in range(5):
    val=folds[k]; train=[p for j in range(5) if j!=k for p in folds[j]]
    json.dump({'dataset':'amos','fold':k,'train':items(train),'val':items(val)}, open(f'{HERE}/folds/amos_fold{k}.json','w'))
    print(f'amos fold{k}: train {len(train)} val {len(val)}')
print('wrote folds/amos_fold*.json')
