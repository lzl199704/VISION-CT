#!/usr/bin/env python3
"""Run all 5 folds for a config and aggregate CV Dice (mean +/- std). Runs folds sequentially.
  python run_cv.py --config configs/amos_cv.yaml           # fine-tune (default)
  python run_cv.py --config configs/btcv_cv.yaml --scratch  # from scratch
"""
import argparse, subprocess, sys, os, json, yaml, numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--config', required=True); ap.add_argument('--scratch', action='store_true')
ap.add_argument('--folds', default='0,1,2,3,4'); a = ap.parse_args()
c = yaml.safe_load(open(a.config)); ds = c['dataset']; out = c['output_dir']
py = sys.executable; here = os.path.dirname(os.path.abspath(__file__))
cfg = a.config
if a.scratch:  # write a scratch variant next to the config
    c['pretrained'] = False; cfg = a.config.replace('.yaml', '_scratch.yaml'); yaml.safe_dump(c, open(cfg, 'w'))
tag = 'scratch' if a.scratch else 'pretrained'
for k in a.folds.split(','):
    print(f'\n===== {ds} fold {k} ({tag}) =====', flush=True)
    subprocess.run([py, os.path.join(here, 'train_segresnet.py'), '--config', cfg, '--fold', k], check=True)
# aggregate
res = [json.load(open(f'{out}/{ds}_fold{k}_{tag}_dice.json')) for k in a.folds.split(',')]
organs = list(res[0]['per_organ']); M = np.array([[r['per_organ'][o] for o in organs] for r in res])
macro = np.array([r['macro_dice'] for r in res])
print(f'\n===== {ds.upper()} 5-fold CV ({tag}) =====')
for i, o in enumerate(organs): print(f'  {o:12s} {M[:,i].mean():.4f} +/- {M[:,i].std():.4f}')
print(f'  {"MACRO":12s} {macro.mean():.4f} +/- {macro.std():.4f}')
json.dump({'dataset': ds, 'mode': tag, 'macro_mean': float(macro.mean()), 'macro_std': float(macro.std()),
           'per_organ_mean': {o: float(M[:,i].mean()) for i,o in enumerate(organs)}},
          open(f'{out}/{ds}_cv_summary_{tag}.json', 'w'), indent=1)
print(f'wrote {out}/{ds}_cv_summary_{tag}.json')
