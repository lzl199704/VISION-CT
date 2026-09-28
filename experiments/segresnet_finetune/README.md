# SegResNet fine-tuning — AMOS & BTCV 5-fold cross-validation

Fine-tune (or train from scratch) a MONAI **SegResNet** on AMOS and BTCV **separately**, with 5-fold
patient-disjoint cross-validation, and report per-organ + macro Dice. Same SegResNet architecture as
the `wholeBody_ct_segmentation` bundle used as the zero-shot Stage-1 baseline, so the fine-tuned
numbers are directly comparable to that baseline (AMOS 0.861 / BTCV 0.851 zero-shot), TotalSegmentator,
and Uniferum.

```
segresnet_finetune/
├── prep_folds.py          # build folds/{amos,btcv}_fold{0-4}.json (image + TRUE GT mask)
├── train_segresnet.py     # train ONE fold (pretrained fine-tune or scratch); per-organ val Dice
├── run_cv.py              # run all 5 folds + aggregate CV mean±std
├── configs/{amos,btcv}_cv.yaml
└── folds/                 # generated fold item-lists
```

## Key data facts (already handled)
- **True GT masks**, not pseudo-labels. The AMOS manifest / BTCV cv folds point their `msk_file_path` at
  `mask_ts/` = **TotalSegmentator predictions**. Training on those would distill TotalSeg. `prep_folds.py`
  instead uses `.../mask/<pid>.npz` = the **native GT**, with verified label schemes:
  - AMOS: spleen=1, kidney=2&3, gallbladder=4, liver=6, stomach=7, aorta=8, pancreas=10, prostate=15
  - BTCV: spleen=1, kidney=2&3, gallbladder=4, liver=6, stomach=7, aorta=8, pancreas=11
- Masks are remapped to a **contiguous target set** (kidney L+R merged): AMOS → 8 organs (+bg = 9 classes),
  BTCV → 7 organs (+bg = 8). Matches the 7/8-organ Dice used elsewhere in the study.
- Images: our npz (`arr`, raw HU); clipped to [-1000,1000] and z-score normalised (matches the bundle's
  `NormalizeIntensity`).

## Cross-validation
- **BTCV**: reuses the prebuilt `btcv_cv5_fold{0-4}` splits (patient-disjoint; 18 train / 6 val per fold).
- **AMOS**: 200 training patients split into 5 patient-disjoint folds (160 train / 40 val).

## Run (mrseg env has monai)
```bash
PY=python
cd ${HOME}/projects/RadVILLA/uniferum_release/experiments/segresnet_finetune
$PY prep_folds.py                                  # once

# fine-tune from wholeBody pretrained (DEFAULT), all 5 folds + aggregate
$PY run_cv.py --config configs/amos_cv.yaml
$PY run_cv.py --config configs/btcv_cv.yaml
# from-scratch control (same folds)
$PY run_cv.py --config configs/amos_cv.yaml --scratch
# single fold (debug)
$PY train_segresnet.py --config configs/btcv_cv.yaml --fold 0
```
Each fold saves `<ds>_fold<k>_<mode>.pt` + `_dice.json`; `run_cv.py` writes `<ds>_cv_summary_<mode>.json`
with per-organ and macro Dice **mean ± std** over the 5 folds.

## Design defaults (change in the config)
- **`pretrained: true`** — fine-tune the wholeBody SegResNet (final 105-class conv re-init to n_classes).
  Set `false` (or `run_cv.py --scratch`) for the from-scratch control.
- SegResNet matches the bundle (`init_filters=32`, `blocks_down=[1,2,2,4]`, `blocks_up=[1,1,1]`).
- Train on random foreground patches (96³, `RandCropByPosNegLabel`), DiceCE loss, AdamW; validate with
  sliding-window inference (160³). Sweep `learning_rate`, `epochs`, `patch`, `batch_size` as needed.

## Comparison / next step
To place these next to the existing Stage-1 table (zero-shot SegResNet / TotalSeg / Uniferum), optionally
evaluate the best fold model on the standard held-out test sets (`amos_seg_eval_8organ` / `btcv_seg_eval_7organ`)
with the same scorer — the CV mean±std is the headline; a single-split test number aids the comparison.
