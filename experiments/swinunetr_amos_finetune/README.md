# SwinUNETR fine-tuning — AMOS 5-fold cross-validation

Fine-tune (or train from scratch) a MONAI **SwinUNETR** on AMOS with 5-fold patient-disjoint
cross-validation; report per-organ + macro Dice. Same SwinUNETR architecture as the
`swin_unetr_btcv_segmentation` bundle used as the zero-shot Stage-1 baseline, so results are directly
comparable to that baseline, to the SegResNet AMOS CV (`../segresnet_finetune`, identical folds), and to
Uniferum / TotalSegmentator.

```
swinunetr_amos_finetune/
├── prep_folds.py        # build folds/amos_fold{0-4}.json (image + TRUE GT mask)
├── train_swinunetr.py   # train ONE fold (pretrained fine-tune or scratch); per-organ val Dice
├── run_cv.py            # run all 5 folds + aggregate CV mean±std
├── configs/amos_cv.yaml
└── folds/               # AMOS fold item-lists (same splits as segresnet_finetune)
```

## Key data facts (already handled)
- **True GT masks**, not pseudo-labels: `prep_folds.py` uses `.../amos/mask/<pid>.npz` (native AMOS
  scheme), not `mask_ts/` (TotalSegmentator predictions). Verified label scheme:
  spleen=1, kidney=2&3, gallbladder=4, liver=6, stomach=7, aorta=8, pancreas=10, prostate=15 →
  remapped to a contiguous set (kidney merged): **8 organs (+bg = 9 classes)**.
- Images: npz (`arr`, raw HU) clipped to [-1000,1000] + z-score normalised.
- **Folds are identical to `../segresnet_finetune`** (200 patients → 160 train / 40 val per fold), so
  SegResNet vs SwinUNETR is a clean same-split comparison.

## Run (mrseg env has monai)
```bash
PY=python
cd ${HOME}/projects/RadVILLA/uniferum_release/experiments/swinunetr_amos_finetune
$PY prep_folds.py                                  # once (or reuse the copied folds)

# fine-tune from BTCV-pretrained SwinUNETR (DEFAULT), all 5 folds + aggregate
$PY run_cv.py --config configs/amos_cv.yaml
# from-scratch control (same folds)
$PY run_cv.py --config configs/amos_cv.yaml --scratch
# single fold (debug)
$PY train_swinunetr.py --config configs/amos_cv.yaml --fold 0
```
Each fold saves `amos_fold<k>_<mode>.pt` + `_dice.json`; `run_cv.py` writes `amos_cv_summary_<mode>.json`
with per-organ and macro Dice **mean ± std**.

## Design defaults (change in the config)
- **`pretrained: true`** — fine-tune the BTCV SwinUNETR bundle weights (final 14-class conv re-init to
  9). Set `false` (or `--scratch`) for the from-scratch control.
- SwinUNETR `feature_size=48`, `img_size=96`, default depths/heads (matches the bundle weights).
- Random foreground patches (96³), DiceCE loss, AdamW, sliding-window val (96³). SwinUNETR is heavier
  than SegResNet — reduce `batch_size` or enable `use_checkpoint` (already on) if memory is tight.

## Comparison / next step
To slot into the Stage-1 table, optionally evaluate the best fold model on the standard held-out AMOS
test set (`amos_seg_eval_8organ`) with the shared scorer. CV mean±std is the headline.
