# BTCV segmentation fine-tuning — 5-fold cross-validation

**Goal.** Fine-tune the Uniferum Stage-1 segmentation model on BTCV and push its organ-segmentation
Dice as high as possible, reported by **5-fold cross-validation over all 30 BTCV patients**.
Everything runs with the existing release scripts; you edit a YAML config and read one number.

## Why cross-validation (not a single split)
BTCV has only **30 labelled CTs** — too few for a stable single train/test split. We therefore use
5 folds: each patient is the held-out **test** in exactly one fold, so pooling the 5 folds gives one
leakage-free Dice per patient over all 30. Per fold: **18 train / 6 val / 6 test** (all disjoint).
The splits are pre-made in `${DATA_ROOT}/step1/btcv_cv5/` (fixed seed 42).

## Background (what "good" looks like)
| model | BTCV macro Dice | note |
|---|---|---|
| Uniferum zero-shot | 0.843 | starting point (no fine-tuning) — **beat this** |
| TotalSegmentator v2 | 0.891 | **leakage-free** specialist — the fair target |
| SegResNet | 0.851 | leakage-free |
| Swin UNETR | 0.900 | **trained on BTCV** (test leaked) — not a fair target |
| SuPreM | 0.907 | trained on BTCV — not a fair target |

**Beat 0.843, aim toward the leakage-free TotalSegmentator v2 (0.891).** Swin UNETR / SuPreM are
higher only because BTCV was in their training — don't chase them.

### Per-fold baseline reference
The headline comparison is the **pooled all-30 CV** number vs the all-30 references above. But since
each fold's test set is only 6 patients, we also provide the baselines' macro Dice **on each fold's
exact 6 test cases** so you can sanity-check a single fold in isolation: `baseline_reference_per_fold.csv`
(regenerate with `python experiments/btcv_seg_finetune/baseline_per_fold.py`).

| fold | TotalSeg v2 | Swin UNETR* | SegResNet |
|---|---|---|---|
| 0 | 0.881 | 0.874 | 0.841 |
| 1 | 0.906 | 0.893 | 0.882 |
| 2 | 0.908 | 0.910 | 0.852 |
| 3 | 0.896 | 0.903 | 0.865 |
| 4 | 0.864 | 0.920 | 0.815 |
| **ALL30** | **0.891** | 0.900* | 0.851 |

\*Swin UNETR is train-on-test (leaked). Per-fold numbers swing (n=6), so **do not** tune on a single
fold — always judge by the pooled 30-case CV.

## Fixed inputs (do not change)
- CV splits: `${DATA_ROOT}/step1/btcv_cv5/fold{k}_trainval.parquet` (18 train + 6 val, `split` column)
  and `fold{k}_test.parquet` (6 held-out) — **never train on the test parquets**.
- Init checkpoint: `${WEIGHTS_ROOT}/Step1_Pretrain_Seg_0328_abd/checkpoint-10000`
- The train script reads `input_file` and splits it internally by its `split` column (train/val).

## What to tune (edit in the base YAML `hf_TrainingArguments`)
| field | current | try |
|---|---|---|
| `learning_rate` | 3e-5 | 1e-5, 5e-5, 1e-4 |
| `max_steps` | 3000 | 2000, 3000, 4000 (only 18 train CTs → watch overfitting) |
| `warmup_steps` | 300 | ~10% of max_steps |
| `per_device_train_batch_size` | 4 | 2, 6 (watch GPU memory) |

Leave `training_stage: finetune`, `pretrain_checkpoint`, `freeze_llm: true`, `use_skip_connection: true`
as-is. Cosine LR decays to 0 at `max_steps`. Base configs in `configs/`: `btcv_base_lr3e5_3k` +
`btcv_lr1e4_3k`, `btcv_lr5e5_3k`, `btcv_lr3e5_2k`.

## Run (from the release repo root)

`cd ${HOME}/projects/RadVILLA/uniferum_release`

**1. Expand a base config into the 5 per-fold configs** (sets input_file + output_dir per fold):
```bash
python experiments/btcv_seg_finetune/make_fold_configs.py \
  experiments/btcv_seg_finetune/configs/btcv_base_lr3e5_3k.yaml
```
This writes `configs/_generated/btcv_base_lr3e5_3k_fold{0..4}.yaml` and prints the 5 train commands.

**2. Train all 5 folds** (each into `<output_dir>/fold{k}`):
```bash
for k in 0 1 2 3 4; do
  torchrun --nproc_per_node=8 bin/train_semantic_seg_two_stage.py \
    --config_file experiments/btcv_seg_finetune/configs/_generated/btcv_base_lr3e5_3k_fold${k}.yaml
done
```

**3. Evaluate each fold on its held-out test set** (`PYTHONPATH=.` required):
```bash
ROOT=${WEIGHTS_ROOT}/btcv_ft_sweep/base_lr3e5_3k
for k in 0 1 2 3 4; do
  PYTHONPATH=. python bin/eval_pretrain_seg.py \
    --run_dir $ROOT/fold${k} \
    --val_file ${DATA_ROOT}/step1/btcv_cv5/fold${k}_test.parquet \
    --devices 0,1,2,3,4,5,6,7 --ckpt_name checkpoint-3000 --prefix btcv_fold${k}
done
```
Each writes `$ROOT/fold{k}/btcv_fold{k}_checkpoint-3000.parquet`.

**4. Pool the 5 folds → CV macro Dice over all 30:**
```bash
python experiments/btcv_seg_finetune/score_eval.py $ROOT checkpoint-3000
```

## How to read results / tips
- The number that matters is **MACRO Dice** over the pooled 30 patients. Beat 0.843.
- `score_eval.py` warns if it doesn't see all 30 patients — that means a fold is missing or the
  splits overlap; re-check before trusting the number.
- Report **one CV number** (mean over 30 with 95% CI), not per-fold cherry-picking. Keep the same
  hyperparameters across all 5 folds for a given config.
- Convergence: with only 18 training CTs, more steps can overfit — eval a couple of checkpoints
  (`--ckpt_name checkpoint-2000`, `-3000`) and pick the setting with the best pooled CV Dice.
- Weakest organs are **gallbladder and pancreas** — most headroom is there.
- Keep a simple log (config name → pooled macro Dice). Report the best config + its per-organ table.

## Splits regeneration (only if lost)
`${DATA_ROOT}/step1/btcv_cv5/make_btcv_cv5.py` (seed 42) regenerates the fold parquets;
the per-fold `*_trainval.parquet` are `fold{k}_train` + `fold{k}_val` concatenated.
