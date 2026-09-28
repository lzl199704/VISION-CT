# AMOS segmentation fine-tuning — hyperparameter tuning

**Goal.** Fine-tune the Uniferum Stage-1 segmentation model on the AMOS training split and push
its organ-segmentation Dice on the AMOS test set as high as possible. Everything runs with the
existing release scripts; you only edit a YAML config and read one number.

## Background (what "good" looks like)
| model | AMOS macro Dice | note |
|---|---|---|
| Uniferum zero-shot | 0.839 | starting point (no fine-tuning) |
| **Uniferum fine-tuned (current best)** | **0.876** | lr 3e-5, 6k steps — your baseline to beat |
| TotalSegmentator v2 | 0.905 | **leakage-free** specialist — the fair target |
| SegResNet | 0.861 | leakage-free |
| VISTA3D / SuPreM | 0.919 / 0.910 | **trained on AMOS** (test leaked) — not fair targets |

So: **beat 0.876, aim toward 0.905.** VISTA3D/SuPreM are higher only because AMOS was in their
training — don't chase them.

## Fixed inputs (do not change)
- Train manifest: `${DATA_ROOT}/step1/amos_finetune_train_manifest.parquet` (200 CT, disjoint from test)
- Init checkpoint: `${WEIGHTS_ROOT}/Step1_Pretrain_Seg_0328_abd/checkpoint-10000`
- Test set: `${DATA_ROOT}/step1/amos_seg_eval_8organ.parquet` (100 CT — **never train on this**)

## What to tune (edit these in the YAML `hf_TrainingArguments`)
| field | current | try |
|---|---|---|
| `learning_rate` | 3e-5 | 1e-5, 5e-5, 1e-4 |
| `max_steps` | 6000 | 4000, 6000, 8000, 10000 |
| `warmup_steps` | 600 | ~10% of max_steps |
| `per_device_train_batch_size` | 4 | 2, 6, 8 (watch GPU memory) |

Also tunable (advanced): `dice_weight`/`bce_weight` (loss mix), `train_transform` (augmentation).
Leave `training_stage: finetune`, `pretrain_checkpoint`, `freeze_llm: true`, `use_skip_connection: true`
as-is. Note: the LR schedule is cosine-to-0 at `max_steps`, so raising `max_steps` keeps LR higher longer.

Four starting configs are in `configs/`: `base_lr3e5_6k` (the current best), `lr1e4_6k`, `lr5e5_8k`,
`lr1e5_6k`. **Make a new run by copying one, changing the hyperparameter AND the `output_dir`** (give each
run its own folder or they overwrite each other).

## Run (from the release repo root)

Train (8 GPUs):
```bash
cd ${HOME}/projects/RadVILLA/uniferum_release
torchrun --nproc_per_node=8 bin/train_semantic_seg_two_stage.py \
  --config_file experiments/amos_seg_finetune/configs/base_lr3e5_6k.yaml
```

Evaluate the final checkpoint on the AMOS test set (`PYTHONPATH=.` is required for the eval script):
```bash
PYTHONPATH=. python bin/eval_pretrain_seg.py \
  --run_dir <output_dir from the config> \
  --val_file ${DATA_ROOT}/step1/amos_seg_eval_8organ.parquet \
  --devices 0,1,2,3,4,5,6,7 --ckpt_name checkpoint-6000 --prefix amos_eval
```
This writes `<output_dir>/amos_eval_checkpoint-6000.parquet`.

Score it (macro Dice + per-organ + comparison to the references):
```bash
python experiments/amos_seg_finetune/score_eval.py <output_dir>/amos_eval_checkpoint-6000.parquet amos
```

## How to read results / tips
- The number that matters is **MACRO Dice** printed by `score_eval.py`. Beat 0.876.
- Check convergence: eval several checkpoints (`--ckpt_name checkpoint-4000`, `-6000`, ...). If Dice is
  still rising at the last checkpoint, increase `max_steps`; if it plateaued earlier, it's converged
  (in the current best, test Dice plateaued by ~step 4000).
- The weakest organs are **gallbladder and pancreas** — watch those per-organ numbers; most headroom is there.
- Keep a simple log (config name -> macro Dice). Report the best config + its per-organ table.
- Optional cross-check: run the same eval with `--val_file ${DATA_ROOT}/step1/btcv_seg_eval_7organ.parquet --prefix btcv_eval`
  then `score_eval.py ... btcv` — a good config should not *hurt* BTCV (should stay ~0.85).

## Data regeneration (only if the manifest is ever lost)
The train manifest / npz were built by `${DATA_ROOT}/step1/` scripts:
`build_amos_train_csv` → `resampled_wl_img_msk_amos.py` (preprocess) → `remap_amos_train_mask_ts.py`
(TS label remap) → `build_amos_finetune_manifest.py`. You should not need these — the manifest already exists.
