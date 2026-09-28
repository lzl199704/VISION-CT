# Model weights

Weights are large (5–15 GB per model folder; ~1 GB per individual checkpoint) and are **not** included in this repository. They live on the
lab storage at the paths below; each checkpoint directory also contains the `train_config.yaml`
used to build the model (copied into `configs/` here for convenience).

| Pipeline | Stage | Checkpoint dir | Paper checkpoint | Size |
|---|---|---|---|---|
| **Chest** (one-stage) | multimodal | `${WEIGHTS_ROOT}/Step1_chest_ermi_sinai_0504_NO_AUG_updatedQA/` | `checkpoint-75000` | 15 GB |
| **Abdomen (contrast)** | Step 1 — segmentation pretrain | `${WEIGHTS_ROOT}/Step1_Pretrain_Seg_0328_abd/` | `checkpoint-10000` | 5.0 GB |
| **Abdomen (contrast)** | Step 2 — multimodal finetune | `${WEIGHTS_ROOT}/Step2_Finetune_Multimodal_0508_abd_r1/` | `checkpoint-50000` | 10 GB |
| **Abdomen (non-contrast)** | Step 1 — segmentation pretrain | `${WEIGHTS_ROOT}/Step1_Pretrain_Seg_0328_abd/` (shared with contrast) | `checkpoint-10000` | 5.0 GB |
| **Abdomen (non-contrast)** | Step 2 — multimodal finetune | `${WEIGHTS_ROOT}/Step2_Finetune_Multimodal_abd_0512_r1_noncon/` | `checkpoint-15000` | 5.0 GB |
| **Abdomen (mixed contrast + non-contrast) — stones** | Step 2 — multimodal finetune | `${WEIGHTS_ROOT}/Step2_Finetune_Multimodal_abd_0727_r1_mixncstone/` | `checkpoint-15000` | 5.0 GB |

Notes
- The abdomen Step-1 segmentation-pretrain weights are **shared** by both the contrast and
  non-contrast Step-2 models (Step 2 is initialised from the Step-1 vision decoder).
- **Stone findings use the mixed model.** `…_0727_r1_mixncstone` is trained on contrast and
  non-contrast studies pooled, and supersedes *both* abdomen Step-2 models for the two stone
  findings — **Gallbladder gallstone** and **Kidney renal calculus**. These are the numbers
  reported in the paper for those findings (`csv/stage2_abd_stone_perfinding.csv`).
- The `…_0512_r1_noncon` checkpoint is still required: after the stone findings move to the
  mixed model it remains the source for the remaining non-contrast finding
  (Liver fatty infiltration). It is superseded for stones only, not replaced outright.
- Each `checkpoint-*/` holds HuggingFace-Trainer state (`model.safetensors`, optimizer, etc.).
- For public release, upload the checkpoint directories to a model host (HF Hub / Zenodo) and
  update the paths above.
