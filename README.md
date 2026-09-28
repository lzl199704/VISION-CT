# Uniferum — segmentation-grounded multimodal CT model

Reference implementation for the Uniferum / RadViLLA CT pathology model: a multimodal
architecture that couples a 3D vision encoder–decoder (segmentation) with a language model
(pathology question answering), trained in two stages for the abdomen and a single stage for
the chest.

This repository contains the **training / evaluation / inference scripts** and their model,
dataset and image-processing modules. Model weights are documented in [`WEIGHTS.md`](WEIGHTS.md);
the train/test data manifests are kept separately (see **Data** below).

## Pipelines

| Pipeline | Stages | Train script | Eval / predict script | Config |
|---|---|---|---|---|
| **Chest** | one stage (multimodal) | `bin/train_uniferum_radvilla_combined_chest.py` | `bin/predict_uniferum_combined_v2.py` | `configs/chest_train_config.yaml` |
| **Abdomen — Step 1** | segmentation pretrain | `bin/train_semantic_seg_two_stage.py` (stage 1) | `bin/eval_pretrain_seg.py` | `configs/abd_step1_pretrain_seg_config.yaml` |
| **Abdomen — Step 2 (contrast)** | multimodal finetune | `bin/train_semantic_seg_two_stage.py` (stage 2) | `bin/eval_finetune.py` | `configs/abd_step2_contrast_config.yaml` |
| **Abdomen — Step 2 (non-contrast)** | multimodal finetune | `bin/train_semantic_seg_two_stage.py` (stage 2) | `bin/eval_finetune.py` | `configs/abd_step2_noncontrast_config.yaml` |
| **Abdomen — Step 2 (mixed, stones)** | multimodal finetune | `bin/train_semantic_seg_two_stage.py` (stage 2) | `bin/eval_finetune.py` | `configs/abd_step2_mixncstone_config.yaml` |
| **Chest + Abdomen (combined) — Step 1** | segmentation pretrain | `bin/train_semantic_seg_two_stage.py` (stage 1) | `bin/eval_pretrain_seg.py` | `configs/chest_abd_step1_pretrain_seg_config.yaml` |
| **Chest + Abdomen (combined) — Step 2** | multimodal finetune | `bin/train_semantic_seg_two_stage.py` (stage 2) | `bin/eval_finetune.py` | `configs/chest_abd_step2_finetune_config.yaml` |

The abdomen is **two-stage**: Step 1 pretrains the vision encoder–decoder on organ segmentation,
then Step 2 finetunes the full multimodal model for pathology QA (initialised from the Step-1
vision decoder). The non-contrast Step-2 model shares the same Step-1 segmentation weights as the
contrast model. The chest is a **single multimodal stage**.

A third abdomen Step-2 run, `abd_step2_mixncstone`, is trained on contrast and non-contrast
studies pooled and provides the reported results for the two stone findings (gallbladder
gallstone, kidney renal calculus); see `WEIGHTS.md`.

**Combined chest + abdomen model (2026-09-03).** One two-stage model trained on chest, abdomen-contrast
and abdomen-non-contrast data pooled (40 findings), using the abdomen (segmentation-grounded) architecture
and recipe. Step 1 pretrains the encoder–decoder on 15 organs — the original 11 abdominal ones plus the 4
chest targets used by the established chest recipe (lung = the 5 TotalSegmentator lobe labels, heart,
thoracic aorta, and the 47-label `chest` catch-all region) — balanced to a fixed number of rows per organ; Step 2 finetunes for classification + segmentation on the plain union of the three existing train
sets (same splits as before). Splits, harmonisation and the overlap report are documented in
`data_splits/combined/MANIFEST.md`; the builder is `data_splits/build_combined_splits.py`. Step 2 runs
125k steps = the 50k (abdomen) + 75k (chest) specialist budgets, so the combined model can be compared to
the two specialists at equal total compute.

## Directory layout

```
uniferum_release/
├── bin/                     # entry-point scripts + shared utils
│   ├── train_uniferum_radvilla_combined_chest.py   # chest: train
│   ├── predict_uniferum_combined_v2.py             # chest: inference
│   ├── train_semantic_seg_two_stage.py            # abdomen: train (step 1 & 2)
│   ├── eval_pretrain_seg.py                        # abdomen: eval step-1 segmentation (Dice)
│   ├── eval_finetune.py                            # abdomen: eval step-2 (AUROC / detection)
│   └── utils.py                                    # config loading, transforms, checkpoint I/O
├── models/                  # Uniferum model + vision backbone + losses
│   ├── combined_multimodal_models.py               # chest Uniferum
│   ├── combined_multimodal_models_semantic_v3.py   # abdomen Uniferum (segmentation-grounded)
│   ├── vision_models.py, loss_fcts.py, utils.py
├── data_utils/              # VQA + mask datasets, anatomical label map
├── img_utils/               # CT volume loading (.zst) and windowing/preprocessing
├── configs/                 # train_config.yaml for each model (incl. chest_abd_* combined two-step)
├── requirements.txt
└── WEIGHTS.md               # model-weight locations
```

Data manifests are **not** part of this code release; they are kept separately (see **Data**).

Scripts import modules as `bin.utils`, `models.*`, `data_utils.*`, `img_utils.*`, so **run them
from the repository root** (they call `sys.path.append(os.getcwd())`).

## Data

Train/test data manifests are **kept separately from this code release** (they contain internal
patient IDs and file paths). An organized copy currently lives at
`${HOME}/projects/RadVILLA/uniferum_data/`, grouped by pipeline:

| Folder | Train | Test |
|---|---|---|
| `chest/` | `train_val.parquet` | `test_{ermi,sinai,ctrate,dubai}.parquet` |
| `abd_contrast/` | `train_val_step1.parquet`, `train_val_step2.parquet` | `test_{ermi,sinai,merlin,dubai}.parquet` |
| `abd_noncontrast/` | `train_val.parquet` | `test_{ermi,sinai,merlin}.parquet` |
| `step1_seg/` | — | `amos_eval.parquet`, `btcv_eval.parquet` (organ-segmentation eval) |
| `data_splits/combined/` (this repo) | `Sinai_ERMI_chest_abd_train_val_step1_seg_0903.parquet`, `Sinai_ERMI_chest_abd_train_val_step2_0903.parquet` | `{ERMI,Sinai}_chest_abd_test_0903.parquet` (internal), `CTRATE_chest_test_0903.parquet`, `Merlin_abd_test_0903.parquet` (public) |

Each row is one (patient, finding) question with the CT volume path, the QA question, the binary
label, and the segmentation-mask path. ERMI / Sinai = internal institutions; Merlin / CT-RATE =
public benchmarks; Dubai = external. Point each script's `--parquet` (and the `input_file:` in the
`configs/*.yaml`) at the manifest you want to use.

## Usage (run from repo root)

```bash
pip install -r requirements.txt

# --- Chest (one stage) ---
python bin/train_uniferum_radvilla_combined_chest.py --config configs/chest_train_config.yaml
python bin/predict_uniferum_combined_v2.py --config configs/chest_train_config.yaml \
    --checkpoint <chest_ckpt>/checkpoint-75000 --parquet data/chest/test_ermi.parquet --out preds.parquet

# --- Abdomen Step 1: segmentation pretrain ---
python bin/train_semantic_seg_two_stage.py --config configs/abd_step1_pretrain_seg_config.yaml
python bin/eval_pretrain_seg.py --config configs/abd_step1_pretrain_seg_config.yaml \
    --checkpoint <abd_step1>/checkpoint-10000 --parquet data/step1_seg/amos_eval.parquet --out amos_dice.parquet

# --- Abdomen Step 2: multimodal finetune (contrast) ---
python bin/train_semantic_seg_two_stage.py --config configs/abd_step2_contrast_config.yaml
python bin/eval_finetune.py --config configs/abd_step2_contrast_config.yaml \
    --checkpoint <abd_step2>/checkpoint-50000 --parquet data/abd_contrast/test_ermi.parquet --out ermi_pred.parquet

# --- Chest + Abdomen combined (two stages, DDP on GPUs 0-3; see "Environment" above) ---
python data_splits/build_combined_splits.py --tag 0903        # rebuild splits (CPU, ~8 min: reads masks)

DK="docker run --rm --gpus \"device=0,1,2,3\" --shm-size 64g --ipc=host --ulimit memlock=-1 \
  --entrypoint torchrun -e CUDA_DEVICE_ORDER=PCI_BUS_ID -e HF_HOME=/hf -e OMP_NUM_THREADS=8 \
  -v /raid:/raid -v $PWD:$PWD -v $HOME/.cache/huggingface:/hf -w $PWD uniferum_train:latest"
eval $DK --nproc_per_node 4 bin/train_semantic_seg_two_stage.py \
    --config_file configs/chest_abd_step1_pretrain_seg_config.yaml     # 15k steps, ~7 h
eval $DK --nproc_per_node 4 bin/train_semantic_seg_two_stage.py \
    --config_file configs/chest_abd_step2_finetune_config.yaml         # 125k steps, ~60 h (needs Step 1)

python bin/eval_finetune.py --run_dir ${WEIGHTS_ROOT}/Step2_Finetune_Multimodal_0903_chest_abd \
    --ckpt_name checkpoint-125000 --val_file data_splits/combined/ERMI_chest_abd_test_0903.parquet --prefix ermi_test
# repeat --val_file for Sinai_chest_abd_test_0903 / CTRATE_chest_test_0903 / Merlin_abd_test_0903;
# slice the output parquet by `region` for per-region metrics.
```
(Exact flag names are defined by each script's argument parser — check `--help`.)

## Environment (2026-09-03)
No conda env on this host has the complete training stack (`llava-med` has transformers 4.41, too old for
the ModernBERT tokenizer; `${DATA_ROOT}/vlmenv` has a torch that needs `libcudart.so.11.0`). Training and
evaluation run in the **`uniferum_train` Docker image**, built from [`Dockerfile.train`](Dockerfile.train)
= the pinned `vlm3d_abnclass_r8` stack (python 3.10, torch 2.3.0+cu121, transformers 4.45, monai 1.3,
timm 1.0.3, peft) plus `timm_3d`, `pyarrow`, `matplotlib`, `SimpleITK`, `acvl_utils`, `zstandard`:

```bash
docker build -f Dockerfile.train -t uniferum_train:latest .
```

The container runs as uid 999, so mount world-readable paths and keep scratch/output under `/raid`.
Mount the repo at its own absolute path so the absolute `input_file:` in each config resolves inside the
container. `data_splits/smoke_test_combined_dataset.py` (CPU, no GPU) checks that the image and the
combined parquets load correctly.

### GPU note — two broken NVLink P2P pairs
NCCL fails with `unhandled cuda error` on any GPU set containing **both 4 and 5, or both 6 and 7**;
`NCCL_DEBUG=INFO` reports *"P2P is disabled between NVLINK connected GPUs 5 and 4 ... probably due to a
hardware issue"*. Raw peer copies, fabricmanager and the NV18 topology are otherwise healthy, so this is
hardware state rather than a container problem. **8-GPU runs are therefore impossible** until it is fixed
(`NCCL_P2P_DISABLE=1` works but is 3x slower than a single GPU). Use **GPUs 0-3**, a clean full-NVLink
group; 0,1,2,3,4,6 and 0,1,2,3,5,7 also work.

Also prefer **DDP via `torchrun`** over HF Trainer's default DataParallel, which rebroadcasts the whole
model every step. Measured on Step-1 data over 150 steps: 18.6 samples/s (0.58 optimizer steps/s) at effective
batch 32 on 4 GPUs, peak 43.5 GB per GPU. The combined configs already set `gradient_accumulation_steps: 2` and
`ddp_find_unused_parameters: true` so 4 GPUs reproduce the published effective batch of 32.

## Weights
See [`WEIGHTS.md`](WEIGHTS.md). The LLM backbone is `Simonlee711/Clinical_ModernBERT` (pulled from
HuggingFace Hub at runtime).

## Notes for public release
- This repo ships **code + weight locations only**; data manifests are kept separately and are
  not part of the release.
- Upload model checkpoints to HF Hub / Zenodo and update `WEIGHTS.md`.
- De-identify the data manifests before distributing them anywhere.
