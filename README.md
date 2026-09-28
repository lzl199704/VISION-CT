# VISION-CT — segmentation-grounded multimodal CT model

Reference implementation of VISION-CT: a multimodal architecture that couples a 3D vision
encoder–decoder (organ segmentation) with a language model (pathology question answering).
Two models are released: a **chest** model trained in a single multimodal stage, and an
**abdomen** model trained in two stages (segmentation pretraining, then multimodal finetuning).

This repository contains the training, evaluation and inference scripts with their model, dataset
and image-processing modules. Model weights are distributed separately — see [`WEIGHTS.md`](WEIGHTS.md).
Training and test manifests are not part of the release (they contain institutional patient
identifiers and file paths).

## Pipelines

| Pipeline | Stage | Train script | Eval / predict script | Config |
|---|---|---|---|---|
| **Chest** | one multimodal stage | `bin/train_visionct_combined_chest.py` | `bin/predict_visionct_combined_v2.py` | `configs/chest_train_config.yaml` |
| **Abdomen — Step 1** | segmentation pretraining | `bin/train_semantic_seg_two_stage.py` | `bin/eval_pretrain_seg.py` | `configs/abd_step1_pretrain_seg_config.yaml` |
| **Abdomen — Step 2** | multimodal finetuning | `bin/train_semantic_seg_two_stage.py` | `bin/eval_finetune.py` | `configs/abd_step2_contrast_config.yaml` |

Abdomen Step 1 pretrains the vision encoder–decoder on organ segmentation; Step 2 initialises
the full multimodal model from the Step-1 vision decoder and finetunes it for pathology question
answering together with segmentation. The chest model is trained in a single multimodal stage.

## Directory layout

```
VISION-CT/
├── bin/                                        # entry-point scripts + shared utils
│   ├── train_visionct_combined_chest.py        # chest: train
│   ├── predict_visionct_combined_v2.py         # chest: inference
│   ├── train_semantic_seg_two_stage.py         # abdomen: train (step 1 and step 2)
│   ├── eval_pretrain_seg.py                    # abdomen: evaluate step-1 segmentation (Dice)
│   ├── eval_finetune.py                        # abdomen: evaluate step-2 (AUROC / detection)
│   └── utils.py                                # config loading, transforms, checkpoint I/O
├── models/                                     # VISION-CT model, vision backbone, losses
│   ├── combined_multimodal_models.py           # chest VISION-CT
│   ├── combined_multimodal_models_semantic_v3.py  # abdomen VISION-CT (segmentation-grounded)
│   └── vision_models.py, loss_fcts.py, utils.py
├── data_utils/                                 # VQA + mask datasets, anatomical label map
├── img_utils/                                  # CT volume loading (.zst) and windowing
├── configs/                                    # training config for each of the three stages
├── Dockerfile.train                            # pinned training/evaluation environment
├── requirements.txt
└── WEIGHTS.md                                  # released checkpoints
```

Scripts import modules as `bin.utils`, `models.*`, `data_utils.*`, `img_utils.*`, so run them
from the repository root (they call `sys.path.append(os.getcwd())`).

## Data format

Each manifest is a parquet table with one row per (study, finding) question: the CT volume path,
the question text, the binary label and the segmentation-mask path. Point each script's
`--parquet` argument (and `input_file:` in `configs/*.yaml`) at your own manifest; the configs use
`${DATA_ROOT}` / `${WEIGHTS_ROOT}` placeholders for the data and checkpoint locations.

## Usage (run from the repository root)

```bash
pip install -r requirements.txt          # or: docker build -f Dockerfile.train -t visionct_train:latest .

# --- Chest (one stage) ---
python bin/train_visionct_combined_chest.py --config configs/chest_train_config.yaml
python bin/predict_visionct_combined_v2.py --config configs/chest_train_config.yaml \
    --checkpoint <weights>/chest_multimodal --parquet <test>.parquet --out preds.parquet

# --- Abdomen Step 1: segmentation pretraining ---
python bin/train_semantic_seg_two_stage.py --config configs/abd_step1_pretrain_seg_config.yaml
python bin/eval_pretrain_seg.py --config configs/abd_step1_pretrain_seg_config.yaml \
    --checkpoint <weights>/abd_step1_seg_pretrain --parquet <seg_eval>.parquet --out dice.parquet

# --- Abdomen Step 2: multimodal finetuning (initialised from Step 1) ---
python bin/train_semantic_seg_two_stage.py --config configs/abd_step2_contrast_config.yaml
python bin/eval_finetune.py --config configs/abd_step2_contrast_config.yaml \
    --checkpoint <weights>/abd_step2_contrast --parquet <test>.parquet --out preds.parquet
```
Exact flag names are defined by each script's argument parser — see `--help`.

## Environment

```bash
conda create -n visionct python=3.10 -y && conda activate visionct
pip install -r requirements.txt
```
`requirements.txt` pins the versions used for training and evaluation (torch 2.3.0+cu121,
transformers, monai, timm, peft, pyarrow, SimpleITK, zstandard). A CUDA-capable GPU is required;
multi-GPU training uses DDP via `torchrun`. Alternatively build the pinned image with
`docker build -f Dockerfile.train -t visionct_train:latest .`. The language backbone
`Simonlee711/Clinical_ModernBERT` is downloaded from the Hugging Face Hub at first run.

## Weights

See [`WEIGHTS.md`](WEIGHTS.md).
