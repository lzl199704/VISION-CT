# Vision-encoder fine-tuning — chest-Uniferum vs ImageNet

**Goal.** Test whether the **chest-Uniferum vision encoder** (learned from chest CT + reports) is a
better downstream representation than the **same architecture initialised from ImageNet**. We take an
identical 3D backbone (timm_3d `tf_efficientnetv2_b0`, `in_chans=1`), attach a classification head,
fine-tune on a downstream classification task, and compare the two initialisations under identical
hyperparameters. Any accuracy gap is attributable to the pre-training.

```
vision_encoder_finetune/
├── README.md
├── train_classifier.py          # backbone + head, loads either encoder, trains, prints test metrics
└── configs/
    ├── uniferum_encoder.yaml     # encoder_source: uniferum (checkpoint-75000)
    └── imagenet_encoder.yaml     # encoder_source: imagenet  (same HP — the control)
```

The chest-Uniferum weights come from `checkpoint-75000/model.safetensors` (the script loads the 418
`vision_model.model.*` keys into the backbone). ImageNet is `timm_3d ... pretrained=True`.

## 1. Your CSVs
Prepare `train.csv`, `val.csv`, `test.csv` (patient-disjoint). Each row = one study:
- `img_file_path` — path to the preprocessed CT volume **`.npz`** (256×256×Z, same format as the model's data)
- label column(s) — see the three task types below.

Point `train_csv`/`val_csv`/`test_csv` and `img_col` in the config at your files.

## 2. Choose the classification head (edit the config `task`)
The head is `Linear(1280 → num_outputs)`; the **task** sets `num_outputs`, the loss, the output
activation, and the metric. Change only the config — no code edits needed for these three cases:

| task | CSV label | config fields | head output | loss | metric |
|---|---|---|---|---|---|
| **binary** | one 0/1 column | `task: binary`, `label_col: label` | 1 | `BCEWithLogits` (sigmoid) | AUROC |
| **multiclass** (mutually exclusive) | one integer column 0..K-1 | `task: multiclass`, `label_col: label`, `num_classes: K` | K | `CrossEntropy` (softmax) | accuracy + macro-AUROC |
| **multi-label** (independent findings) | one 0/1 column **per label** | `task: multilabel`, `label_cols: [a, b, c]` | len(label_cols) | `BCEWithLogits` per label (sigmoid) | per-label + macro AUROC |

Key point: **multi-label ≠ multi-class.** Multi-class uses softmax + CrossEntropy (exactly one class per
study). Multi-label uses **sigmoid + BCE per label** (a study can have several findings, or none) — this
is the usual setting for CT findings. The script already implements all three; you just set `task`.

Deeper head (optional): edit `Classifier.__init__` in `train_classifier.py` (marked `EDIT THE HEAD HERE`),
e.g. `nn.Sequential(nn.Linear(feat,512), nn.SiLU(), nn.Dropout(d), nn.Linear(512,num_outputs))`. Keep it
identical across both encoder sources.

## 3. Run (from the release repo root)
```bash
cd ${HOME}/projects/RadVILLA/uniferum_release
# Uniferum encoder
python experiments/vision_encoder_finetune/train_classifier.py \
    --config experiments/vision_encoder_finetune/configs/uniferum_encoder.yaml
# ImageNet control (same hyperparameters)
python experiments/vision_encoder_finetune/train_classifier.py \
    --config experiments/vision_encoder_finetune/configs/imagenet_encoder.yaml
```
Each prints per-epoch val metrics and a final **TEST** line (best-val checkpoint), and saves `best.pt`.
Run both with **identical** hyperparameters — the comparison is only fair if the sole difference is
`encoder_source`.

## 4. Hyperparameter-tuning strategy (downstream)
Do it in two stages, and run every setting for **both** encoders:

**Stage A — linear probe (encoder frozen, `freeze_encoder: true`).**
Trains only the head, so it measures representation quality directly and is cheap. Sweep `learning_rate`
(1e-3, 3e-4, 1e-4), `weight_decay`, `dropout`. This alone often answers "is the Uniferum encoder better?"

**Stage B — full fine-tune (`freeze_encoder: false`).**
Use **discriminative LR**: head at `learning_rate`, encoder at `learning_rate * encoder_lr_mult`
(start `encoder_lr_mult: 0.1` so the pretrained encoder isn't wiped out). Sweep, in rough priority:
1. `learning_rate` (head): 3e-4, 1e-4, 3e-5
2. `encoder_lr_mult`: 0.01, 0.1, 0.3, 1.0
3. `epochs` / early-stopping (the script keeps the best-val checkpoint automatically)
4. `weight_decay` (0.0, 0.01, 0.05), `dropout` (0.0–0.3)
5. `batch_size` (2/4/8 — memory permitting) and, for multi-label class imbalance, consider `pos_weight` in the BCE loss
6. augmentation (add flips/intensity jitter in the dataset if under-fitting-free but overfitting)

**Rules for a valid comparison**
- Same CSVs, same `window_*`, `input_size`, and HP grid for both encoders.
- Select on **validation**, report **test** (the script does best-val selection).
- Report the metric matching the task (AUROC for binary/multi-label; accuracy + macro-AUROC for multi-class),
  and keep a log: `(encoder_source, stage, HP) → val, test`.
- Headline result = best Uniferum vs best ImageNet on the **test** set, both selected on val.

## 5. Environment
Run with the model env (has `torch`, `timm_3d`, `safetensors`, `pandas`, `scikit-learn`), e.g.
`python`. Needs one GPU; 256³ volumes are memory-heavy, so start with
`batch_size: 2–4` (reduce `input_size` to 224/192 only if you must, but keep it identical across encoders).

## 6. Worked example — LUNG1 / NSCLC-Radiomics (ready to run)

A concrete downstream benchmark with data + splits already prepared on the cluster
(`${BENCHMARKS_ROOT}/NSCLC-Radiomics/`). Two tasks ship as configs:

| Config | Task | CSVs | npz | notes |
|---|---|---|---|---|
| `lung1_5yr_survival_{uniferum,imagenet}.yaml` | binary (5-yr overall survival, >1826 d) | `lung1_5yr_survival_{train,val,test}.csv` (155/39/200) | `npz_5yr_survival/` **raw HU** | ~18.5% positive → `pos_weight: 3.4`; volume windowed at load (`prewindowed: false`) |
| `lung1_histology_{uniferum,imagenet}.yaml` | multiclass (4 histology classes) | `nsclc_histology_{train,val,test}_preprocessed.csv` (256/55/55) | `npz_histology_preprocessed/` **already [0,1]** | `prewindowed: true` (no re-windowing) |

Two config knobs added for these:
- **`prewindowed`** — `true` if the npz is already normalised to [0,1] (skips HU windowing); `false` for raw-HU npz (applies `window_center`/`window_width`). Must be identical across both encoders.
- **`pos_weight`** — upweights positives in the BCE loss for imbalanced binary/multilabel targets. Scalar for binary (≈ n_neg/n_pos), list-per-label for multilabel. Omit to disable.

Run (from the release repo root), both encoders with the **same** grid:
```bash
cd ${HOME}/projects/RadVILLA/uniferum_release
# 5-yr survival
python experiments/vision_encoder_finetune/train_classifier.py --config experiments/vision_encoder_finetune/configs/lung1_5yr_survival_uniferum.yaml
python experiments/vision_encoder_finetune/train_classifier.py --config experiments/vision_encoder_finetune/configs/lung1_5yr_survival_imagenet.yaml
# histology
python experiments/vision_encoder_finetune/train_classifier.py --config experiments/vision_encoder_finetune/configs/lung1_histology_uniferum.yaml
python experiments/vision_encoder_finetune/train_classifier.py --config experiments/vision_encoder_finetune/configs/lung1_histology_imagenet.yaml
```
Headline = best Uniferum vs best ImageNet on **test**, both selected on val (see §4 for the sweep).
Note: the data root is local per-machine disk — make sure the `uniferum_ckpt` safetensors exists at that
path on the machine you run on (copy it there if not). The prior intern handoff (data pipeline, GTV-mask
regeneration, cohort/label definitions) lives in `NSCLC-Radiomics/HANDOFF.md`.
