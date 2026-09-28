# Foundation-encoder fine-tuning — Merlin & CT-CLIP downstream

**Goal.** Benchmark two *published 3D CT foundation vision encoders* as downstream
representations on a CT classification task, next to the chest-Uniferum vs ImageNet
comparison in [`../vision_encoder_finetune`](../vision_encoder_finetune):

| encoder_source | model | pretraining | feature | native input |
|---|---|---|---|---|
| `merlin` | Merlin (Stanford) I3-ResNet152 | abdominal CT + reports/EHR | 2048-d image embedding | 224×224×160, spacing (1.5,1.5,3), HU[-1000,1000]→[0,1] |
| `ctclip` | CT-CLIP_v2 CTViT | CT-RATE chest CT + reports | 512-d mean-pooled tokens | 240×480×480, spacing (0.75,0.75,1.5), HU[-1000,1000]→[-1,1] |

Unlike the sibling kit (same architecture, Uniferum-vs-ImageNet init), **these are different
architectures with their own preprocessing** — so this answers "*which pretrained encoder transfers
best?*", not a controlled init comparison. Each encoder loads its published weights, preprocesses the
**raw** volume with its own recipe (monai; reorients LAS→RAS), emits a pooled feature, and we train a
classification head on top.

```
foundation_encoder_finetune/
├── README.md
├── train_foundation_classifier.py     # encoder plug-ins + head + linear-probe / full-FT loop
└── configs/
    ├── lung1_5yr_survival_merlin.yaml   ├── lung1_5yr_survival_ctclip.yaml
    └── lung1_histology_merlin.yaml      └── lung1_histology_ctclip.yaml
```

## 1. Environments (each encoder needs its own venv — conflicting deps)
| encoder | python |
|---|---|
| `merlin` | `python  # (Merlin environment)` (merlin-vlm, torch cu121, monai 1.3.2, numpy<2) |
| `ctclip` | `python  # (CT-CLIP environment)` (ct_clip + transformer_maskgit, monai, numpy<2) |

Both have `torch, monai, nibabel, pandas, scikit-learn, pyyaml`. Merlin weights auto-download from HF
(`stanfordmimi/Merlin`, cached in the venv). CT-CLIP uses `ctclip_ckpt: ${BASELINES_ROOT}/CT-CLIP/CT-CLIP_v2.pt`.

## 2. Data
Configs point at raw LUNG1 / NSCLC-Radiomics volumes via a `nifti_template` (built from the CSV
`PatientID`), so each encoder does its *own* preprocessing — do **not** reuse the 256³ Uniferum npz here.
- `nifti_template: .../NSCLC-Radiomics/image/{pid}/{pid}.nii.gz` (raw HU, LAS → reoriented to RAS)
- Labels/splits: the same CSVs as the sibling kit (`lung1_5yr_survival_*` binary; `nsclc_histology_*_preprocessed` multiclass, 4 classes).

## 3. Run
```bash
cd ${HOME}/projects/RadVILLA/uniferum_release
# Merlin
python  # (Merlin environment) experiments/foundation_encoder_finetune/train_foundation_classifier.py \
    --config experiments/foundation_encoder_finetune/configs/lung1_5yr_survival_merlin.yaml
# CT-CLIP
python  # (CT-CLIP environment) experiments/foundation_encoder_finetune/train_foundation_classifier.py \
    --config experiments/foundation_encoder_finetune/configs/lung1_5yr_survival_ctclip.yaml
```
Prints a final **TEST** line (best-val head) and saves `head_best.pt`.

## 4. Full fine-tune (default) vs linear probe (`freeze_encoder`)
"Fine-tune the vision encoder" here = **extract only the pretrained vision encoder, add a fresh
classification head, and train them together.** The text tower / contrastive projections / EHR head are
dropped (see `build_merlin` / `build_ctclip`).
- **`freeze_encoder: false` (DEFAULT) — full fine-tune.** Trains the extracted encoder **and** the head
  end-to-end, with discriminative LR (encoder LR = `learning_rate * encoder_lr_mult`). This is the
  headline setting. CT-CLIP's CTViT at 480³ is memory-heavy → configs use `batch_size: 1`; Merlin
  (224³) tolerates `batch_size: 4`.
- **`freeze_encoder: true` — linear probe.** Encoder frozen, encoded once, features cached, only the
  head trains (fast; a useful representation-quality sanity check). Validated end-to-end: Merlin
  LUNG1 5-yr-survival linear-probe test AUROC ≈ 0.70.

## 5. Hyperparameter sweep (keep identical across encoders for a fair table)
Linear probe: sweep head `learning_rate` (3e-3, 1e-3, 3e-4), `weight_decay`, `dropout`. For imbalanced
binary (LUNG1 survival ~18.5% positive) tune `pos_weight` (≈ n_neg/n_pos). Select on **val**, report
**test** (AUROC for binary/multilabel; accuracy + macro-AUROC for multiclass). Log
`(encoder_source, freeze, HP) → val, test`. Headline = best of each encoder on test.

## 6. Adding an encoder or task
- New encoder: add a `build_*`, a monai `*_transform`, and an `arrange_*` (volume → model input), then a
  row in the `ENCODERS` registry. Return `(module, feature_dim, feature_fn)`.
- New task/dataset: point the CSVs + `nifti_template` at your data; set `task` (binary/multiclass/multilabel)
  as in the sibling kit's README §2.

## 7. Cross-kit comparison
For a single downstream table across all four encoders, run **Uniferum** and **ImageNet** with the
sibling kit (`../vision_encoder_finetune`, timm_3d backbone) and **Merlin** and **CT-CLIP** here, on the
same LUNG1 splits, and compare test AUROC. Note the encoders differ in architecture and input size, so
report those alongside the numbers.
