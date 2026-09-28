# Model weights

Three checkpoints are released (Hugging Face `model.safetensors` + `config.json` + tokenizer files;
no optimizer state), as a single download folder:

**Download:** <GOOGLE_DRIVE_LINK>

| Folder | Pipeline | Stage | Training checkpoint | Size |
|---|---|---|---|---|
| `chest_multimodal` | Chest | one multimodal stage | `checkpoint-75000` | 0.83 GB |
| `abd_step1_seg_pretrain` | Abdomen | Step 1 — segmentation pretraining | `checkpoint-10000` | 0.85 GB |
| `abd_step2_contrast` | Abdomen | Step 2 — multimodal finetuning (initialised from Step 1) | `checkpoint-50000` | 0.85 GB |

Each folder also contains the `train_config.yaml` used for that run (the same files are in
`configs/`). Verify a download with `sha256sum -c SHA256SUMS.txt` inside the folder.

To use a checkpoint, pass its folder to the corresponding script (`--checkpoint <weights>/<folder>`,
see the README); the scripts load `model.safetensors` directly.
