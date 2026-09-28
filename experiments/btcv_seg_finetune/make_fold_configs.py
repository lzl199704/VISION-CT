#!/usr/bin/env python3
"""Expand a BASE BTCV config into 5 per-fold configs.

For each fold k in 0..4 it writes a config with:
  input_file  -> ${DATA_ROOT}/step1/btcv_cv5/fold{k}_trainval.parquet  (18 train + 6 val)
  output_dir  -> <base output_dir>/fold{k}
  logging_dir -> <base output_dir>/fold{k}/logs

Usage:  python experiments/btcv_seg_finetune/make_fold_configs.py configs/btcv_base_lr3e5_3k.yaml
Writes: configs/_generated/<base_name>_fold{0..4}.yaml  and prints the train commands.
"""
import sys, os, yaml, copy

CV = "${DATA_ROOT}/step1/btcv_cv5"

def main():
    base_path = sys.argv[1]
    base = yaml.safe_load(open(base_path))
    name = os.path.splitext(os.path.basename(base_path))[0]
    out_root = base["hf_TrainingArguments"]["output_dir"].rstrip("/")
    gen_dir = os.path.join(os.path.dirname(base_path), "_generated")
    os.makedirs(gen_dir, exist_ok=True)
    made = []
    for k in range(5):
        cfg = copy.deepcopy(base)
        cfg["input_file"] = f"{CV}/fold{k}_trainval.parquet"
        cfg["hf_TrainingArguments"]["output_dir"] = f"{out_root}/fold{k}"
        cfg["hf_TrainingArguments"]["logging_dir"] = f"{out_root}/fold{k}/logs"
        p = os.path.join(gen_dir, f"{name}_fold{k}.yaml")
        yaml.safe_dump(cfg, open(p, "w"), sort_keys=False)
        made.append(p)
    print(f"wrote {len(made)} fold configs to {gen_dir}/")
    print("\n# train all 5 folds (edit --nproc_per_node to your GPU count):")
    for k, p in enumerate(made):
        print(f"torchrun --nproc_per_node=8 bin/train_semantic_seg_two_stage.py --config_file {p}")

if __name__ == "__main__":
    main()
