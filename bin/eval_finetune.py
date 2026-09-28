"""
Evaluation script for Stage 2 (Fine-tuned Multimodal Model).
Evaluates both classification and segmentation on the TEST split.

Metrics reported:
  Classification : Accuracy, AUC, F1, Precision, Recall  (overall + per-organ)
  Segmentation   : Dice score  (overall + per-organ)

Usage:
    python bin/eval_finetune.py \
        --run_dir vision_models/Step2_Finetune_Multimodal_0310 \
        --val_file Sinai_ERMI_abd_train_val_0220.parquet \
        --devices 0,1,2,3,4,5,6,7 \
        [--ckpt_name checkpoint-50000] \
        [--save_masks] \
        [--prefix eval_finetune]
"""

import glob
import os
from typing import Optional

import nibabel as nib
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tap import Tap
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

from bin.utils import create_transform, load_yaml
from data_utils.combined_vqa_dataset_semantic import VQABinaryDataCollator, VQAMaskDataset
from models.combined_multimodal_models_semantic_v3 import Uniferum
from models.utils import load_safetensors

mp.set_start_method("spawn", force=True)


class Arguments(Tap):
    run_dir: str
    val_file: str
    ckpt_name: Optional[str] = None  # If None, evaluates the latest checkpoint
    devices: str = "0,1,2,3,4,5,6,7"
    prefix: str = "eval_finetune"
    save_masks: bool = False  # Set True to save NIfTI prediction masks
    cls_threshold: float = 0.5  # Sigmoid threshold for binary classification


def get_ckpt_files(args):
    if args.ckpt_name is not None:
        ckpt_file = os.path.join(args.run_dir, args.ckpt_name, "model.safetensors")
        yield ckpt_file, args.ckpt_name
    else:
        search_str = os.path.join(args.run_dir, "checkpoint-*/model.safetensors")
        ckpt_files = glob.glob(search_str)
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoints found in {args.run_dir}")
        ckpt_files.sort(key=lambda x: int(os.path.basename(os.path.dirname(x)).split("-")[1]))
        ckpt_file = ckpt_files[-1]
        ckpt_name = os.path.basename(os.path.dirname(ckpt_file))
        yield ckpt_file, ckpt_name


def compute_dice_score(pred, target, smooth=1e-4):
    pred = np.nan_to_num(pred).astype(np.float32)
    target = np.nan_to_num(target).astype(np.float32)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    if union < smooth:
        return 1.0  # Both empty → perfect prediction
    return float((2.0 * intersection + smooth) / (union + smooth))


def save_nifti_prediction(pred_vol, patient_id, organ, output_dir):
    try:
        ni_img = nib.Nifti1Image(pred_vol.astype(np.float32), affine=np.eye(4))
        fname = f"{patient_id}_{organ}_pred.nii.gz"
        save_path = os.path.join(output_dir, fname)
        nib.save(ni_img, save_path)
        return save_path
    except Exception as e:
        print(f"Error saving NIfTI for {patient_id}: {e}")
        return ""


def evaluate_worker(
    rank: int,
    config: dict,
    df_chunk: pd.DataFrame,
    ckpt_file: str,
    outputs,
    save_masks: bool,
    output_dir: str,
    cls_threshold: float,
):
    """Per-GPU worker: runs inference and collects classification + segmentation results."""
    tokenizer = AutoTokenizer.from_pretrained(config["llm_args"]["model_id"])

    transform = None
    if "predict_transform" in config:
        transform = create_transform(config["predict_transform"])

    dataset = VQAMaskDataset(
        df_chunk.question.to_list(),
        df_chunk.img_file_path.to_list(),
        df_chunk.msk_file_path.to_list(),
        df_chunk.organ.to_list(),
        tokenizer,
        labels=df_chunk.label.to_list(),  # Real classification labels
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        collate_fn=dataset.collate_fn,
        num_workers=4,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    model = Uniferum(config).cuda(rank).eval()
    model = load_safetensors(model, ckpt_file)

    if save_masks:
        mask_out_dir = os.path.join(output_dir, "predicted_masks")
        os.makedirs(mask_out_dir, exist_ok=True)

    local_results = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(dataloader, desc=f"GPU {rank}", position=rank)):
            try:
                row = df_chunk.iloc[idx]
                patient_id = row.get("patient_id", f"gpu{rank}_idx{idx}")
                organ = row.get("organ", "unknown")
                gt_cls_label = int(row.get("label", -1))

                # Keep seg_label on CPU for Dice computation
                gt_mask = None
                if "seg_label" in batch:
                    gt_np = batch["seg_label"].numpy()
                    if gt_np.ndim == 5:
                        gt_mask = gt_np[0, 0]
                    elif gt_np.ndim == 4:
                        gt_mask = gt_np[0]
                    else:
                        gt_mask = gt_np

                batch_gpu = {k: v.cuda(rank) for k, v in batch.items() if k != "seg_label"}

                outputs_model = model(**batch_gpu)

                # --- Classification ---
                cls_logit = outputs_model["logits"]          # (1, 1)
                cls_prob = torch.sigmoid(cls_logit).item()
                pred_cls = int(cls_prob >= cls_threshold)

                # --- Segmentation ---
                seg_logits = outputs_model["logits_seg"]     # (1, 1, D, H, W)
                seg_prob = torch.sigmoid(seg_logits).detach().cpu().numpy()
                pred_mask = (seg_prob[0, 0] > 0.5).astype(np.uint8)

                dice = -1.0
                shape_match = True
                if gt_mask is not None:
                    if pred_mask.shape != gt_mask.shape:
                        shape_match = False
                        dice = -2.0
                    else:
                        dice = compute_dice_score(pred_mask, gt_mask)

                mask_path = ""
                if save_masks:
                    mask_path = save_nifti_prediction(pred_mask, patient_id, organ, mask_out_dir)

                local_results.append({
                    "rank": rank,
                    "index": idx,
                    "patient_id": patient_id,
                    "question": row.get("question", ""),
                    "organ": organ,
                    # Classification
                    "label": gt_cls_label,
                    "pred_cls_label": pred_cls,
                    "predict": cls_prob,
                    # Segmentation
                    "dice_score": dice,
                    "shape_match": shape_match,
                    "gt_positive": float(gt_mask.sum()) > 0 if gt_mask is not None else None,
                    "pred_positive": float(pred_mask.sum()) > 0,
                    "mask_path": mask_path,
                })

                if idx % 100 == 0:
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"[GPU {rank}] Error at index {idx}: {e}")

    outputs.append(local_results)
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def cls_metrics(gt, prob, threshold=0.5):
    """Return dict of classification metrics given arrays of GT labels and predicted probs."""
    pred = (np.array(prob) >= threshold).astype(int)
    gt = np.array(gt)
    metrics = {
        "accuracy": accuracy_score(gt, pred),
        "f1": f1_score(gt, pred, zero_division=0),
        "precision": precision_score(gt, pred, zero_division=0),
        "recall": recall_score(gt, pred, zero_division=0),
    }
    if len(np.unique(gt)) == 2:
        metrics["auc"] = roc_auc_score(gt, prob)
    else:
        metrics["auc"] = float("nan")
    return metrics


def print_summary(df_out: pd.DataFrame, cls_threshold: float = 0.5):
    valid_seg = df_out[df_out.dice_score >= 0]
    #valid_cls = df_out[df_out.gt_cls_label >= 0]

    # ── Classification ──────────────────────────────────────────────────────


    # ── Segmentation ────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print(f"{'Segmentation Results':^55}")
    print("=" * 55)
    print(f"{'Organ':<25} {'N':>5} {'Mean Dice':>10} {'Std':>8}")
    print("-" * 55)

    for organ in sorted(valid_seg.organ.unique()):
        sub = valid_seg[valid_seg.organ == organ]
        print(f"{organ:<25} {len(sub):>5} {sub.dice_score.mean():>10.4f} {sub.dice_score.std():>8.4f}")

    print("-" * 55)
    print(
        f"{'Overall':<25} {len(valid_seg):>5} "
        f"{valid_seg.dice_score.mean():>10.4f} {valid_seg.dice_score.std():>8.4f}"
    )
    print("=" * 55)

    # Dice on positive-GT only
    pos_gt = valid_seg[valid_seg.gt_positive == True]
    if len(pos_gt) > 0:
        print(f"\nDice on positive-GT samples only (N={len(pos_gt)}):")
        print(f"  Mean: {pos_gt.dice_score.mean():.4f}  Std: {pos_gt.dice_score.std():.4f}")
    print()


def main():
    args = Arguments().parse_args()
    run_dir = args.run_dir
    config_file = os.path.join(run_dir, "train_config.yaml")
    config = load_yaml(config_file)

    df_all = pd.read_parquet(args.val_file)
    if "split" not in df_all.columns:
        print("WARNING: No 'split' column found. Using all rows as test set.")
        df_test = df_all.reset_index(drop=True)
    else:
        df_test = df_all[df_all.split == "test"].reset_index(drop=True)

    print(f"Test samples   : {len(df_test)}")
    print(f"Organs         : {sorted(df_test.organ.unique())}")
    pos_rate = df_test.label.mean() if "label" in df_test.columns else float("nan")
    print(f"Positive rate  : {pos_rate:.3f}")

    gpus = sorted(set(int(r) for r in args.devices.split(",")))
    num_gpus = len(gpus)
    chunk_size = int(np.ceil(len(df_test) / num_gpus))
    chunks = [df_test.iloc[i * chunk_size: (i + 1) * chunk_size] for i in range(num_gpus)]

    for ckpt_file, ckpt_name in get_ckpt_files(args):
        output_file = os.path.join(run_dir, f"{args.prefix}_{ckpt_name}.parquet")

        if os.path.exists(output_file):
            df_out = pd.read_parquet(output_file)
            required_cols = {"dice_score", "organ", "label", "predict"}
            if not required_cols.issubset(df_out.columns):
                print(
                    f"Stale/incompatible parquet at {output_file} "
                    f"(missing: {required_cols - set(df_out.columns)}). Re-running..."
                )
                os.remove(output_file)
            else:
                print(f"Already exists, loading: {output_file}")
                print_summary(df_out, args.cls_threshold)
                continue

        print(f"\nEvaluating checkpoint: {ckpt_name}")

        manager = mp.Manager()
        results = manager.list()
        processes = []
        for i, rank in enumerate(gpus):
            p = mp.Process(
                target=evaluate_worker,
                args=(
                    rank,
                    config,
                    chunks[i],
                    ckpt_file,
                    results,
                    args.save_masks,
                    run_dir,
                    args.cls_threshold,
                ),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        all_results = [item for sublist in results for item in sublist]
        all_results.sort(key=lambda x: (x["rank"], x["index"]))

        df_out = pd.DataFrame(all_results)
        df_out["ckpt"] = ckpt_name
        df_out["val_file"] = args.val_file
        df_out.to_parquet(output_file)
        print(f"Saved results to {output_file}")

        print_summary(df_out, args.cls_threshold)


if __name__ == "__main__":
    main()
