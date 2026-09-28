import os
import sys
import types
import pandas as pd
import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
from peft import get_peft_model, LoraConfig

# Add repo root to path
sys.path.append(os.getcwd())

from bin.utils import (
    ConfigFileArgs,
    create_transform,
    load_checkpoint_from_config_if_available,
    load_yaml,
    save_yaml,
)
from data_utils.combined_vqa_dataset_abd import VQABinaryDataCollator, VQAMaskDataset
from models.combined_multimodal_models_abd import VisionCT

def load_vision_decoder_weights(model, ckpt_path):
    print(f"Loading vision encoder and decoder weights from {ckpt_path}...")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"pretrain_checkpoint does not exist: {ckpt_path}\n"
            "Stage 2 must be initialised from the Stage-1 segmentation weights. Run Step 1 first, or "
            "point `pretrain_checkpoint:` at an existing checkpoint. (Set it to null to train Stage 2 "
            "from scratch on purpose.)")
    state_dict = None
    try:
        # Handle DeepSpeed checkpoints (directory)
        if os.path.isdir(ckpt_path):
            if os.path.exists(os.path.join(ckpt_path, "model.safetensors")):
                ckpt_path = os.path.join(ckpt_path, "model.safetensors")
            elif os.path.exists(os.path.join(ckpt_path, "pytorch_model.bin")):
                ckpt_path = os.path.join(ckpt_path, "pytorch_model.bin")
            else:
                # Fallback: try to find the newest .safetensors, .bin, or .pt in dir
                files = [os.path.join(ckpt_path, f) for f in os.listdir(ckpt_path)
                         if f.endswith('.safetensors') or f.endswith('.bin') or f.endswith('.pt')]
                if files:
                    ckpt_path = max(files, key=os.path.getmtime)

        if ckpt_path.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(ckpt_path, device="cpu")
        else:
            state_dict = torch.load(ckpt_path, map_location="cpu")
    except Exception as e:
        raise RuntimeError(f"Failed to read pretrain_checkpoint {ckpt_path}: {e}") from e

    # Unpack if needed
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    elif "module" in state_dict:
        state_dict = state_dict["module"]

    new_sd = {}
    loaded_keys = []
    model_sd = model.state_dict()
    
    # Target prefixes to load
    prefixes = ["vision_model.", "seg_decoder."]
    
    for k, v in state_dict.items():
        # Remove DDP 'module.' prefix if present
        k_clean = k.replace("module.", "")
        
        # Check if key belongs to vision or decoder
        if any(k_clean.startswith(p) for p in prefixes):
             if k_clean in model_sd:
                 # Check shape compat
                 if v.shape == model_sd[k_clean].shape:
                     new_sd[k_clean] = v
                     loaded_keys.append(k_clean)
                 else:
                     print(f"Skipping {k_clean}: shape mismatch {v.shape} vs {model_sd[k_clean].shape}")
             else:
                 pass # Key not in current model
    
    if not loaded_keys:
        raise RuntimeError(
            f"pretrain_checkpoint {ckpt_path} contained no usable vision_model./seg_decoder. weights "
            f"(checkpoint has {len(state_dict)} keys). Stage 2 would silently train the vision tower "
            "from scratch -- refusing to continue.")
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"Successfully loaded {len(loaded_keys)} keys for Vision/Decoder from {ckpt_path}.")


def create_datasets(
    input_file: str,
    tokenizer,
    train_transform=None,
    predict_transform=None,
    train_transform_imgonly=None,
    num_samples=None,
    stage="standard"
):
    df_data = pd.read_parquet(input_file)
    df_train = df_data[df_data.split == "train"]
    df_val = df_data[df_data.split == "val"]
    
    # Subset for pretraining if requested
    if num_samples is not None and stage == "pretrain":
        print(f"Subsetting training data to {num_samples} samples for pre-training...")
        # Stratified sampling by organ to ensure balance
        try:
             # Sample n per group, then concat
             groups = df_train.groupby("organ")
             n_per_group = max(1, num_samples // len(groups))
             df_train = groups.apply(lambda x: x.sample(min(len(x), n_per_group))).reset_index(drop=True)
             
             # If exact number not matched (due to small groups), fill up randomly
             if len(df_train) < num_samples:
                 remaining = num_samples - len(df_train)
                 others = df_data[(df_data.split == "train") & (~df_data.index.isin(df_train.index))]
                 if len(others) > 0:
                     fill = others.sample(min(len(others), remaining))
                     df_train = pd.concat([df_train, fill])
        except Exception as e:
             print(f"Stratified sampling failed: {e}. Falling back to random sampling.")
             df_train = df_train.sample(n=min(len(df_train), num_samples))
        
        print(f"New training size: {len(df_train)}")

    train_dataset = VQAMaskDataset(
        df_train.question.to_list(),
        df_train.img_file_path.to_list(),
        df_train.msk_file_path.to_list(),
        df_train.organ.to_list(),
        tokenizer,
        df_train.label.to_list(),
        transform=train_transform,
        transform_imgonly=train_transform_imgonly,
    )

    eval_dataset = VQAMaskDataset(
        df_val.question.to_list(),
        df_val.img_file_path.to_list(),
        df_val.msk_file_path.to_list(),
        df_val.organ.to_list(),
        tokenizer,
        df_val.label.to_list(),
        transform=predict_transform,
    )

    collector = VQABinaryDataCollator(train_dataset.tokenizer, True)
    return train_dataset, eval_dataset, collector


GRAD_STATE = {"nonfinite": 0, "skipped": 0, "total": 0}


class SkipNonFiniteTrainer(Trainer):
    """Skip the optimizer step whenever ANY rank saw a non-finite gradient in that step.

    Zeroing non-finite gradients (what register_nan_grad_hooks does) is not sufficient. With
    AdamW the step still applies decoupled weight decay and still advances the moment
    estimates, so a long run of "zeroed" steps quietly shrinks the weights instead of leaving
    the model untouched. In the 0903 run 78% of steps had a non-finite gradient norm and the
    model degraded from eval 0.638 to 0.941 while training loss also rose -- the signature of
    damage, not overfitting. Skipping the step outright leaves the weights exactly as they were.

    The decision is all-reduced with MAX so every rank skips together; otherwise the DDP
    replicas would silently drift apart.
    """

    def create_optimizer(self):
        optimizer = super().create_optimizer()
        orig_step = optimizer.step          # bound method of the real optimizer
        device = self.args.device

        # Must stay a BOUND method: the LR scheduler wraps optimizer.step with
        # torch.optim.lr_scheduler.with_counter, which reads method.__self__/__func__.
        def _step(self, *args, **kwargs):
            flag = torch.tensor(1.0 if GRAD_STATE["nonfinite"] else 0.0, device=device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
            GRAD_STATE["nonfinite"] = 0
            GRAD_STATE["total"] += 1
            if flag.item() > 0:
                GRAD_STATE["skipped"] += 1
                return None
            return orig_step(*args, **kwargs)

        optimizer.step = types.MethodType(_step, optimizer)
        return optimizer


class GradSkipCallback(TrainerCallback):
    """Report how many optimizer steps were skipped for non-finite gradients."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        tot, sk = GRAD_STATE["total"], GRAD_STATE["skipped"]
        if tot and state.is_world_process_zero:
            print(f"[GRAD] step {state.global_step}: skipped {sk}/{tot} optimizer steps "
                  f"({sk / tot:.2%}) for non-finite gradients", flush=True)


def register_nan_grad_hooks(model):
    """Zero out NaN/Inf gradients before the optimizer step.

    In BF16 training, the backward pass through multiple gradient paths
    (seg + cls + ROI) can overflow BF16 range (max ~65504). When two paths
    produce +Inf and -Inf, their sum is NaN, which corrupts model weights.
    These hooks catch that AFTER backward() but BEFORE optimizer.step(),
    preventing weight corruption regardless of where the overflow originates.
    """
    _warned = set()

    def make_hook(name):
        def hook(grad):
            if grad is None:
                return grad
            bad = ~torch.isfinite(grad)
            if bad.any():
                GRAD_STATE["nonfinite"] += 1
                n = bad.sum().item()
                if name not in _warned:
                    print(f"[GRAD WARN] {name}: {n}/{grad.numel()} NaN/Inf grads zeroed")
                    _warned.add(name)
                return grad.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
            return grad
        return hook

    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.register_hook(make_hook(name))
            count += 1
    print(f"Registered NaN/Inf gradient hooks on {count} trainable parameters.")


def create_and_prepare_model(config):
    tokenizer = AutoTokenizer.from_pretrained(config["llm_args"]["model_id"])
    model = VisionCT(config)
    
    # Load checkpoint if resuming standard training (optimizer states etc)
    # But if finetuning from pretrain, we handle it separately
    model = load_checkpoint_from_config_if_available(model, config)
    
    training_stage = config.get("training_stage", "finetune")
    print(f"Initializing model for stage: {training_stage}")

    if training_stage == "pretrain":
        # 1. Force weights for Segmentation Only
        model.cls_loss_weight = 0.0
        model.seg_loss_weight = 1.0
        
        # 2. Freeze LLM entirely
        print("Pre-training: Freezing LLM layers and heads...")
        for param in model.model.parameters():
            param.requires_grad = False
        for param in model.lm_head_cls.parameters():
            param.requires_grad = False
            
    elif training_stage == "finetune":
        # 1. Load Pretrained Weights (Vision + Decoder)
        # Only if not resuming from a full checkpoint
        if config.get("pretrain_checkpoint") and not config.get("resume_from_checkpoint"):
            load_vision_decoder_weights(model, config["pretrain_checkpoint"])
        elif not config.get("resume_from_checkpoint"):
            print("WARNING: training_stage=finetune with no `pretrain_checkpoint` -- the vision encoder "
                  "and segmentation decoder start from ImageNet/random init, not from Stage 1.")
            
        # 2. Set Weights normal (Already set by init from config)
        
        # 3. Freeze LLM logic (standard)
        freeze_llm = config.get("freeze_llm", True)
        if freeze_llm:
            print("Fine-tuning: Freezing LLM backbone...")
            for param in model.model.parameters():
                 param.requires_grad = False
        
        # Ensure Classification head is trainable
        for param in model.lm_head_cls.parameters():
            param.requires_grad = True
        # Ensure skip ROI projection (v5+) is trainable
        if hasattr(model, 'skip_roi_proj'):
            for param in model.skip_roi_proj.parameters():
                param.requires_grad = True

    # Always ensure Vision + Decoder + Mixer are trainable
    print("Setting gradients for Vision and Decoder...")
    for param in model.vision_model.parameters():
        param.requires_grad = True
        
    if hasattr(model, 'seg_decoder'):
        for param in model.seg_decoder.parameters():
            param.requires_grad = True
            
    if hasattr(model, 'mixer'):
        for param in model.mixer.parameters():
            param.requires_grad = True
            
    # Allow vision model pooler tuning if present
    if hasattr(model.model, 'pooler') and model.model.pooler is not None:
         for param in model.model.pooler.parameters():
            param.requires_grad = True

    # Report trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params} / {all_params} ({trainable_params/all_params:.2%})")

    # Register gradient hooks AFTER all requires_grad flags are set so hooks
    # are only placed on the final set of trainable parameters.
    register_nan_grad_hooks(model)

    return model, tokenizer


def main():
    args = ConfigFileArgs().parse_args()
    config_file = args.config_file
    config = load_yaml(config_file)
    
    if "logging_dir" not in config["hf_TrainingArguments"]:
        out_dir = config["hf_TrainingArguments"].get("output_dir", "./")
        config["hf_TrainingArguments"]["logging_dir"] = os.path.join(out_dir, "logs")
    
    # Ensure output directory exists
    os.makedirs(config["hf_TrainingArguments"]["output_dir"], exist_ok=True)

    file_path = os.path.join(
        config["hf_TrainingArguments"]["output_dir"], "train_config.yaml"
    )
    save_yaml(config, file_path)

    # Setup Transforms
    train_transform = None
    if config.get("train_transform") is not None:
        train_transform = create_transform(config["train_transform"])
    
    train_transform_imgonly = None
    if config.get("train_transform_imgonly") is not None:
        train_transform_imgonly = create_transform(config["train_transform_imgonly"])
        
    predict_transform = None
    if config.get("predict_transform") is not None:
        predict_transform = create_transform(config["predict_transform"])

    model, tokenizer = create_and_prepare_model(config)
    
    # Fix for deepspeed/distributed training serialization
    if hasattr(model, 'state_dict'):
        orig_sd = model.state_dict
        def contiguous_state_dict(*args, **kwargs):
            sd = orig_sd(*args, **kwargs)
            for k, v in sd.items():
                if isinstance(v, torch.Tensor) and not v.is_contiguous():
                    sd[k] = v.contiguous()
            return sd
        model.state_dict = contiguous_state_dict
 
    # Enable CuDNN benchmark for 3D convolutions on H100
    torch.backends.cudnn.benchmark = True
    
    print("Creating Datasets...")
    training_stage = config.get("training_stage", "finetune")
    num_samples = config.get("num_pretrain_samples", None)
    
    # Only use num_samples if we are in pretrain stage
    if training_stage != "pretrain":
        num_samples = None
        
    train_dataset, eval_dataset, collector = create_datasets(
        config["input_file"],
        tokenizer,
        train_transform=train_transform,
        predict_transform=predict_transform,
        train_transform_imgonly=train_transform_imgonly,
        num_samples=num_samples,
        stage=training_stage
    )

    # Set training arguments
    training_args = TrainingArguments(**config["hf_TrainingArguments"])
    
    skip_nonfinite = config.get("skip_nonfinite_grad_steps", True)
    TrainerCls = SkipNonFiniteTrainer if skip_nonfinite else Trainer
    print(f"Optimizer-step skipping on non-finite gradients: {'ON' if skip_nonfinite else 'OFF'}")
    trainer = TrainerCls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collector,
        callbacks=[GradSkipCallback()] if skip_nonfinite else None,
    )

    print(f"Starting Training (Stage: {training_stage})...")
    trainer.train()#resume_from_checkpoint=True) 

if __name__ == "__main__":
    main()
