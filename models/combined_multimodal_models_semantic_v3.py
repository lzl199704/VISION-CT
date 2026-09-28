"""
combined_multimodal_models_semantic_v3.py

Derived from combined_multimodal_models_semantic_v2.py with two changes:

  1. Loss kwargs wiring
     `loss_fct_seg_kwargs` (and `loss_fct_cls_kwargs`) in the YAML are now passed
     through to the loss-function constructor. This finally enables setting
     alpha/gamma/focal_weight/dice_weight/smooth for DiceFocalLoss from config.

  2. Empty-sample-aware seg loss routing
     For DiceFocalLoss specifically:
       - Focal loss is computed on ALL samples (handles FP suppression on
         lesion-free patches via per-voxel penalty).
       - Dice loss is computed ONLY on lesion-bearing samples (avoids the
         structural 1.0 floor that empty samples impose on Dice — empty
         samples produce dice_loss == 1.0 regardless of model output, which
         floods the gradient with constant signal).
     For any other loss type, behavior is identical to v2.

  Two extra fields appear in the return dict for diagnostics:
       - seg_focal_loss:   mean focal over all samples (raw, pre-weight)
       - seg_dice_loss:    mean Dice over lesion-bearing samples only (raw)
       - lesion_sample_frac: fraction of batch that has positive voxels

Δ vs v2: __init__ loss construction, compute_combined_loss seg branch, return dict.

Config example:
    loss_fct_seg: "DiceFocalLoss"
    loss_fct_seg_kwargs:
      alpha: 0.75       # upweight rare positive voxels (RetinaNet default 0.25 is wrong here)
      gamma: 2.0
      focal_weight: 1.0
      dice_weight: 1.0
      smooth: 1.e-6
"""
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, BertConfig, BertModel

from models import vision_models
from models.loss_fcts import DiceFocalLossModule, get_loss_function


def setup_llm(llm_config, lora_config=None):
    llm_module = getattr(transformers, llm_config["model_class"])
    model = llm_module.from_pretrained(
        llm_config["model_id"],
    )

    if llm_config.get("bert_encoder_layers", None):
        n_layers = llm_config["bert_encoder_layers"]
        model.encoder.layer = model.encoder.layer[:n_layers]

    if lora_config is not None:
        lora_config = LoraConfig(**lora_config)
        model = get_peft_model(model, lora_config)
    return model


def setup_vision(vision_config):
    model_module = getattr(vision_models, vision_config["model_class"])
    model = model_module(vision_config)
    return model


class ConcatLayer(nn.Module):
    def forward(self, llm_embeds, vis_embeds, attention_mask, position_ids):
        mm_embeds = torch.cat([vis_embeds, llm_embeds], dim=1)
        batchsize, vseqlen = vis_embeds.size(0), vis_embeds.size(1)
        if attention_mask is not None:
            padding_ = torch.ones(
                (batchsize, vseqlen),
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            attention_mask = torch.cat((padding_, attention_mask), dim=1)

        if position_ids is not None:
            vis_pos_ids = torch.arange(
                0, vseqlen, dtype=position_ids.dtype, device=position_ids.device
            )
            vis_pos_ids = vis_pos_ids.unsqueeze(0).expand(batchsize, -1)
            position_ids = torch.cat((vis_pos_ids, position_ids), dim=1)
        return mm_embeds, attention_mask, position_ids


def _pos_id_from_embeds(embeds):
    pos_ids = torch.arange(0, embeds.size(1), device=embeds.device, dtype=torch.long)
    return pos_ids.unsqueeze(0).expand(embeds.size(0), -1)


class LatentMemLayer(nn.Module):
    def __init__(self, config):
        super(LatentMemLayer, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config["d_model"], nhead=config["nhead"], batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config["num_layers"]
        )
        max_pos_ids = config.get("max_pos_ids", 576)
        self.lm_pos_embeds = nn.Embedding(max_pos_ids, config["d_model"])
        self.vis_pos_embeds = nn.Embedding(max_pos_ids, config["d_model"])
        self.vision_as_mem = config["vision_as_mem"]

    def forward(self, lm_embeds, vis_embeds, attention_mask, position_ids):
        lm_padding_mask = attention_mask == 0

        vis_pos_ids = _pos_id_from_embeds(vis_embeds)
        lm_embeds = lm_embeds + self.lm_pos_embeds(_pos_id_from_embeds(lm_embeds))
        vis_embeds = vis_embeds + self.vis_pos_embeds(vis_pos_ids)

        if self.vision_as_mem:
            mm_embeds = self.decoder(
                lm_embeds, vis_embeds, tgt_key_padding_mask=lm_padding_mask
            )
            return mm_embeds, attention_mask, position_ids
        mm_embeds = self.decoder(
            vis_embeds, lm_embeds, memory_key_padding_mask=lm_padding_mask
        )
        attention_mask = None
        return mm_embeds, attention_mask, vis_pos_ids


def setup_mixer(mixer_config):
    if mixer_config is None or mixer_config["name"] == "concatenate":
        return ConcatLayer()
    if mixer_config["name"] == "latent_mem":
        return LatentMemLayer(mixer_config)


class Unified3DSegmentationDecoder(nn.Module):
    """
    Unified 3D Segmentation Decoder — unchanged from v1.
    """
    def __init__(self, hidden_size, use_skip_connection=True, out_channels=1,
                 base_filters=64, skip_channels=[16, 32, 48, 112]):
        super().__init__()
        self.use_skip_connection = use_skip_connection

        dims = [base_filters * 4, base_filters * 2, base_filters,
                base_filters // 2, base_filters // 4]

        self.up1 = nn.ConvTranspose3d(hidden_size, dims[0], kernel_size=2, stride=2)
        if use_skip_connection:
            self.skip1_proj = nn.Conv3d(skip_channels[3], dims[0], kernel_size=1)
            in_ch1 = dims[0] * 2
        else:
            self.skip1_proj = None
            in_ch1 = dims[0]
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch1, dims[0], kernel_size=3, padding=1),
            nn.InstanceNorm3d(dims[0], affine=True),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose3d(dims[0], dims[1], kernel_size=2, stride=2)
        if use_skip_connection:
            self.skip2_proj = nn.Conv3d(skip_channels[2], dims[1], kernel_size=1)
            in_ch2 = dims[1] * 2
        else:
            self.skip2_proj = None
            in_ch2 = dims[1]
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_ch2, dims[1], kernel_size=3, padding=1),
            nn.InstanceNorm3d(dims[1], affine=True),
            nn.ReLU(inplace=True),
        )

        self.up3 = nn.ConvTranspose3d(dims[1], dims[2], kernel_size=2, stride=2)
        if use_skip_connection:
            self.skip3_proj = nn.Conv3d(skip_channels[1], dims[2], kernel_size=1)
            in_ch3 = dims[2] * 2
        else:
            self.skip3_proj = None
            in_ch3 = dims[2]
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_ch3, dims[2], kernel_size=3, padding=1),
            nn.InstanceNorm3d(dims[2], affine=True),
            nn.ReLU(inplace=True),
        )

        self.up4 = nn.ConvTranspose3d(dims[2], dims[3], kernel_size=2, stride=2)
        if use_skip_connection:
            self.skip4_proj = nn.Conv3d(skip_channels[0], dims[3], kernel_size=1)
            in_ch4 = dims[3] * 2
        else:
            self.skip4_proj = None
            in_ch4 = dims[3]
        self.conv4 = nn.Sequential(
            nn.Conv3d(in_ch4, dims[3], kernel_size=3, padding=1),
            nn.InstanceNorm3d(dims[3], affine=True),
            nn.ReLU(inplace=True),
        )

        self.up5 = nn.Sequential(
            nn.ConvTranspose3d(dims[3], dims[4], kernel_size=2, stride=2),
            nn.InstanceNorm3d(dims[4], affine=True),
            nn.ReLU(inplace=True),
            nn.Conv3d(dims[4], dims[4], kernel_size=3, padding=1),
            nn.InstanceNorm3d(dims[4], affine=True),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv3d(dims[4], out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x, skips=None):
        if self.use_skip_connection:
            if skips is None or len(skips) < 4:
                raise ValueError(
                    f"Decoder configured with use_skip_connection=True "
                    f"but got invalid skips: {len(skips) if skips else 'None'}"
                )

        x = self.up1(x)
        if self.use_skip_connection:
            s = self.skip1_proj(skips[3])
            if x.shape[2:] != s.shape[2:]:
                s = F.interpolate(s, size=x.shape[2:], mode="nearest")
            x = torch.cat([x, s], dim=1)
        x = self.conv1(x)

        x = self.up2(x)
        if self.use_skip_connection:
            s = self.skip2_proj(skips[2])
            if x.shape[2:] != s.shape[2:]:
                s = F.interpolate(s, size=x.shape[2:], mode="nearest")
            x = torch.cat([x, s], dim=1)
        x = self.conv2(x)

        x = self.up3(x)
        if self.use_skip_connection:
            s = self.skip3_proj(skips[1])
            if x.shape[2:] != s.shape[2:]:
                s = F.interpolate(s, size=x.shape[2:], mode="nearest")
            x = torch.cat([x, s], dim=1)
        x = self.conv3(x)

        x = self.up4(x)
        if self.use_skip_connection:
            s = self.skip4_proj(skips[0])
            if x.shape[2:] != s.shape[2:]:
                s = F.interpolate(s, size=x.shape[2:], mode="nearest")
            x = torch.cat([x, s], dim=1)
        x = self.conv4(x)

        x = self.up5(x)
        x = self.final_conv(x)
        return x


class VisionCT(BertModel):
    config_class = BertConfig

    def __init__(self, config):
        llm_config = AutoConfig.from_pretrained(config["llm_args"]["model_id"])
        super().__init__(llm_config)
        self.config = llm_config
        self.setup_config = config
        self.model = setup_llm(config["llm_args"], lora_config=config.get("lora_args"))
        self.vision_model = setup_vision(config["vision_args"])
        self.mixer = setup_mixer(config.get("mixer_args"))
        self.vocab_size = llm_config.vocab_size

        self.use_3d_decoder = config.get("use_3d_decoder", True)
        self.use_skip_connection = config.get("use_skip_connection", False)

        self.use_seg_cls_fusion = config.get("use_seg_cls_fusion", True)
        seg_out_channels = 1
        cls_in_dim = llm_config.hidden_size + (seg_out_channels if self.use_seg_cls_fusion else 0)
        self.dropout = nn.Dropout(p=0.15)
        self.lm_head_cls = nn.Linear(cls_in_dim, 1)

        if self.use_3d_decoder:
            self.seg_decoder = Unified3DSegmentationDecoder(
                llm_config.hidden_size,
                use_skip_connection=self.use_skip_connection,
                out_channels=seg_out_channels,
            )
        else:
            self.lm_head_seg = nn.Linear(llm_config.hidden_size, 64)

        self.loss_fct_cls = get_loss_function(
            config.get("loss_fct_cls", "BCEWithLogitsLoss"),
            reduction="none",
            **config.get("loss_fct_cls_kwargs", {}),
        )
        self.loss_fct_seg = get_loss_function(
            config.get("loss_fct_seg", "CombinedBCEDiceLoss"),
            reduction="none",
            **config.get("loss_fct_seg_kwargs", {}),
        )

        self.cls_loss_weight = config.get("cls_loss_weight", 1.0)
        self.seg_loss_weight = config.get("seg_loss_weight", 10.0)
        self.loss_combination_strategy = config.get("loss_combination_strategy", "weighted_sum")

        self.post_init()

    def get_multimodal_embeds(self, input_ids, position_ids, attention_mask, images):
        vis_outputs = self.vision_model(images)
        skips = None
        if isinstance(vis_outputs, (tuple, list)):
            vis_embeds, skips = vis_outputs
        else:
            vis_embeds = vis_outputs

        lm_embeds = self.model.embeddings(input_ids)

        v_batch = vis_embeds.size(0)
        lm_batch = lm_embeds.size(0)
        if v_batch == 1 and lm_batch > 1:
            vis_embeds = vis_embeds.expand(lm_batch, -1, -1, -1)

        mm_embeds, attention_mask, position_ids = self.mixer(
            lm_embeds, vis_embeds, attention_mask, position_ids
        )

        position_ids = _pos_id_from_embeds(mm_embeds)
        return None, position_ids, attention_mask, mm_embeds, skips

    def get_model(self):
        return self.model

    def _compute_seg_loss(self, seg_logits, seg_labels):
        """
        Per-sample seg loss with empty-aware routing.

        DiceFocalLoss path:
          - focal_per_sample: focal mean over voxels, for every sample        (B,)
          - dice_per_sample:  Dice for lesion-bearing samples; 0 for empties (B,)
          - per_sample = focal_weight * focal_per_sample + dice_weight * dice_per_sample

        Other loss types: same behavior as v2 (single loss call, applied to all).

        Returns:
          per_sample (B,), focal_mean (scalar), dice_mean_pos (scalar over
          positives only — diagnostic), has_mask (B,) or None.
        """
        bs = seg_labels.size(0)
        device = seg_logits.device

        if isinstance(self.loss_fct_seg, DiceFocalLossModule):
            has_mask = seg_labels.reshape(bs, -1).sum(dim=1) > 0  # (B,)

            focal_raw = self.loss_fct_seg.focal_loss(seg_logits, seg_labels)
            if focal_raw.ndim > 1:
                focal_per_sample = focal_raw.reshape(bs, -1).mean(dim=1)
            else:
                focal_per_sample = focal_raw

            dice_per_sample = torch.zeros(bs, device=device, dtype=focal_per_sample.dtype)
            n_pos = int(has_mask.sum().item())
            if n_pos > 0:
                dice_subset = self.loss_fct_seg.dice_loss(
                    seg_logits[has_mask], seg_labels[has_mask]
                )
                dice_per_sample[has_mask] = dice_subset
                dice_mean_pos = dice_subset.mean().detach()
            else:
                dice_mean_pos = torch.zeros((), device=device)

            per_sample = (
                self.loss_fct_seg.focal_weight * focal_per_sample
                + self.loss_fct_seg.dice_weight * dice_per_sample
            )
            return per_sample, focal_per_sample.mean().detach(), dice_mean_pos, has_mask

        seg_losses = self.loss_fct_seg(seg_logits, seg_labels)
        if seg_losses.ndim > 1:
            if seg_logits.ndim == 5:
                seg_losses = seg_losses.mean(dim=list(range(1, seg_losses.ndim)))
            else:
                seg_losses = seg_losses.mean(dim=[1, 2])
        zero = torch.zeros((), device=device)
        return seg_losses, zero, zero, None

    def compute_combined_loss(self,
                              cls_logits: torch.Tensor,
                              seg_logits: torch.Tensor,
                              cls_labels: torch.Tensor,
                              seg_labels: torch.Tensor,
                              sample_weights: Optional[torch.Tensor] = None) -> dict:
        batch_size = cls_logits.shape[0]

        cls_losses = self.loss_fct_cls(cls_logits, cls_labels)
        cls_losses = cls_losses.squeeze(-1)

        if seg_logits.ndim == 5:
            if seg_labels.ndim == 3:
                bs = seg_labels.size(0)
                labels_vol = seg_labels.view(bs, 8, 8, 8, 4, 4, 4)
                labels_vol = labels_vol.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
                labels_vol = labels_vol.view(bs, 1, 32, 32, 32)
                if seg_logits.shape[-1] != 32:
                    labels_vol = F.interpolate(labels_vol.float(), size=seg_logits.shape[2:], mode="nearest")
                seg_labels = labels_vol
            elif seg_labels.ndim == 5:
                if seg_labels.shape[2:] != seg_logits.shape[2:]:
                    seg_labels = F.interpolate(seg_labels.float(), size=seg_logits.shape[2:], mode="nearest")

        seg_losses, seg_focal_term, seg_dice_term, has_mask = self._compute_seg_loss(
            seg_logits, seg_labels
        )

        weighted_cls_losses = self.cls_loss_weight * cls_losses
        weighted_seg_losses = self.seg_loss_weight * seg_losses

        if self.loss_combination_strategy == "weighted_sum":
            combined_losses = weighted_cls_losses + weighted_seg_losses
        elif self.loss_combination_strategy == "adaptive_weight":
            cls_magnitude = cls_losses.detach()
            seg_magnitude = seg_losses.detach()
            total_magnitude = cls_magnitude + seg_magnitude + 1e-8
            adaptive_cls_weight = seg_magnitude / total_magnitude
            adaptive_seg_weight = cls_magnitude / total_magnitude
            combined_losses = (adaptive_cls_weight * weighted_cls_losses +
                               adaptive_seg_weight * weighted_seg_losses)
        else:
            combined_losses = weighted_cls_losses + weighted_seg_losses

        if sample_weights is not None:
            combined_losses = combined_losses * sample_weights

        if torch.isnan(combined_losses).any():
            n_nan = torch.isnan(combined_losses).sum().item()
            print(f"[WARNING] {n_nan}/{combined_losses.numel()} NaN loss values detected and zeroed out.")
            combined_losses = torch.nan_to_num(combined_losses, nan=0.0)

        final_loss = combined_losses.mean()

        if has_mask is not None:
            lesion_frac = has_mask.float().mean().detach()
        else:
            lesion_frac = torch.ones((), device=seg_logits.device)

        return {
            "total_loss": final_loss,
            "cls_loss": cls_losses.mean(),
            "seg_loss": seg_losses.mean(),
            "seg_focal_loss": seg_focal_term,
            "seg_dice_loss": seg_dice_term,
            "lesion_sample_frac": lesion_frac,
            "weighted_cls_loss": weighted_cls_losses.mean(),
            "weighted_seg_loss": weighted_seg_losses.mean(),
            "combined_losses_per_sample": combined_losses,
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        images=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        seg_label=None,
        sample_weights: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[Tuple, dict]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        (
            input_ids,
            position_ids,
            attention_mask,
            inputs_embeds,
            skips,
        ) = self.get_multimodal_embeds(
            input_ids,
            position_ids,
            attention_mask,
            images,
        )

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = transformer_outputs.last_hidden_state

        if self.use_3d_decoder:
            seg_tokens = last_hidden_state[:, 1:513]
            seg_tokens = seg_tokens.transpose(1, 2)
            seg_input = seg_tokens.view(-1, 768, 8, 8, 8)
            logits_seg = self.seg_decoder(seg_input, skips=skips)
        else:
            logits_seg = self.lm_head_seg(last_hidden_state[:, 1:513])

        cls_token = last_hidden_state[:, 0]
        if self.use_seg_cls_fusion and self.use_3d_decoder:
            seg_global = torch.sigmoid(logits_seg).mean(dim=[2, 3, 4])
            cls_input = torch.cat([cls_token, seg_global], dim=-1)
        else:
            cls_input = cls_token

        cls_input = self.dropout(cls_input)
        logits_cls = self.lm_head_cls(cls_input)

        if labels is not None and seg_label is not None:
            loss_info = self.compute_combined_loss(
                cls_logits=logits_cls,
                seg_logits=logits_seg,
                cls_labels=labels,
                seg_labels=seg_label,
                sample_weights=sample_weights,
            )
            loss = loss_info["total_loss"]
        else:
            loss = None
            loss_info = {}

        if not return_dict:
            output = (logits_cls, logits_seg) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        result = {
            "loss": loss,
            "logits": logits_cls,
            "logits_seg": logits_seg,
        }
        if loss_info:
            result.update({
                "cls_loss": loss_info["cls_loss"],
                "seg_loss": loss_info["seg_loss"],
                "seg_focal_loss": loss_info["seg_focal_loss"],
                "seg_dice_loss": loss_info["seg_dice_loss"],
                "lesion_sample_frac": loss_info["lesion_sample_frac"],
                "weighted_cls_loss": loss_info["weighted_cls_loss"],
                "weighted_seg_loss": loss_info["weighted_seg_loss"],
            })

        return result
