from typing import List, Optional, Tuple, Union

import torch
import transformers
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, BertConfig, BertModel

from models import vision_models
from models.loss_fcts import get_loss_function
import time 

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
            # check if to set attention_mask as ones if given None
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
        
        #print('vis_embeds shape', vis_embeds.shape, 'lm_embeds shape', lm_embeds.shape, 'mm embeds shape', mm_embeds)
        if self.vision_as_mem:
            mm_embeds = self.decoder(
                lm_embeds, vis_embeds, tgt_key_padding_mask=lm_padding_mask
            )
            #print('vis_embeds shape', vis_embeds.shape, 'lm_embeds shape', lm_embeds.shape, 'mm embeds shape', mm_embeds.shape)
            return mm_embeds, attention_mask, position_ids
        # language_as_mem
        # add attention as memory_mask
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


class VisionCT(BertModel):
    config_class = BertConfig

    def __init__(self, config):
        # this is annoying, fixme
        llm_config = AutoConfig.from_pretrained(config["llm_args"]["model_id"])
        super().__init__(llm_config)
        self.config = llm_config
        self.setup_config = config
        self.model = setup_llm(config["llm_args"], lora_config=config.get("lora_args"))
        self.vision_model = setup_vision(config["vision_args"])
        self.mixer = setup_mixer(config.get("mixer_args"))
        self.vocab_size = llm_config.vocab_size
        self.lm_head_cls = nn.Linear(llm_config.hidden_size, 1)  # Classification head
        self.lm_head_seg = nn.Linear(llm_config.hidden_size, 64) # Segmentation head (512 patches × 64 voxels) 64 = 4x4x4
        self.loss_fct_cls = get_loss_function(
            config.get("loss_fct_cls", "BCEWithLogitsLoss"), reduction="none"
        )
        self.loss_fct_seg = get_loss_function(
            config.get("loss_fct_seg", "SigmoidFocalLoss"), reduction="none"
        )
        
        # Loss weighting parameters
        self.cls_loss_weight = config.get("cls_loss_weight", 1.0)
        self.seg_loss_weight = config.get("seg_loss_weight", 10.0)
        self.loss_combination_strategy = config.get("loss_combination_strategy", "weighted_sum")
        
        # Initialize weights
        self.post_init()

    def get_multimodal_embeds(self, input_ids, position_ids, attention_mask, images):
        """Create multimodal embeddings by combining vision and text features"""
        vis_embeds = self.vision_model(images)  # [batch, seqlen0, fea_dim]
        lm_embeds = self.model.embeddings(input_ids)  # [batch, seqlen1, fea_lm]
        
        v_batch = vis_embeds.size(0)
        lm_batch = lm_embeds.size(0)
        if v_batch == 1 and lm_batch > 1:
            vis_embeds = vis_embeds.expand(lm_batch, -1, -1, -1)

        # Combine vision and text embeddings
        mm_embeds, attention_mask, position_ids = self.mixer(
            lm_embeds, vis_embeds, attention_mask, position_ids
        )

        # Generate position IDs
        position_ids = _pos_id_from_embeds(mm_embeds)
        return None, position_ids, attention_mask, mm_embeds

    def get_model(self):
        return self.model

    def compute_combined_loss(self, 
                            cls_logits: torch.Tensor,
                            seg_logits: torch.Tensor, 
                            cls_labels: torch.Tensor,
                            seg_labels: torch.Tensor,
                            sample_weights: Optional[torch.Tensor] = None) -> dict:
        """
        Compute combined classification and segmentation loss for each sample.
        
        Args:
            cls_logits: [batch_size, 1] - classification predictions
            seg_logits: [batch_size, 512, 64] - segmentation predictions  
            cls_labels: [batch_size, 1] - classification ground truth
            seg_labels: [batch_size, 512, 64] - segmentation ground truth
            sample_weights: [batch_size] - optional per-sample weights
            
        Returns:
            dict with individual and combined losses
        """
        batch_size = cls_logits.shape[0]
        
        # Compute classification losses (per sample)
        cls_losses = self.loss_fct_cls(cls_logits, cls_labels)  # [batch_size, 1]
        cls_losses = cls_losses.squeeze(-1)  # [batch_size]
        
        # Compute segmentation losses (per sample) 
        seg_losses = self.loss_fct_seg(seg_logits, seg_labels)  # [batch_size, 512, 64]
        seg_losses = seg_losses.mean(dim=[1, 2])  # [batch_size] - average over patches and voxels
        
        # Apply loss weighting
        weighted_cls_losses = self.cls_loss_weight * cls_losses
        weighted_seg_losses = self.seg_loss_weight * seg_losses
        
        # Combine losses based on strategy
        if self.loss_combination_strategy == "weighted_sum":
            # Simple weighted combination
            combined_losses = weighted_cls_losses + weighted_seg_losses
            
        elif self.loss_combination_strategy == "adaptive_weight":
            # Adaptive weighting based on loss magnitudes
            cls_magnitude = cls_losses.detach()
            seg_magnitude = seg_losses.detach()
            
            # Normalize weights to prevent one loss from dominating
            total_magnitude = cls_magnitude + seg_magnitude + 1e-8
            adaptive_cls_weight = seg_magnitude / total_magnitude
            adaptive_seg_weight = cls_magnitude / total_magnitude
            
            combined_losses = (adaptive_cls_weight * weighted_cls_losses + 
                             adaptive_seg_weight * weighted_seg_losses)
            
        elif self.loss_combination_strategy == "uncertainty_weight":
            # Uncertainty-based weighting (requires additional parameters)
            # This would need uncertainty estimation parameters in the model
            # For now, fallback to weighted sum
            combined_losses = weighted_cls_losses + weighted_seg_losses
            
        else:
            # Default: simple weighted sum
            combined_losses = weighted_cls_losses + weighted_seg_losses
        
        # Apply sample weights if provided
        if sample_weights is not None:
            combined_losses = combined_losses * sample_weights
        
        # Compute final loss (mean over batch)
        final_loss = combined_losses.mean()
        
        return {
            'total_loss': final_loss,
            'cls_loss': cls_losses.mean(),
            'seg_loss': seg_losses.mean(), 
            'weighted_cls_loss': weighted_cls_losses.mean(),
            'weighted_seg_loss': weighted_seg_losses.mean(),
            'combined_losses_per_sample': combined_losses  # For analysis/debugging
        }

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,  # Classification labels
        images=None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        seg_label=None,  # Segmentation labels
        sample_weights: Optional[torch.Tensor] = None,  # Per-sample weights
        **kwargs,
    ) -> Union[Tuple, dict]:
        """
        Forward pass with combined classification and segmentation.
        
        Args:
            labels: Classification labels [batch_size, 1]
            seg_label: Segmentation labels [batch_size, 512, 64]
            sample_weights: Optional per-sample loss weights [batch_size]
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        
        # Get multimodal embeddings
        (
            input_ids,
            position_ids,
            attention_mask,
            inputs_embeds,
        ) = self.get_multimodal_embeds(
            input_ids,
            position_ids,
            attention_mask,
            images,
        )
        
        # Forward through transformer
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
        
        # Extract features and make predictions
        last_hidden_state = transformer_outputs.last_hidden_state
        
        # Generate predictions
        logits_cls = self.lm_head_cls(last_hidden_state[:, 0])      # [batch_size, 1]
        logits_seg = self.lm_head_seg(last_hidden_state[:, 1:513]) # [batch_size, 512, 64]
        
        # Compute losses if labels are provided
        if labels is not None and seg_label is not None:
            loss_info = self.compute_combined_loss(
                cls_logits=logits_cls,
                seg_logits=logits_seg,
                cls_labels=labels,
                seg_labels=seg_label,
                sample_weights=sample_weights
            )
            loss = loss_info['total_loss']
        else:
            loss = None
            loss_info = {}

        # Return results
        if not return_dict:
            output = (logits_cls, logits_seg) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        result = {
            "loss": loss,
            "logits": logits_cls,
            "logits_seg": logits_seg,
        }
        
        # Add detailed loss information
        if loss_info:
            result.update({
                "cls_loss": loss_info['cls_loss'],
                "seg_loss": loss_info['seg_loss'],
                "weighted_cls_loss": loss_info['weighted_cls_loss'], 
                "weighted_seg_loss": loss_info['weighted_seg_loss'],
            })
        
        return result

