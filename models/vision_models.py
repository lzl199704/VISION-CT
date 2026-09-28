from typing import Sequence
import timm_3d
import torch
from torch import nn

class EfficientNetViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = timm_3d.create_model(**config["model_args"])
        self.feature_dim = config["feature_dim"]  # checkout efficientnet specs
        self.llm_emb_dim = config["llm_emb_dim"]
        self.seqlen = config["seqlen"]
        self.emb_start = nn.Parameter(torch.randn(1, 1, self.llm_emb_dim))
        self.emb_end = nn.Parameter(torch.randn(1, 1, self.llm_emb_dim))
        self.lm_adaptor = nn.Sequential(
            nn.Linear(self.feature_dim, 1280),
            nn.SiLU(),  # Swish activation, consistent with EfficientNet
            nn.Linear(1280, self.llm_emb_dim),
        )
        #self.lm_reduce = nn.Linear(512, self.seqlen - 2) ###1352 512

    def forward(self, imgs):
        """
        take imgs and output sequence of embeddings for llama
        imgs: (B, in_channels, H, W, D), (H, W, D) = image_sizes
        out: (B, seqlen, lm_dim)
        """
        batchsize = imgs.size(0)
        features = self.model(imgs)
        
        skips = None
        if isinstance(features, (list, tuple)):
            # If features_only=True, we get a list of features
            # The last one is the most abstract (bottleneck)
            out = features[-1] # B, 1280, 8, 8, 8 (assuming 256 input)
            
            # Robust unwrap: sometimes features[-1] itself is a list/tuple
            if isinstance(out, (list, tuple)):
                out = out[-1]
                
            skips = features[:-1] # Keep the rest as skips
        else:
            out = features
            
        out = out.view(batchsize, out.size(1), -1).swapaxes(1, 2)  # B, 8x8x8 , 1280
        out = self.lm_adaptor(out)  # B, 512, 768
        
        #out = self.lm_reduce(out.swapaxes(1, 2)).swapaxes(1, 2) # B, 256, 768
        
        emb_s = self.emb_start.expand(batchsize, -1, -1)
        emb_e = self.emb_end.expand(batchsize, -1, -1)
        out = torch.cat([emb_s, out, emb_e], 1)
        
        if skips is not None:
             return out, skips
        return out
