from functools import partial
import torch.nn as nn
import torch

from dlvc.models.mit_transformer import MixVisionTransformer
from dlvc.models.segformer_head import SegFormerHead, resize


class SegFormer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Encoder: Hierarchical Transformer
        self.encoder = MixVisionTransformer(
            embed_dims=[32, 64, 160, 256],      # Output dims of each stage
            num_heads=[1, 2, 5, 8],             # Multi-head attention heads
            mlp_ratios=[4, 4, 4, 4],            # MLP expansion ratios
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            depths=[2, 2, 2, 2],                # Layers per stage
            sr_ratios=[8, 4, 2, 1],             # Attention spatial reduction
            drop_rate=0.0,
            drop_path_rate=0.0
        )

        # Decoder: Fuse multi-scale features
        self.decoder = SegFormerHead(
            feature_strides=[4, 8, 16, 32],
            in_channels=[32, 64, 160, 256],
            in_index=[0, 1, 2, 3],
            decoder_params=dict(embed_dim=256),
            num_classes=num_classes
        )

    def forward(self, x):
        enc = self.encoder(x)                         # List of 4 features
        out = self.decoder(enc)                       # Decode to segmentation map
        out = resize(out, size=x.shape[2:], mode='bilinear', align_corners=False)  # Upsample to input size
        return out
