from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class MeshImageFusionTransformer(nn.Module):
    """Fuse mesh tokens with image tokens using cross-attention.

    Input:
      image_features: (H,W,C)
      vertex_features: (N,C)
    Output:
      fused_vertex_features: (N,C)
    """

    def __init__(self, feature_dim: int = 128, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, image_features: np.ndarray, vertex_features: np.ndarray) -> np.ndarray:
        img = torch.from_numpy(image_features).float().unsqueeze(0)  # (1,H,W,C)
        vtx = torch.from_numpy(vertex_features).float().unsqueeze(0)  # (1,N,C)

        b, h, w, c = img.shape
        memory = img.view(b, h * w, c)
        tgt = vtx

        fused = self.decoder(tgt=tgt, memory=memory)
        fused = self.norm(fused)
        return fused.squeeze(0).detach().cpu().numpy()
