from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class UVFeatureEncoder(nn.Module):
    """Encode UV unprojected features with 3D position priors.

    Input:
      uv_features: (H_uv,W_uv,C)
      uv_positions_3d: (H_uv,W_uv,3)
      uv_visible: (H_uv,W_uv) bool
    Output:
      encoded_uv_features: (H_uv,W_uv,C)
    """

    def __init__(self, feature_dim: int = 128, num_heads: int = 8, num_layers: int = 2):
        super().__init__()
        self.in_proj = nn.Linear(feature_dim + 3, feature_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(
        self,
        uv_features: np.ndarray,
        uv_positions_3d: np.ndarray,
        uv_visible: np.ndarray,
    ) -> np.ndarray:
        h, w, c = uv_features.shape
        uv_feat_t = torch.from_numpy(uv_features).float()
        uv_pos_t = torch.from_numpy(uv_positions_3d).float()
        visible = torch.from_numpy(uv_visible.astype(np.bool_)).view(1, h * w)

        tokens = torch.cat([uv_feat_t, uv_pos_t], dim=-1).view(1, h * w, c + 3)
        tokens = self.in_proj(tokens)

        # True means "masked" in transformer key_padding_mask.
        key_padding_mask = ~visible
        out = self.encoder(tokens, src_key_padding_mask=key_padding_mask)
        out = self.norm(out).view(h, w, c)

        out = out * torch.from_numpy(uv_visible.astype(np.float32))[..., None]
        return out.detach().cpu().numpy()
