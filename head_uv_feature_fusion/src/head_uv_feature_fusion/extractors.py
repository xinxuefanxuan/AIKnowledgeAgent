from __future__ import annotations

from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F


class DenseFeatureExtractor(Protocol):
    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return dense features of shape (H,W,C)."""


class DummyDinoExtractor:
    """Lightweight fallback extractor for local debugging.

    It is NOT real DINOv2. It just converts RGB to a deterministic dense embedding.
    """

    def __init__(self, out_dim: int = 64):
        self.out_dim = out_dim

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        x = image_rgb.astype(np.float32)
        if x.max() > 1.0:
            x = x / 255.0
        h, w, _ = x.shape
        base = np.concatenate([
            x,
            x.mean(axis=-1, keepdims=True),
            x[..., :1] - x[..., 1:2],
            x[..., 1:2] - x[..., 2:3],
        ], axis=-1)
        rep = int(np.ceil(self.out_dim / base.shape[-1]))
        feat = np.tile(base, (1, 1, rep))[..., : self.out_dim]
        return feat.astype(np.float32)


class DinoV2Extractor:
    """Frozen DINOv2 dense extractor skeleton.

    This class intentionally keeps integration points explicit. Users can wire a real
    DINOv2 checkpoint and patch-token reshaping in `_forward_features`.
    """

    def __init__(self, model_name: str = "dinov2_vits14", device: str = "cpu"):
        self.model_name = model_name
        self.device = torch.device(device)
        self.model = None

    def load_model(self) -> None:
        if self.model is not None:
            return
        try:
            self.model = torch.hub.load("facebookresearch/dinov2", self.model_name)
            self.model.eval().to(self.device)
            for p in self.model.parameters():
                p.requires_grad = False
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DINOv2 from torch.hub. "
                "Use DummyDinoExtractor for offline debug or provide a local checkpoint."
            ) from exc

    def _forward_features(self, image_rgb: np.ndarray) -> np.ndarray:
        self.load_model()
        x = torch.from_numpy(image_rgb).float().to(self.device)
        if x.max() > 1.0:
            x = x / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)

        with torch.no_grad():
            out = self.model.forward_features(x)

        if "x_norm_patchtokens" not in out:
            raise RuntimeError("DINO output missing x_norm_patchtokens; adjust extractor implementation.")

        toks = out["x_norm_patchtokens"]  # (1, N, C)
        n, c = toks.shape[1], toks.shape[2]
        h, w = image_rgb.shape[:2]
        side = int(np.sqrt(n))
        feat = toks[0].reshape(side, side, c).permute(2, 0, 1).unsqueeze(0)
        feat = F.interpolate(feat, size=(h, w), mode="bilinear", align_corners=True)
        return feat.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)

    def __call__(self, image_rgb: np.ndarray) -> np.ndarray:
        return self._forward_features(image_rgb)


def build_feature_extractor(name: str, out_dim: int = 64, device: str = "cpu") -> DenseFeatureExtractor:
    """Factory helper for demos/integration config.

    name:
      - "dummy": DummyDinoExtractor
      - "dinov2_vits14" / "dinov2_vitb14" / etc: DinoV2Extractor via torch.hub
    """

    key = name.lower()
    if key in {"dummy", "dummy_dino", "debug"}:
        return DummyDinoExtractor(out_dim=out_dim)
    if key.startswith("dinov2_"):
        return DinoV2Extractor(model_name=name, device=device)
    raise ValueError(f"Unsupported extractor name: {name}")
