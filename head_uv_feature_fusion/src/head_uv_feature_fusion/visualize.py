from __future__ import annotations

from pathlib import Path

import numpy as np


def _to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-8:
        return np.zeros_like(x, dtype=np.uint8)
    x = (x - x_min) / (x_max - x_min)
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def _save_ppm_rgb(path: Path, rgb: np.ndarray) -> None:
    """Save RGB image as binary PPM (P6) without external dependencies."""

    h, w, c = rgb.shape
    if c != 3:
        raise ValueError(f"Expected RGB image with 3 channels, got {c}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(f"P6\n{w} {h}\n255\n".encode("ascii"))
        f.write(rgb.tobytes())


def save_gray(path: str, x: np.ndarray) -> None:
    gray = _to_uint8(x)
    rgb = np.stack([gray, gray, gray], axis=-1)
    _save_ppm_rgb(Path(path), rgb)


def save_feature_rgb(path: str, feat: np.ndarray, channels: tuple[int, int, int] = (0, 1, 2)) -> None:
    c0, c1, c2 = channels
    rgb = np.stack([feat[..., c0], feat[..., c1], feat[..., c2]], axis=-1)
    rgb = _to_uint8(rgb)
    _save_ppm_rgb(Path(path), rgb)
