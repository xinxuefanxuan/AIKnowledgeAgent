from __future__ import annotations

import numpy as np


def project_points(vertices: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Project 3D points to image plane.

    Args:
        vertices: (N, 3)
        K, R, t: camera params

    Returns:
        points_2d: (N, 2) in pixel coordinates (not clamped).
    """

    cam = (R @ vertices.T).T + t[None, :]
    z = np.clip(cam[:, 2:3], 1e-6, None)
    norm = cam[:, :2] / z
    homog = np.concatenate([norm, np.ones((norm.shape[0], 1), dtype=norm.dtype)], axis=1)
    px = (K @ homog.T).T
    return px[:, :2]


def sample_feature_map_bilinear(feature_map: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """Sample image feature map (H,W,C) at floating-point pixel coords.

    Uses bilinear interpolation.
    """

    h, w, c = feature_map.shape
    x = points_2d[:, 0]
    y = points_2d[:, 1]

    x0 = np.floor(x).astype(np.int32)
    x1 = x0 + 1
    y0 = np.floor(y).astype(np.int32)
    y1 = y0 + 1

    x0c = np.clip(x0, 0, w - 1)
    x1c = np.clip(x1, 0, w - 1)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    fa = feature_map[y0c, x0c]
    fb = feature_map[y1c, x0c]
    fc = feature_map[y0c, x1c]
    fd = feature_map[y1c, x1c]

    sampled = (
        wa[:, None] * fa
        + wb[:, None] * fb
        + wc[:, None] * fc
        + wd[:, None] * fd
    )
    return sampled.reshape(-1, c)


def splat_vertices_to_uv(
    vertex_features: np.ndarray,
    uv_coords: np.ndarray,
    uv_size: tuple[int, int],
) -> np.ndarray:
    """Splat per-vertex features onto UV grid with nearest-neighbor accumulation.

    Args:
        vertex_features: (N,C)
        uv_coords: (N,2) normalized [0,1]
        uv_size: (H_uv, W_uv)

    Returns:
        uv_features: (H_uv, W_uv, C)
    """

    h_uv, w_uv = uv_size
    n, c = vertex_features.shape
    uv = np.clip(uv_coords, 0.0, 1.0)

    xs = np.round(uv[:, 0] * (w_uv - 1)).astype(np.int32)
    ys = np.round(uv[:, 1] * (h_uv - 1)).astype(np.int32)

    out = np.zeros((h_uv, w_uv, c), dtype=np.float32)
    cnt = np.zeros((h_uv, w_uv, 1), dtype=np.float32)

    for i in range(n):
        out[ys[i], xs[i]] += vertex_features[i]
        cnt[ys[i], xs[i], 0] += 1.0

    cnt = np.clip(cnt, 1.0, None)
    out = out / cnt
    return out
