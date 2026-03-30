from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .rasterizer import BaseRasterizer


def project_points(vertices: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project 3D points to image plane.

    Returns:
      points_2d: (N,2) pixel coordinates.
      depth: (N,) camera-space depth.
    """

    cam = (R @ vertices.T).T + t[None, :]
    z = np.clip(cam[:, 2], 1e-6, None)
    norm = cam[:, :2] / z[:, None]
    homog = np.concatenate([norm, np.ones((norm.shape[0], 1), dtype=norm.dtype)], axis=1)
    px = (K @ homog.T).T[:, :2]
    return px, z


def pixel_to_grid(points_2d: np.ndarray, image_size: tuple[int, int]) -> np.ndarray:
    """Convert pixel coords to [-1,1] grid coords for torch.grid_sample."""

    h, w = image_size
    grid = np.zeros_like(points_2d, dtype=np.float32)
    grid[:, 0] = 2.0 * (points_2d[:, 0] / (w - 1)) - 1.0
    grid[:, 1] = 2.0 * (points_2d[:, 1] / (h - 1)) - 1.0
    return grid


def sample_feature_map_bilinear(feature_map: np.ndarray, points_2d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear sample via grid_sample.

    Returns:
      sampled_feature: (N,C)
      normalized_grid: (N,2) in [-1,1]
    """

    h, w, c = feature_map.shape
    grid = pixel_to_grid(points_2d, (h, w))

    feat = torch.from_numpy(feature_map).float().permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    grd = torch.from_numpy(grid).float().view(1, -1, 1, 2)  # 1,N,1,2

    out = F.grid_sample(feat, grd, mode="bilinear", padding_mode="zeros", align_corners=True)
    out = out.squeeze(0).squeeze(-1).permute(1, 0).cpu().numpy()  # N,C
    return out, grid


def visible_faces_from_image_space(
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: tuple[int, int],
    rasterizer: BaseRasterizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Image-space rasterization pass."""

    projected_vertices, depth = project_points(vertices, K, R, t)
    result = rasterizer.rasterize(
        verts_2d=projected_vertices,
        faces=faces,
        canvas_size=image_size,
        vertex_depth=depth,
    )
    visible_face_ids = np.unique(result.face_map[result.face_map >= 0])
    return visible_face_ids, projected_vertices, depth, result.face_map, result.bary_map


def uv_space_rasterization(
    uv_coords: np.ndarray,
    faces: np.ndarray,
    uv_size: tuple[int, int],
    rasterizer: BaseRasterizer,
) -> tuple[np.ndarray, np.ndarray]:
    """UV-space rasterization pass: R_UV^i (face id + barycentric per texel)."""

    h_uv, w_uv = uv_size
    uv_pixels = np.empty_like(uv_coords, dtype=np.float32)
    uv_pixels[:, 0] = uv_coords[:, 0] * (w_uv - 1)
    uv_pixels[:, 1] = uv_coords[:, 1] * (h_uv - 1)

    result = rasterizer.rasterize(
        verts_2d=uv_pixels,
        faces=faces,
        canvas_size=uv_size,
        vertex_depth=None,
    )
    return result.face_map, result.bary_map


def propagate_visibility_to_uv(uv_face_map: np.ndarray, visible_face_ids: np.ndarray) -> np.ndarray:
    """Propagate image-space visibility to UV domain -> UV_visible^i."""

    if visible_face_ids.size == 0:
        return np.zeros_like(uv_face_map, dtype=bool)
    max_face_from_uv = int(uv_face_map.max()) if np.any(uv_face_map >= 0) else -1
    max_face_from_vis = int(visible_face_ids.max()) if visible_face_ids.size > 0 else -1
    visible_set = np.zeros(max(max_face_from_uv, max_face_from_vis) + 1, dtype=bool)
    visible_set[visible_face_ids] = True

    uv_visible = np.zeros_like(uv_face_map, dtype=bool)
    valid = uv_face_map >= 0
    uv_visible[valid] = visible_set[uv_face_map[valid]]
    return uv_visible


def uv_to_surface_points(
    uv_face_map: np.ndarray,
    uv_bary_map: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Recover 3D point x(u,v) by barycentric interpolation in UV domain."""

    h_uv, w_uv = uv_face_map.shape
    uv_pos_3d = np.zeros((h_uv, w_uv, 3), dtype=np.float32)

    ys, xs = np.where(uv_face_map >= 0)
    for y, x in zip(ys, xs):
        f_idx = uv_face_map[y, x]
        tri_ids = faces[f_idx]
        tri_v = vertices[tri_ids]
        bary = uv_bary_map[y, x]
        uv_pos_3d[y, x] = bary[0] * tri_v[0] + bary[1] * tri_v[1] + bary[2] * tri_v[2]

    return uv_pos_3d


def texture_mapped_feature_rendering(
    uv_feature_map: np.ndarray,
    uv_coords: np.ndarray,
    faces: np.ndarray,
    image_face_map: np.ndarray,
    image_bary_map: np.ndarray,
) -> np.ndarray:
    """Render UV feature texture back to image: R_feat^i."""

    h_img, w_img = image_face_map.shape
    c = uv_feature_map.shape[-1]
    rendered = np.zeros((h_img, w_img, c), dtype=np.float32)

    ys, xs = np.where(image_face_map >= 0)
    if ys.size == 0:
        return rendered

    uv_points = np.zeros((ys.size, 2), dtype=np.float32)
    for k, (y, x) in enumerate(zip(ys, xs)):
        f_idx = image_face_map[y, x]
        tri_vid = faces[f_idx]
        tri_uv = uv_coords[tri_vid]
        bary = image_bary_map[y, x]
        uv = bary[0] * tri_uv[0] + bary[1] * tri_uv[1] + bary[2] * tri_uv[2]
        uv_points[k] = uv

    h_uv, w_uv = uv_feature_map.shape[:2]
    uv_pixels = np.empty_like(uv_points)
    uv_pixels[:, 0] = uv_points[:, 0] * (w_uv - 1)
    uv_pixels[:, 1] = uv_points[:, 1] * (h_uv - 1)

    sampled, _ = sample_feature_map_bilinear(uv_feature_map, uv_pixels)
    rendered[ys, xs] = sampled
    return rendered
