from __future__ import annotations

import numpy as np

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


def visible_faces_from_image_space(
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: tuple[int, int],
    rasterizer: BaseRasterizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Image-space visibility pass with per-pixel z-buffer."""

    projected_vertices, depth = project_points(vertices, K, R, t)
    result = rasterizer.rasterize(
        verts_2d=projected_vertices,
        faces=faces,
        canvas_size=image_size,
        vertex_depth=depth,
    )
    visible_face_ids = np.unique(result.face_map[result.face_map >= 0])
    return visible_face_ids, projected_vertices, depth


def rasterize_uv_to_image_space(
    uv_coords: np.ndarray,
    faces: np.ndarray,
    uv_size: tuple[int, int],
    visible_face_ids: np.ndarray,
    rasterizer: BaseRasterizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """UV-space pass producing visible face map and barycentric map."""

    h_uv, w_uv = uv_size
    uv_pixels = np.empty_like(uv_coords, dtype=np.float32)
    uv_pixels[:, 0] = uv_coords[:, 0] * (w_uv - 1)
    uv_pixels[:, 1] = uv_coords[:, 1] * (h_uv - 1)

    mask = np.zeros((faces.shape[0],), dtype=bool)
    mask[visible_face_ids] = True
    vis_faces = faces[mask]
    original_ids = np.nonzero(mask)[0]

    if vis_faces.size == 0:
        empty_face = np.full((h_uv, w_uv), -1, dtype=np.int32)
        return np.zeros((h_uv, w_uv), dtype=bool), empty_face, np.zeros((h_uv, w_uv, 3), dtype=np.float32)

    result = rasterizer.rasterize(
        verts_2d=uv_pixels,
        faces=vis_faces,
        canvas_size=uv_size,
        vertex_depth=None,
    )
    uv_face_map = np.full_like(result.face_map, -1)
    valid = result.face_map >= 0
    uv_face_map[valid] = original_ids[result.face_map[valid]]
    uv_visible = uv_face_map >= 0
    return uv_visible, uv_face_map, result.bary_map


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


def sample_feature_map_bilinear(feature_map: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """Sample image feature map (H,W,C) at floating-point pixel coords."""

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

    return wa[:, None] * fa + wb[:, None] * fb + wc[:, None] * fc + wd[:, None] * fd
