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


def sample_feature_map_bilinear(feature_map: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """Sample feature map (H,W,C) at floating-point pixel coordinates."""

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


def visible_faces_from_image_space(
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: tuple[int, int],
    rasterizer: BaseRasterizer,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Image-space rasterization pass.

    Returns:
      visible_face_ids: current-view visible faces F_vis^i
      projected_vertices: (N,2)
      vertex_depth: (N,)
      image_face_map: (H_img,W_img)
      image_bary_map: (H_img,W_img,3)
    """

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
    """Render UV feature texture back to image: R_feat^i.

    For each visible image pixel, interpolate UV with barycentric weights, then bilinear sample UV feature map.
    """

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

    sampled = sample_feature_map_bilinear(uv_feature_map, uv_pixels)
    rendered[ys, xs] = sampled
    return rendered
