from __future__ import annotations

import numpy as np

from .data_types import PipelineInput
from .geometry import (
    project_points,
    rasterize_uv_to_image_space,
    sample_feature_map_bilinear,
    uv_to_surface_points,
    visible_faces_from_image_space,
)
from .model import UVFeatureEncoder
from .rasterizer import build_rasterizer


class UVFeatureUnprojectionPipeline:
    """Pipeline matching DINO feature unprojection with pluggable rasterizer backend."""

    def __init__(
        self,
        feature_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        rasterizer_backend: str = "cpu",
    ):
        self.encoder = UVFeatureEncoder(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.rasterizer = build_rasterizer(rasterizer_backend)
        self.rasterizer_backend = rasterizer_backend

    def run(self, data: PipelineInput) -> dict[str, np.ndarray]:
        img_f = data.image_features
        mesh = data.mesh
        cam = data.camera

        visible_face_ids, _, _ = visible_faces_from_image_space(
            vertices=mesh.vertices,
            faces=mesh.faces,
            K=cam.K,
            R=cam.R,
            t=cam.t,
            image_size=cam.image_size,
            rasterizer=self.rasterizer,
        )

        uv_visible, uv_face_map, uv_bary_map = rasterize_uv_to_image_space(
            uv_coords=mesh.uv_coords,
            faces=mesh.faces,
            uv_size=data.uv_size,
            visible_face_ids=visible_face_ids,
            rasterizer=self.rasterizer,
        )

        uv_pos_3d = uv_to_surface_points(
            uv_face_map=uv_face_map,
            uv_bary_map=uv_bary_map,
            vertices=mesh.vertices,
            faces=mesh.faces,
        )

        flat_pos = uv_pos_3d.reshape(-1, 3)
        uv_proj_2d, _ = project_points(flat_pos, cam.K, cam.R, cam.t)
        sampled = sample_feature_map_bilinear(img_f, uv_proj_2d).reshape(*data.uv_size, -1)

        uv_feat = sampled * uv_visible[..., None].astype(np.float32)
        if data.uv_mask is not None:
            uv_feat = uv_feat * data.uv_mask[..., None].astype(np.float32)
            uv_visible = uv_visible & data.uv_mask.astype(bool)

        uv_feat_encoded = self.encoder(uv_feat, uv_pos_3d, uv_visible)

        return {
            "uv_feat": uv_feat,
            "uv_feat_encoded": uv_feat_encoded,
            "uv_visible": uv_visible.astype(np.float32),
            "uv_position": uv_pos_3d,
            "uv_proj_2d": uv_proj_2d.reshape(*data.uv_size, 2),
            "visible_face_ids": visible_face_ids,
        }
