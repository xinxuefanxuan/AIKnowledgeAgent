from __future__ import annotations

import numpy as np

from .data_types import PipelineInput
from .extractors import DenseFeatureExtractor
from .geometry import (
    project_points,
    propagate_visibility_to_uv,
    sample_feature_map_bilinear,
    texture_mapped_feature_rendering,
    uv_space_rasterization,
    uv_to_surface_points,
    visible_faces_from_image_space,
)
from .model import UVFeatureEncoder
from .rasterizer import build_rasterizer


class UVFeatureUnprojectionPipeline:
    """Ii -> DINO dense feature -> F_vis -> R_UV -> UV_visible -> UV_feat -> R_feat."""

    def __init__(
        self,
        feature_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 2,
        rasterizer_backend: str = "cpu",
        feature_extractor: DenseFeatureExtractor | None = None,
    ):
        self.encoder = UVFeatureEncoder(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.rasterizer = build_rasterizer(rasterizer_backend)
        self.rasterizer_backend = rasterizer_backend
        self.feature_extractor = feature_extractor

    def _get_dino_feature(self, data: PipelineInput) -> np.ndarray:
        if data.image_features is not None:
            return data.image_features.astype(np.float32)
        if data.image_rgb is None:
            raise ValueError("Need either image_features or image_rgb in PipelineInput.")
        if self.feature_extractor is None:
            raise ValueError("image_rgb is provided but feature_extractor is None.")
        return self.feature_extractor(data.image_rgb)

    def run(self, data: PipelineInput) -> dict[str, np.ndarray]:
        img_f = self._get_dino_feature(data)
        mesh = data.mesh
        cam = data.camera

        # 1) image-space visibility pass on coarse FLAME mesh.
        visible_face_ids, _, _, img_face_map, img_bary_map = visible_faces_from_image_space(
            vertices=mesh.vertices,
            faces=mesh.faces,
            K=cam.K,
            R=cam.R,
            t=cam.t,
            image_size=cam.image_size,
            rasterizer=self.rasterizer,
        )

        # 2) UV-space rasterization to get R_UV (face+bary).
        uv_face_map, uv_bary_map = uv_space_rasterization(
            uv_coords=mesh.uv_coords,
            faces=mesh.faces,
            uv_size=data.uv_size,
            rasterizer=self.rasterizer,
        )

        # 3) propagate visibility: image visibility -> UV_visible mask.
        uv_visible = propagate_visibility_to_uv(uv_face_map, visible_face_ids)

        # 4) recover x(u,v) and project to image p(u,v).
        uv_pos_3d = uv_to_surface_points(
            uv_face_map=uv_face_map,
            uv_bary_map=uv_bary_map,
            vertices=mesh.vertices,
            faces=mesh.faces,
        )
        flat_pos = uv_pos_3d.reshape(-1, 3)
        uv_proj_2d, _ = project_points(flat_pos, cam.K, cam.R, cam.t)

        # 5) bilinear sample from F_dino with normalized grid p_hat in [-1,1].
        sampled, uv_proj_grid = sample_feature_map_bilinear(img_f, uv_proj_2d)
        sampled = sampled.reshape(*data.uv_size, -1)
        uv_proj_grid = uv_proj_grid.reshape(*data.uv_size, 2)

        uv_feat = sampled * uv_visible[..., None].astype(np.float32)
        if data.uv_mask is not None:
            uv_feat = uv_feat * data.uv_mask[..., None].astype(np.float32)
            uv_visible = uv_visible & data.uv_mask.astype(bool)

        uv_feat_encoded = self.encoder(uv_feat, uv_pos_3d, uv_visible)

        # 6) texture-mapped rasterization: UV feature texture -> image-space R_feat.
        r_feat = texture_mapped_feature_rendering(
            uv_feature_map=uv_feat,
            uv_coords=mesh.uv_coords,
            faces=mesh.faces,
            image_face_map=img_face_map,
            image_bary_map=img_bary_map,
        )

        return {
            "F_dino": img_f,
            "F_vis": visible_face_ids,
            "R_UV_face": uv_face_map,
            "R_UV_bary": uv_bary_map,
            "UV_visible": uv_visible.astype(np.float32),
            "UV_position": uv_pos_3d,
            "UV_proj_2d": uv_proj_2d.reshape(*data.uv_size, 2),
            "UV_proj_grid": uv_proj_grid,
            "UV_feat": uv_feat,
            "UV_feat_encoded": uv_feat_encoded,
            "R_feat": r_feat,
        }
