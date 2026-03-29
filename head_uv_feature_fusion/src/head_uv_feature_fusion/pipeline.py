from __future__ import annotations

import numpy as np

from .data_types import PipelineInput
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
    """Pipeline aligned with user-described flow:

    Ii -> DINO -> image-space rasterization -> F_vis
      -> UV-space rasterization (R_UV)
      -> visibility propagation (UV_visible)
      -> x(u,v) -> p(u,v) -> bilinear sample -> UV_feat
      -> texture-mapped rasterization -> R_feat
    """

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

        # 1) Image-space rasterization and visibility.
        visible_face_ids, _, _, img_face_map, img_bary_map = visible_faces_from_image_space(
            vertices=mesh.vertices,
            faces=mesh.faces,
            K=cam.K,
            R=cam.R,
            t=cam.t,
            image_size=cam.image_size,
            rasterizer=self.rasterizer,
        )

        # 2) UV-space rasterization (R_UV: face id + bary).
        uv_face_map, uv_bary_map = uv_space_rasterization(
            uv_coords=mesh.uv_coords,
            faces=mesh.faces,
            uv_size=data.uv_size,
            rasterizer=self.rasterizer,
        )

        # 3) Propagate image-space visibility into UV domain.
        uv_visible = propagate_visibility_to_uv(uv_face_map, visible_face_ids)

        # 4) Recover x(u,v) and project to image p(u,v).
        uv_pos_3d = uv_to_surface_points(
            uv_face_map=uv_face_map,
            uv_bary_map=uv_bary_map,
            vertices=mesh.vertices,
            faces=mesh.faces,
        )
        flat_pos = uv_pos_3d.reshape(-1, 3)
        uv_proj_2d, _ = project_points(flat_pos, cam.K, cam.R, cam.t)

        # 5) Bilinear sample from F_dino -> UV_feat.
        sampled = sample_feature_map_bilinear(img_f, uv_proj_2d).reshape(*data.uv_size, -1)
        uv_feat = sampled * uv_visible[..., None].astype(np.float32)

        if data.uv_mask is not None:
            uv_feat = uv_feat * data.uv_mask[..., None].astype(np.float32)
            uv_visible = uv_visible & data.uv_mask.astype(bool)

        uv_feat_encoded = self.encoder(uv_feat, uv_pos_3d, uv_visible)

        # 6) Texture-mapped rasterization from UV feature to image feature rendering R_feat.
        r_feat = texture_mapped_feature_rendering(
            uv_feature_map=uv_feat,
            uv_coords=mesh.uv_coords,
            faces=mesh.faces,
            image_face_map=img_face_map,
            image_bary_map=img_bary_map,
        )

        return {
            "F_vis": visible_face_ids,
            "R_UV_face": uv_face_map,
            "R_UV_bary": uv_bary_map,
            "UV_visible": uv_visible.astype(np.float32),
            "UV_position": uv_pos_3d,
            "UV_proj_2d": uv_proj_2d.reshape(*data.uv_size, 2),
            "UV_feat": uv_feat,
            "UV_feat_encoded": uv_feat_encoded,
            "R_feat": r_feat,
        }
