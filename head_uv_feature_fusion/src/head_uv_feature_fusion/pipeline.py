from __future__ import annotations

import numpy as np

from .data_types import PipelineInput
from .geometry import project_points, sample_feature_map_bilinear, splat_vertices_to_uv
from .model import MeshImageFusionTransformer


class UVFeatureFusionPipeline:
    """End-to-end pipeline for one frame."""

    def __init__(self, feature_dim: int = 128, num_heads: int = 8, num_layers: int = 2):
        self.model = MeshImageFusionTransformer(
            feature_dim=feature_dim,
            num_heads=num_heads,
            num_layers=num_layers,
        )

    def run(self, data: PipelineInput) -> np.ndarray:
        """Return UV feature map (H_uv, W_uv, C)."""

        img_f = data.image_features
        vertices = data.mesh.vertices
        uv_coords = data.mesh.uv_coords
        cam = data.camera

        points_2d = project_points(vertices, cam.K, cam.R, cam.t)
        vertex_img_features = sample_feature_map_bilinear(img_f, points_2d)

        fused_vertex_features = self.model(img_f, vertex_img_features)
        uv_feature_map = splat_vertices_to_uv(
            vertex_features=fused_vertex_features,
            uv_coords=uv_coords,
            uv_size=data.uv_size,
        )
        return uv_feature_map
