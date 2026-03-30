from __future__ import annotations

import numpy as np

from head_uv_feature_fusion import (
    CameraParams,
    build_feature_extractor,
    MeshData,
    PipelineInput,
    UVFeatureUnprojectionPipeline,
)


def make_dummy_data(
    image_h: int = 48,
    image_w: int = 48,
    uv_h: int = 48,
    uv_w: int = 48,
    n_vertices: int = 160,
) -> PipelineInput:
    rng = np.random.default_rng(42)

    image_rgb = rng.uniform(0, 255, size=(image_h, image_w, 3)).astype(np.float32)
    vertices = rng.normal(size=(n_vertices, 3)).astype(np.float32)
    vertices[:, 2] += 3.0

    faces = np.stack(
        [
            np.arange(0, n_vertices - 2),
            np.arange(1, n_vertices - 1),
            np.arange(2, n_vertices),
        ],
        axis=1,
    ).astype(np.int32)
    uv_coords = rng.uniform(0, 1, size=(n_vertices, 2)).astype(np.float32)

    fx = fy = 45.0
    cx = image_w / 2.0
    cy = image_h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.array([0, 0, 0], dtype=np.float32)

    camera = CameraParams(K=K, R=R, t=t, image_size=(image_h, image_w))
    mesh = MeshData(vertices=vertices, faces=faces, uv_coords=uv_coords)

    return PipelineInput(
        image_rgb=image_rgb,
        mesh=mesh,
        camera=camera,
        uv_size=(uv_h, uv_w),
    )


def main() -> None:
    data = make_dummy_data()
    extractor = build_feature_extractor("dummy", out_dim=64)
    pipeline = UVFeatureUnprojectionPipeline(
        feature_dim=64,
        rasterizer_backend="cpu",
        feature_extractor=extractor,
    )
    outputs = pipeline.run(data)

    print("F_dino shape:", outputs["F_dino"].shape)
    print("UV_feat shape:", outputs["UV_feat"].shape)
    print("R_feat shape:", outputs["R_feat"].shape)
    print("visible faces:", int(outputs["F_vis"].shape[0]))
    print("UV_visible ratio:", float(outputs["UV_visible"].mean()))


if __name__ == "__main__":
    main()
