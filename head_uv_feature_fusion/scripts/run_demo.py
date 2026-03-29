from __future__ import annotations

import numpy as np

from head_uv_feature_fusion import CameraParams, MeshData, PipelineInput, UVFeatureUnprojectionPipeline


def make_dummy_data(
    image_h: int = 64,
    image_w: int = 64,
    uv_h: int = 128,
    uv_w: int = 128,
    c: int = 128,
    n_vertices: int = 1200,
) -> PipelineInput:
    rng = np.random.default_rng(42)

    image_features = rng.normal(size=(image_h, image_w, c)).astype(np.float32)
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

    fx = fy = 60.0
    cx = image_w / 2.0
    cy = image_h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.array([0, 0, 0], dtype=np.float32)

    camera = CameraParams(K=K, R=R, t=t, image_size=(image_h, image_w))
    mesh = MeshData(vertices=vertices, faces=faces, uv_coords=uv_coords)

    return PipelineInput(
        image_features=image_features,
        mesh=mesh,
        camera=camera,
        uv_size=(uv_h, uv_w),
    )


def main() -> None:
    data = make_dummy_data()
    pipeline = UVFeatureUnprojectionPipeline(
        feature_dim=data.image_features.shape[-1],
        rasterizer_backend="cpu",
    )
    outputs = pipeline.run(data)

    print("F_dino shape:", data.image_features.shape)
    print("UV_feat shape:", outputs["UV_feat"].shape)
    print("R_feat shape:", outputs["R_feat"].shape)
    print("visible faces:", int(outputs["F_vis"].shape[0]))
    print("UV_visible ratio:", float(outputs["UV_visible"].mean()))


if __name__ == "__main__":
    main()
