from __future__ import annotations

from pathlib import Path

import numpy as np

from head_uv_feature_fusion import (
    CameraParams,
    MeshData,
    PipelineInput,
    UVFeatureUnprojectionPipeline,
    save_feature_rgb,
    save_gray,
)


def make_dummy_data(
    image_h: int = 32,
    image_w: int = 32,
    uv_h: int = 32,
    uv_w: int = 32,
    c: int = 16,
    n_vertices: int = 100,
) -> PipelineInput:
    rng = np.random.default_rng(123)

    image_features = rng.normal(size=(image_h, image_w, c)).astype(np.float32)
    vertices = rng.normal(size=(n_vertices, 3)).astype(np.float32)
    vertices[:, 2] += 4.0

    faces = np.stack(
        [
            np.arange(0, n_vertices - 2),
            np.arange(1, n_vertices - 1),
            np.arange(2, n_vertices),
        ],
        axis=1,
    ).astype(np.int32)
    uv_coords = rng.uniform(0, 1, size=(n_vertices, 2)).astype(np.float32)

    fx = fy = 80.0
    cx = image_w / 2.0
    cy = image_h / 2.0
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    R = np.eye(3, dtype=np.float32)
    t = np.array([0, 0, 0], dtype=np.float32)

    return PipelineInput(
        image_features=image_features,
        mesh=MeshData(vertices=vertices, faces=faces, uv_coords=uv_coords),
        camera=CameraParams(K=K, R=R, t=t, image_size=(image_h, image_w)),
        uv_size=(uv_h, uv_w),
    )


def main() -> None:
    out_dir = Path("artifacts/debug_dump")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = make_dummy_data()
    pipe = UVFeatureUnprojectionPipeline(feature_dim=data.image_features.shape[-1], rasterizer_backend="cpu")
    out = pipe.run(data)

    save_feature_rgb(str(out_dir / "F_dino_rgb.ppm"), data.image_features, channels=(0, 1, 2))
    save_gray(str(out_dir / "UV_visible.ppm"), out["UV_visible"])
    save_feature_rgb(str(out_dir / "UV_feat_rgb.ppm"), out["UV_feat"], channels=(0, 1, 2))
    save_feature_rgb(str(out_dir / "R_feat_rgb.ppm"), out["R_feat"], channels=(0, 1, 2))

    np.save(out_dir / "R_UV_face.npy", out["R_UV_face"])
    np.save(out_dir / "R_UV_bary.npy", out["R_UV_bary"])
    np.save(out_dir / "UV_position.npy", out["UV_position"])
    np.save(out_dir / "UV_proj_2d.npy", out["UV_proj_2d"])

    print(f"saved debug artifacts to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
