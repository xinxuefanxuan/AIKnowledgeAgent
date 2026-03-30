from dataclasses import dataclass

import numpy as np


@dataclass
class CameraParams:
    """Pinhole camera parameters.

    K: (3,3)
    R: (3,3)
    t: (3,)
    image_size: (H_img, W_img)
    """

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    image_size: tuple[int, int]


@dataclass
class MeshData:
    """Head mesh and UV correspondence.

    vertices: (N,3)
    faces: (M,3) int
    uv_coords: (N,2) normalized in [0,1]
    """

    vertices: np.ndarray
    faces: np.ndarray
    uv_coords: np.ndarray


@dataclass
class PipelineInput:
    """Single-frame inputs for UV feature unprojection.

    Use either:
      - image_features: (H,W,C) precomputed DINO dense features
      - image_rgb: (H,W,3) raw image for extractor to produce DINO dense features
    """

    mesh: MeshData
    camera: CameraParams
    uv_size: tuple[int, int] = (128, 128)
    uv_mask: np.ndarray | None = None
    image_features: np.ndarray | None = None
    image_rgb: np.ndarray | None = None
