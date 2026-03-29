from dataclasses import dataclass

import numpy as np


@dataclass
class CameraParams:
    """Pinhole camera parameters.

    K: (3,3)
    R: (3,3)
    t: (3,)
    """

    K: np.ndarray
    R: np.ndarray
    t: np.ndarray


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
    """Single-frame inputs for UV feature fusion.

    image_features: (H,W,C) from DINO/backbone.
    """

    image_features: np.ndarray
    mesh: MeshData
    camera: CameraParams
    uv_size: tuple[int, int] = (128, 128)
