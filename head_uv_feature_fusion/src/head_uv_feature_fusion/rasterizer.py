from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RasterizeResult:
    face_map: np.ndarray
    bary_map: np.ndarray
    depth_map: np.ndarray


class BaseRasterizer:
    """Abstract rasterizer interface.

    Backends:
      - cpu: numpy implementation (default)
      - nvdiffrast/pytorch3d: extend this class for differentiable rendering
    """

    def rasterize(
        self,
        verts_2d: np.ndarray,
        faces: np.ndarray,
        canvas_size: tuple[int, int],
        vertex_depth: np.ndarray | None = None,
    ) -> RasterizeResult:
        raise NotImplementedError


class CPURasterizer(BaseRasterizer):
    """CPU rasterizer with per-pixel barycentric depth z-buffer."""

    @staticmethod
    def _barycentric_weights(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        v0 = b - a
        v1 = c - a
        v2 = p - a

        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01

        if abs(denom) < 1e-12:
            return np.array([-1.0, -1.0, -1.0], dtype=np.float32)

        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return np.array([u, v, w], dtype=np.float32)

    def rasterize(
        self,
        verts_2d: np.ndarray,
        faces: np.ndarray,
        canvas_size: tuple[int, int],
        vertex_depth: np.ndarray | None = None,
    ) -> RasterizeResult:
        h, w = canvas_size
        face_map = np.full((h, w), -1, dtype=np.int32)
        bary_map = np.zeros((h, w, 3), dtype=np.float32)
        depth_map = np.full((h, w), np.inf, dtype=np.float32)

        for f_idx, (i0, i1, i2) in enumerate(faces):
            tri = verts_2d[[i0, i1, i2]]
            min_x = max(int(np.floor(tri[:, 0].min())), 0)
            max_x = min(int(np.ceil(tri[:, 0].max())), w - 1)
            min_y = max(int(np.floor(tri[:, 1].min())), 0)
            max_y = min(int(np.ceil(tri[:, 1].max())), h - 1)

            if min_x > max_x or min_y > max_y:
                continue

            z_tri = None if vertex_depth is None else vertex_depth[[i0, i1, i2]]

            for yy in range(min_y, max_y + 1):
                for xx in range(min_x, max_x + 1):
                    p = np.array([xx + 0.5, yy + 0.5], dtype=np.float32)
                    bary = self._barycentric_weights(p, tri[0], tri[1], tri[2])
                    if np.min(bary) < -1e-4:
                        continue

                    z_val = 0.0 if z_tri is None else float(np.dot(bary, z_tri))
                    if z_val < depth_map[yy, xx]:
                        depth_map[yy, xx] = z_val
                        face_map[yy, xx] = f_idx
                        bary_map[yy, xx] = bary

        return RasterizeResult(face_map=face_map, bary_map=bary_map, depth_map=depth_map)


class DifferentiableRasterizer(BaseRasterizer):
    """Adapter placeholder for nvdiffrast / PyTorch3D style rasterization."""

    def __init__(self, backend: str = "nvdiffrast"):
        self.backend = backend

    def rasterize(
        self,
        verts_2d: np.ndarray,
        faces: np.ndarray,
        canvas_size: tuple[int, int],
        vertex_depth: np.ndarray | None = None,
    ) -> RasterizeResult:
        raise NotImplementedError(
            f"Differentiable backend '{self.backend}' is not wired yet. "
            "Plug your renderer here (nvdiffrast/PyTorch3D) and return face_map/bary_map/depth_map."
        )


def build_rasterizer(backend: str) -> BaseRasterizer:
    backend = backend.lower()
    if backend == "cpu":
        return CPURasterizer()
    if backend in {"nvdiffrast", "pytorch3d"}:
        return DifferentiableRasterizer(backend=backend)
    raise ValueError(f"Unsupported rasterizer backend: {backend}")
