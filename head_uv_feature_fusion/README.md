# Head UV Feature Unprojection（按你给出的 6 步流程）

本实现严格对应以下流程：

1. **DINOv2 dense feature**：对每帧 `I_i` 提取 `F_dino^i ∈ R^{H×W×D}`。
2. **image-space visibility pass**：在 coarse FLAME mesh 上 rasterize，得到可见 face 集 `F_vis^i`。
3. **UV-space rasterization**：得到 `R_UV^i`（每个 texel 的 face id + barycentric）。
4. **visibility 传播**：`UV_visible^i(u,v) = face(u,v) ∈ F_vis^i`。
5. **恢复 3D 点并回投采样**：`(u,v) -> x(u,v) -> p(u,v)`，然后从 `F_dino^i` bilinear sample 得 `UV_feat^i`。
6. **texture-mapped rasterization**：把 `UV_feat^i` 渲回 image-space，得到 `R_feat^i`。

---

## 代码映射

- `extractors.py`
  - `DinoV2Extractor`：冻结 DINOv2 的接口骨架（可接真实权重）
  - `DummyDinoExtractor`：本地离线调试用 extractor
  - `build_feature_extractor`：通过字符串（`dummy` / `dinov2_vits14`）创建 extractor
- `geometry.py`
  - `visible_faces_from_image_space`
  - `uv_space_rasterization`
  - `propagate_visibility_to_uv`
  - `uv_to_surface_points`
  - `sample_feature_map_bilinear`（基于 `grid_sample`，坐标归一化到 `[-1,1]`）
  - `texture_mapped_feature_rendering`
- `pipeline.py`
  - `UVFeatureUnprojectionPipeline` 串联完整流程

---

## 为什么是 bilinear sample

- `p(u,v)` 一般是浮点坐标，不在整数像素上。
- bilinear 比 nearest 更平滑、对训练更稳定。
- 实现上对应 `grid_sample(..., mode="bilinear")`。

---

## 运行

```bash
cd head_uv_feature_fusion
python scripts/run_demo.py
```

调试导出：

```bash
cd head_uv_feature_fusion
python scripts/debug_dump.py
```

会在 `artifacts/debug_dump/` 下导出：
- `F_dino_rgb.ppm`
- `UV_visible.ppm`
- `UV_feat_rgb.ppm`
- `R_feat_rgb.ppm`
- `R_UV_face.npy`, `R_UV_bary.npy`, `UV_position.npy`, `UV_proj_2d.npy`, `UV_proj_grid.npy`


示例：
```python
from head_uv_feature_fusion import build_feature_extractor
extractor = build_feature_extractor("dinov2_vits14", device="cuda")
```
