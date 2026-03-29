# Head UV Feature Unprojection (严格按你描述的流程)

当前代码的 `UVFeatureUnprojectionPipeline` 已与下面流程一一对应：

输入图像 `I_i`
→ DINOv2 得到 `F_dino^i`
→ coarse FLAME 做 image-space rasterization
→ 得到可见 faces `F_vis^i`
→ mesh 做 UV-space rasterization
→ 得到 `R_UV^i`（face id + barycentric）
→ 把 image-space visibility 传播到 UV 域
→ 得到 `UV_visible^i`
→ 对每个可见 texel 恢复 `x(u,v)`
→ 投回图像平面 `p(u,v)`
→ 从 `F_dino^i` 上 bilinear sample
→ 得到 `UV_feat^i`
→ texture-mapped rasterization
→ 得到 `R_feat^i`

---

## 关键输出（`pipeline.run`）

- `F_vis`: 可见 face id 集合
- `R_UV_face`, `R_UV_bary`: UV 渲染图（face+重心）
- `UV_visible`: 当前帧可见 UV mask
- `UV_position`: 每个 texel 的 3D 点 `x(u,v)`
- `UV_proj_2d`: `p(u,v)`
- `UV_feat`: UV feature texture
- `UV_feat_encoded`: 可选编码后的 UV feature
- `R_feat`: 从 UV 贴图反向渲染得到的 image-space feature rendering

---

## 模块说明

- `rasterizer.py`
  - `CPURasterizer`: 逐像素 z-buffer（可运行）
  - `DifferentiableRasterizer`: 预留 nvdiffrast/PyTorch3D 对接
- `geometry.py`
  - `visible_faces_from_image_space`
  - `uv_space_rasterization`
  - `propagate_visibility_to_uv`
  - `uv_to_surface_points`
  - `texture_mapped_feature_rendering`
- `pipeline.py`
  - 串联完整流程并返回中间结果，便于调试每一步是否正确

---

## 运行

```bash
cd head_uv_feature_fusion
python scripts/run_demo.py
```


调试导出（中间结果可视化/落盘）：

```bash
cd head_uv_feature_fusion
python scripts/debug_dump.py
```

会在 `artifacts/debug_dump/` 下生成：
- `F_dino_rgb.ppm`
- `UV_visible.ppm`
- `UV_feat_rgb.ppm`
- `R_feat_rgb.ppm`
- `R_UV_face.npy`, `R_UV_bary.npy`, `UV_position.npy`, `UV_proj_2d.npy`
