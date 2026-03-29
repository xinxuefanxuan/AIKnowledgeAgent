# Head UV Feature Unprojection (按论文图示流程)

已按你的确认继续升级：
- ✅ 图像空间可见性改为**逐像素 z-buffer**（不是面平均深度）
- ✅ 增加**可插拔 rasterizer 接口**，默认 `cpu`，并预留 `nvdiffrast/pytorch3d` 对接点

---

## Pipeline（对应图示）

1. `I_i -> DINOv2` 提取 `F_dino^i ∈ R^{H×W×D}`。
2. **Image-space visibility pass**（粗 FLAME mesh + z-buffer）得到 `F_vis^i`。
3. **UV-space rasterization pass**（仅 rasterize 可见面）得到：
   - `UV_visible^i(u,v)`
   - `R_coord^i`（face id + barycentric）
4. barycentric 恢复 `x(u,v)`（UV texel 对应 3D 点）。
5. 投影 `p(u,v)=π(x(u,v),K,R,t)`。
6. 双线性采样 `F_dino^i(p(u,v))`。
7. `UV_feat^i(u,v)=UV_visible^i(u,v) * F_dino^i(p(u,v))`。
8. （可选）`UV_feat + UV_position` 输入编码器得到 `UV_feat_encoded`。

---

## 核心模块

- `rasterizer.py`
  - `BaseRasterizer`：后端抽象
  - `CPURasterizer`：当前可运行版本（逐像素 barycentric + depth z-buffer）
  - `DifferentiableRasterizer`：预留 `nvdiffrast/pytorch3d` 对接入口
- `geometry.py`
  - `visible_faces_from_image_space(..., rasterizer=...)`
  - `rasterize_uv_to_image_space(..., rasterizer=...)`
  - `uv_to_surface_points`
  - `project_points` / `sample_feature_map_bilinear`
- `pipeline.py`
  - `UVFeatureUnprojectionPipeline(..., rasterizer_backend="cpu")`

---

## 运行 demo

```bash
cd head_uv_feature_fusion
python scripts/run_demo.py
```

默认使用：
- `rasterizer_backend: cpu`

---

## 接入 VHAP/FLAME 建议

你已有的 `flame + uvmask + 相机内外参` 可以直接接：

- `mesh.vertices` <- 当前帧 FLAME 顶点
- `mesh.faces` <- FLAME 拓扑
- `mesh.uv_coords` <- 顶点 UV
- `camera.K/R/t/image_size` <- 相机参数
- `uv_mask` <- VHAP 的 UV mask（可选）

如果你要上训练并反传梯度：
1. 在 `DifferentiableRasterizer.rasterize()` 中接入 nvdiffrast/PyTorch3D 输出 `face_map/bary_map/depth_map`。
2. 保持 `pipeline.py` 其余逻辑不变（可直接复用）。
