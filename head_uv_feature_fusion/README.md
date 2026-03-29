# Head UV Feature Fusion (DINO + Mesh + Transformer)

这个项目是一个**可运行的最小原型**，用于把你描述的人头图像信息串起来：

1. 输入视频分帧得到的人头图像。
2. 提取 DINO 特征图 `F_img ∈ R[H,W,C]`。
3. 读取/构建头部 mesh 顶点 `V ∈ R[N,3]`，并用 FLAME/VHAP 提供的参数进行位姿与相机投影。
4. 利用 UV 展开（vertex-to-uv 对应）把 3D 信息映射到 UV 平面。
5. 通过 Transformer 融合图像 token 和 mesh token，输出 UV 特征图 `F_uv ∈ R[H_uv,W_uv,C]`。

> 你在问题中提到的 `[H,W,N]` 这里建议把最后一个维度当作**通道维 C**（feature dimension）而不是顶点数 N。顶点数通常用于 mesh token 数量，特征通道建议单独记为 C。

---

## 项目结构

```text
head_uv_feature_fusion/
├── configs/
│   └── default.yaml          # 参数示例
├── scripts/
│   └── run_demo.py           # 合成数据 demo（可直接运行）
├── src/head_uv_feature_fusion/
│   ├── data_types.py         # 数据结构定义
│   ├── geometry.py           # 相机投影 / UV 栅格化辅助
│   ├── model.py              # Transformer 融合模型
│   ├── pipeline.py           # 端到端流程封装
│   └── __init__.py
└── README.md
```

---

## 快速开始

```bash
cd head_uv_feature_fusion
python scripts/run_demo.py
```

输出示例：
- 输入：`image_feature_map shape = (64, 64, 128)`
- 输出：`uv_feature_map shape = (128, 128, 128)`

---

## 关键思路解释

### 1) 你说的“mesh 和 UV 对应”具体是什么

- 每个 mesh 顶点 `v_i` 对应一个 UV 坐标 `uv_i=(u_i,v_i)`（来自 FLAME 模型或贴图参数）。
- 三角面片在 3D 空间与 UV 空间是同拓扑（同一组三角形索引）。
- 所以可以把任意顶点特征插值到 UV 网格上，得到 `F_uv`。

### 2) 为什么要有相机内外参

VHAP/跟踪器给出：
- 外参：`R, t`（头部/世界到相机）
- 内参：`K`（焦距、主点）

用它们可把 3D 顶点投影到图像平面，建立：
- 顶点 ↔ 图像特征采样点
- 顶点 ↔ UV 位置

从而实现“图像特征注入 UV”。

### 3) Transformer 融合做了什么

- 图像特征图展开为 image tokens。
- mesh 顶点特征构成 mesh tokens。
- 使用 cross-attention，让 mesh token 从 image token 读取上下文。
- 再把增强后的 mesh token 栅格化到 UV 平面。

这能比“纯几何最近邻拷贝”更稳健地融合视角、光照、遮挡信息。

### 4) 与 FLAME / VHAP 真实接入

本 demo 用的是合成数据接口，真实接入时：
1. 替换 `PipelineInput` 中的 `vertices, uv_coords, faces, K,R,t` 为 VHAP 导出。
2. DINO 特征替换为真实提取结果。
3. 若有 `uv_mask`，在 UV 栅格化后做 mask。
4. 多帧情况下做时序聚合（EMA / temporal transformer）。

---

## 下一步建议（可继续扩展）

- 接入真实 VHAP 文件解析器（json / npz）。
- 加入可见性判断（z-buffer）避免背面污染。
- 用可微 rasterizer（如 PyTorch3D/nvdiffrast）替换当前简化版插值。
- 训练目标可用：重建损失 + 感知损失 + 跨帧一致性损失。
