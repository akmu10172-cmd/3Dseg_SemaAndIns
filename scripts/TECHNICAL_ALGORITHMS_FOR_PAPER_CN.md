# OpenGaussian 语义-实例分割技术文档（论文改写版）

本文档面向论文写作，提炼项目中的核心算法、公式、模块边界与可复现实验要点。

## 1. 任务定义

输入：
- 训练后的 OpenGaussian 点云与实例特征（`ins_feat_*`）。
- 多视角 2D 语义掩码（`*_s.npy`）。
- 可选多视角 2D 实例掩码（如 `*_inst.npy` / `sam_mask_instance_view_XXXX.png`）。

输出：
- 点级 `semantic_id` 与 `instance_id`（可写回 PLY）。

目标拆解：
1. 在 OpenGaussian 中学习连续/离散实例表示。
2. 将点云聚成无监督语义候选簇（`class_*.ply`）。
3. 用 3D->2D 投票将候选簇映射到语义 ID。
4. 在每个语义内做实例分割（DBSCAN 或 seg3D 风格 `proj2d`）。

---

## 2. 记号

- 点集：\(\mathcal{P}=\{p_i\}_{i=1}^{N}\), \(p_i=(x_i,y_i,z_i)\)
- 点特征：\(f_i\in\mathbb{R}^{d_f}\)（默认 \(d_f=6\)）
- 相机集合：\(\mathcal{C}=\{c_v\}_{v=1}^{V}\)
- 语义掩码：\(M_v^{sem}(u,v)\)
- 实例掩码：\(M_v^{ins}(u,v)\)
- 投影算子：\(\Pi_v(\cdot)\)

---

## 3. OpenGaussian 训练主算法（Stage0-Stage3）

代码对应：
- [train.py](/d:/OpenGaussian-original-20260310/train.py)
- [scene/kmeans_quantize.py](/d:/OpenGaussian-original-20260310/scene/kmeans_quantize.py)
- [utils/opengs_utlis.py](/d:/OpenGaussian-original-20260310/utils/opengs_utlis.py)

### 3.1 Stage0：3DGS 预训练

在 `start_ins_feat_iter` 前，优化几何与外观（RGB 重建主导）。

### 3.2 Stage1：连续实例特征学习

关键损失：

1) 掩码内聚（代码注释 Eq.(1)）
\[
\mathcal{L}_{coh}=
\frac{1}{N_m}\sum_{m}\frac{1}{|m|}\sum_{q\in m}\|f(q)-\bar f_m\|_2
\]

2) 掩码间分离（代码注释 Eq.(2)）
\[
\mathcal{L}_{sep}\propto
\frac{1}{N_m(N_m-1)}
\sum_{a\neq b}\frac{w_{ab}}{\|\bar f_a-\bar f_b\|_2^2+\epsilon}
\]

实现函数：
- `cohesion_loss(...)`
- `separation_loss(...)`

### 3.3 Stage2：两级码本离散化

`Quantize_kMeans` 两层结构：
- Root（粗层）：`[ins_feat, xyz]` 聚类。
- Leaf（细层）：在各 Root 内对 `ins_feat` 再聚类。

直通估计形式：
\[
f_i^{q}=f_i-\text{stopgrad}(f_i)+c_{z_i}
\]

### 3.4 Stage3：2D 语言特征与 3D 叶子簇关联

核心评分（代码注释 Eq.(5)）：
\[
S=\text{IoU}\cdot(1-d_{L1})
\]

输出文件：
- `cluster_lang.npz`（`leaf_feat`, `leaf_score`, `occu_count`, `leaf_ind`）

---

## 4. 语义候选聚类（后处理）

代码对应：
- [scripts/cluster_semantic_kmeans.py](/d:/OpenGaussian-original-20260310/scripts/cluster_semantic_kmeans.py)

流程：
1. 依据 opacity/scale 过滤有效点。
2. 构建聚类向量（`ins_feat` 或 `ins_feat+xyz`）。
3. KMeans 得到 \(K\) 个簇（默认8）。
4. 输出 `class_0.ply ... class_7.ply`。

该步骤得到“无监督语义候选簇”。

---

## 5. 一体化语义投票 + 实例分割

代码对应：
- [scripts/semantic_instance_pipeline.py](/d:/OpenGaussian-original-20260310/scripts/semantic_instance_pipeline.py)

### 5.1 语义投票映射（3D->2D）

对每个候选簇：
1. 采样 core points（`quality/centroid/mixed`）。
2. 投影到各视角读取 `M_v^{sem}`。
3. 累积类别票数：
\[
\text{votes}(k)=\sum_{v}\sum_{p\in core}\mathbf{1}[M_v^{sem}(\Pi_v(p))=k]
\]
4. 计算 top1 比例：
\[
r_{top1}=\frac{\max_k \text{votes}(k)}{\sum_k \text{votes}(k)}
\]
5. 若 `total_votes` 与 `top1_ratio` 达阈值则接受，否则映射 `unknown_id`（默认255）。

### 5.2 语义内实例分割（两种）

1) `dbscan`：在 `xyz` 或 `xyz+ins_feat` 空间聚类。  
2) `proj2d`（seg3D 风格）：
- 将语义点投影到 2D 实例掩码。
- 形成每视角局部实例点集。
- 跨视角按 IoU 合并局部实例。
- 点级投票选全局实例，剔除低票/小实例。

实现函数：
- `proj2d_instances(...)`

---

## 6. 仅实例阶段（已有语义模型时）

代码对应：
- [scripts/semantic_models_seg3d_instance.py](/d:/OpenGaussian-original-20260310/scripts/semantic_models_seg3d_instance.py)

用途：
- 已有 `semantic_models/*/merged_semantic.ply` 时，跳过语义投票，直接做 `proj2d` 实例。

关键参数：
- `--semantic_only_ids`：只处理指定语义
- `--stuff_ids`：跳过无需实例化的类别

---

## 7. 复杂度与工程特性

### 7.1 复杂度主项

- 语义投票：约 \(O(N_{core}\cdot V)\)
- proj2d 实例：约 \(O(N_{sem}\cdot V)\) + 跨视角集合合并
- DBSCAN：随邻域图密度增大而显著变慢

### 7.2 常见失败模式

1. 语义被判为 `255`：  
`top1_ratio` 太低或有效票太少。  
对策：`vote_ignore_ids`、`min_top1_ratio`、`vote_stride`、`core_ratio` 联动调参。

2. 车辆侧面缺失：  
俯视 mask 占主导，侧面证据不足。  
对策：补充斜视角 mask、降低 `instance_min_point_votes`、降低 `instance_match_iou`。

3. 相邻车粘连：  
对策：降低 `instance_match_iou`、增大 `instance_min_point_votes`，或对比 `xyz_feat` 型 DBSCAN。

---

## 8. 论文可用消融设计

建议最小消融矩阵：

1. 语义投票鲁棒性：
- `core_mode`（quality/centroid/mixed）
- `vote_stride`
- `vote_ignore_ids`
- `min_top1_ratio`

2. 实例方法对比：
- `dbscan` vs `proj2d`
- `proj2d` with/without 斜视角增广

3. proj2d 融合策略：
- `instance_match_iou`
- `instance_min_point_votes`
- `min_instance_points`

4. 表征层消融：
- KMeans: `ins_feat` vs `ins_feat+xyz`

---

## 9. 复现记录建议（论文实验必备）

每次实验建议保存：
- Git 版本号（commit hash）
- 完整命令行
- `semantic_instance_report.json` 或 `semantic_models_seg3d_instance_report.json`
- 2D 实例mask数据版本与生成时间
- 相机来源（`transforms` / `colmap`）

---

## 10. 论文章节映射建议

1. 方法主体：OpenGaussian 四阶段学习（Stage0-3）  
2. 后处理语义建模：KMeans + 3D->2D 语义投票  
3. 实例建模：seg3D 风格 `proj2d` 多视角实例投票  
4. 消融与误差分析：参数敏感性、视角分布、失败案例
