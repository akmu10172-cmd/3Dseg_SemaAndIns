# OpenGaussian 语义 + 实例（训练 + 两种后处理）

本项目基于 OpenGaussian，当前训练主线使用 Stage1 特征，再做语义/实例分割。

## 1. 启动方式（推荐）

当前主用方式是 **WSL 下启动 Qt 界面**：

```bash
cd /mnt/d/3Dseg_SemaAndIns
bash scripts/run_train_qt.sh
```

如需指定环境里的 Python，可直接传路径：

```bash
bash scripts/run_train_qt.sh /home/ysy/miniconda3/envs/opengaussian/bin/python
```

Windows 一键入口（本质还是调用 WSL）：

```bat
scripts\run_train_qt_windows.cmd
```

Qt 内对应开关：
- `跑后处理`：原始后处理（README 经典流程）
- `跑后处理精简版`：抽视角重渲染图片 + SAM + 回投 3D

## 2. 环境与路径约定

- 项目目录：`/mnt/d/3Dseg_SemaAndIns`
- 场景目录（示例）：`/mnt/d/Scene_huijin_0318_q4`
- 2D 语义 mask：`/mnt/d/Scene_huijin_0318_q4/language_features`（`*_s.npy`）
- 2D 实例 mask：`/mnt/d/Scene_huijin_0318_q4/language_features_instance`（`*_inst.npy`）

## 3. 训练：OpenGaussian Stage1（示例到 40000）

```bash
python train.py \
  -s /mnt/d/Scene_huijin_0318_q4 \
  -m /mnt/d/Scene_huijin_0318_q4/out_stage1 \
  --start_checkpoint /mnt/d/Scene_huijin_0318_q4/out_stage1/chkpnt30000_from_ply_aligned.pth \
  --sam_level 0 \
  --iterations 40000 \
  --start_ins_feat_iter 30000 \
  --save_iterations 40000
```

8G 显存容易 OOM 时可加：

```bash
--resolution 4 --data_device cpu
```

关键输出：
- `/mnt/d/Scene_huijin_0318_q4/out_stage1/point_cloud/iteration_40000/point_cloud.ply`

## 4. 后处理A：原始后处理（README经典流程）

### 4.1 KMeans 聚类（8类）

```bash
python scripts/cluster_semantic_kmeans.py \
  --input_ply /mnt/d/Scene_huijin_0318_q4/out_stage1/point_cloud/iteration_40000/point_cloud.ply \
  --output_dir /mnt/d/Scene_huijin_0318_q4/out_stage1/kmeans_8 \
  --n_clusters 8 \
  --assign_full \
  --save_npz
```

### 4.2 语义投票与语义整合（只做语义）

```bash
python scripts/semantic_instance_pipeline.py \
  --scene_path /mnt/d/Scene_huijin_0318_q4 \
  --cluster_dir /mnt/d/Scene_huijin_0318_q4/out_stage1/kmeans_8 \
  --output_dir /mnt/d/Scene_huijin_0318_q4/sem_stage \
  --mask_level 0 \
  --vote_stride 1 \
  --vote_ignore_ids "0,7" \
  --min_votes 800 \
  --min_top1_ratio 0.4 \
  --id2label "0:background,1:vehicle,2:person,3:bicycle,4:vegetation,5:road,6:traffic_facility,7:other" \
  --stuff_ids "0,1,2,3,4,5,6,7,255"
```

说明：
- 这里把 0~7 全部放入 `stuff_ids`，因此只做语义，不做实例。

### 4.3 基于2D实例mask的投影实例分割（proj2d）

```bash
python scripts/semantic_models_seg3d_instance.py \
  --scene_path /mnt/d/Scene_huijin_0318_q4 \
  --semantic_models_dir /mnt/d/Scene_huijin_0318_q4/sem_stage/semantic_models \
  --output_dir /mnt/d/Scene_huijin_0318_q4/sem_ins_proj2d \
  --semantic_only_ids "1" \
  --stuff_ids "0,4,5,6,7,255" \
  --instance_mask_dir /mnt/d/Scene_huijin_0318_q4/language_features_instance \
  --instance_mask_mode name \
  --instance_mask_suffix "_inst.npy" \
  --instance_mask_level 0 \
  --instance_mask_ignore_ids "0" \
  --vote_stride 1 \
  --instance_min_mask_points 30 \
  --instance_match_iou 0.2 \
  --instance_min_point_votes 2 \
  --min_instance_points 180 \
  --save_instance_parts
```

说明：
- `--semantic_only_ids "1"` 表示只对 vehicle 做实例分割。

关键输出：
- `/mnt/d/Scene_huijin_0318_q4/sem_ins_proj2d/semantic_instance/001_vehicle_sem_ins.ply`
- `/mnt/d/Scene_huijin_0318_q4/sem_ins_proj2d/semantic_instance/001_vehicle_instances/instance_*.ply`
- `/mnt/d/Scene_huijin_0318_q4/sem_ins_proj2d/semantic_models_seg3d_instance_report.json`

## 5. 后处理B：简化版（抽视角重渲染 + SAM + 回投3D）

简化版核心脚本：
- `scripts/topdown_sam3_instance_pipeline.py`

流程：
1. 从输入点云抽取 topdown 视角并重渲染 `renders_topdown`
2. 对渲染图跑 SAM3 实例分割
3. 将 2D 实例结果投影回 3D，输出 `semantic_id + instance_id` 点云

示例命令：

```bash
python scripts/topdown_sam3_instance_pipeline.py \
  --scene_path /mnt/d/Scene_huijin_0318_q4 \
  --input_ply /mnt/d/Scene_huijin_0318_q4/out_stage1/point_cloud/iteration_40000/point_cloud.ply \
  --output_subdir topdown_instance_pipeline \
  --num_views 8 \
  --image_size 1024 \
  --sam3_python /mnt/d/sam3/.conda/envs/sam3/bin/python \
  --sam3_checkpoint /mnt/d/modelscope/hub/models/facebook/sam3/sam3.pt \
  --sam3_device cuda \
  --sam3_prompts "vehicle" \
  --semantic_id 1 \
  --min_instance_points 120
```

常见输出目录：
- `<scene_path>/topdown_instance_pipeline/renders_topdown`
- `<scene_path>/topdown_instance_pipeline/sam3_topdown`
- `<scene_path>/topdown_instance_pipeline/instance_projected_sem_ins.ply`
- `<scene_path>/topdown_instance_pipeline/report.json`

## 6. 快速验收

原始后处理看：
- `semantic_models_seg3d_instance_report.json` 里 vehicle 的 `num_instances`
- `001_vehicle_sem_ins.ply` 的语义/实例是否合理

简化版看：
- `report.json` 中 `instances_after`、`points_used` 等统计
- `instance_projected_sem_ins.ply` 是否出现合理实例拆分
