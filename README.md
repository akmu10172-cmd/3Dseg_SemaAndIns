# OpenGaussian 语义 + seg3D实例 完整流程文档

本文档适用于你当前流程：
- 从 Stage1 训练点云开始
- KMeans 聚类得到 class_*.ply
- 2D 语义mask投票完成语义映射
- 对指定语义类别做 seg3D 风格实例分割（proj2d）

## 1. 环境与路径约定

WSL 示例路径：
- 项目目录：`/mnt/d/OpenGaussian-original-20260310`
- 场景目录：`/mnt/d/Scene_roi`
- 2D 语义mask目录：`/mnt/d/Scene_roi/language_features`（`*_s.npy`）
- 2D 实例mask目录：`/mnt/d/Scene_roi/language_features_instance`（`*_inst.npy`）

进入项目：

```bash
cd /mnt/d/OpenGaussian-original-20260310
```

## 2. Stage1 训练流程

### 2.1 直接跑官方脚本（ScanNet / LeRF）

```bash
chmod +x scripts/train_scannet.sh
./scripts/train_scannet.sh
```

或：

```bash
chmod +x scripts/train_lerf.sh
./scripts/train_lerf.sh
```

说明：
- 这两个脚本会跑完整训练（Stage0 -> Stage3）
- 请先修改脚本里的数据路径和 GPU 编号

### 2.2 只训练到Open Gaussian的Stage1

只需要 Stage1 后的点云用于后续语义+实例流程，建议用手动命令只跑到 Stage1 结束：

```bash
训练40000
 python train.py  
 -s /mnt/d/Scene_roi   
 -m /mnt/d/Scene/out_original_roi_optm  
 --start_checkpoint /mnt/d/Scene/out/chkpnt30000.pth   
 --sam_level 0   
 --iterations 40000   
 --start_ins_feat_iter 30000   
 --start_root_cb_iter 100000  
 --start_leaf_cb_iter 120000  
 --checkpoint_iterations 34000 40000  
 --save_iterations 40000
```

说明：
- 输出点云在：
  - `/mnt/d/Scene/out_stage1/point_cloud/iteration_50000/point_cloud.ply`

## 3. Stage1 点云做 KMeans（8类）

```bash
python scripts/cluster_semantic_kmeans.py \
  --input_ply /mnt/d/Scene/out_stage1/point_cloud/iteration_50000/point_cloud.ply \
  --output_dir /mnt/d/Scene/out_stage1/kmeans_8 \
  --n_clusters 8 \
  --assign_full \
  --save_npz
```

输出核心文件：
- `/mnt/d/Scene/out_stage1/kmeans_8/class_0.ply ... class_7.ply`

## 4. 一体化流程：语义投票 + 语义模型合并 + seg3D实例（proj2d）

脚本：`scripts/semantic_instance_pipeline.py`

```bash
python scripts/semantic_instance_pipeline.py \
  --scene_path /mnt/d/Scene_roi \
  --cluster_dir /mnt/d/Scene/out_stage1/kmeans_8 \
  --output_dir /mnt/d/Scene_roi/sem_ins_proj2d \
  --mask_level 0 \
  --vote_stride 1 \
  --vote_ignore_ids "0,7" \
  --min_votes 800 \
  --min_top1_ratio 0.4 \
  --id2label "0:background,1:vehicle,2:person,3:bicycle,4:vegetation,5:road,6:traffic_facility,7:other" \
  --stuff_ids "0,4,5,6,7,255" \
  --instance_only_ids "1" \
  --instance_method proj2d \
  --instance_mask_dir /mnt/d/Scene_roi/language_features_instance \
  --instance_mask_mode name \
  --instance_mask_suffix "_inst.npy" \
  --instance_mask_level 0 \
  --instance_mask_ignore_ids "0" \
  --instance_min_mask_points 30 \
  --instance_match_iou 0.2 \
  --instance_min_point_votes 2 \
  --min_instance_points 180
```

说明：
- `--instance_method proj2d`：使用 seg3D 风格 2D 实例反投影
- `--instance_only_ids "1"`：只对 vehicle 做实例分割
- `--stuff_ids` 中的类别不会做实例分割
- `--vote_ignore_ids "0,7"`：语义投票时忽略背景/other，提升 vehicle 归类稳定性

输出核心文件：
- `.../sem_ins_proj2d/semantic_models/<id_label>/merged_semantic.ply`
- `.../sem_ins_proj2d/semantic_instance/001_vehicle_sem_ins.ply`
- `.../sem_ins_proj2d/semantic_instance/001_vehicle_instances/instance_*.ply`
- `.../sem_ins_proj2d/semantic_instance_report.json`

## 5. 已有语义模型时，只跑实例分割

如果你已经有 `semantic_models`（比如 `sem_ins_center_vote/semantic_models`），可以跳过语义投票，直接跑实例：

脚本：`scripts/semantic_models_seg3d_instance.py`

```bash
python scripts/semantic_models_seg3d_instance.py \
  --scene_path /mnt/d/Scene_roi \
  --semantic_models_dir /mnt/d/Scene_roi/sem_ins_center_vote/semantic_models \
  --output_dir /mnt/d/Scene_roi/sem_ins_seg3d_instance \
  --semantic_only_ids "1" \
  --stuff_ids "0,4,5,6,7,255" \
  --instance_mask_dir /mnt/d/Scene_roi/language_features_instance \
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

## 6. 常用参数速查

- 指定做实例的语义类：
  - 一体化脚本：`--instance_only_ids "1"`
  - 实例-only脚本：`--semantic_only_ids "1"`
- 实例容易粘连：减小 `--instance_match_iou` 或减小 `--min_instance_points`
- 实例噪声多：增大 `--instance_min_point_votes` 或增大 `--instance_min_mask_points`
- 语义归类不稳：保持 `--vote_ignore_ids "0,7"`，必要时降低 `--min_top1_ratio`

## 7. 快速验收

先看这两个文件：
- `semantic_instance/001_vehicle_sem_ins.ply`
- `semantic_instance_report.json`

重点检查：
- `semantic_summary` 里 `semantic_id=1` 的 `num_instances`
- `cluster_infos` 里 vehicle cluster 的 `accepted` 是否为 `true`
- `assigned_semantic_id` 是否为预期类别（如 `1`）
