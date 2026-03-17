# Semantic + Instance Pipeline (MaskID 0-7)

本流程适用于你当前场景：  
- 已完成 Stage1（有 `ins_feat_*`）  
- 已有 8 类语义 `maskid=0..7`  
- 希望得到 `语义 + 实例` 双重分割

## 1) KMeans 先切成 8 个无语义子模型

```powershell
conda run -n <your_env> python scripts/cluster_semantic_kmeans.py `
  --input_ply <MODEL_PATH>\point_cloud\iteration_<ITER>\point_cloud.ply `
  --output_dir <OUT_DIR>\kmeans_8 `
  --n_clusters 8 `
  --opacity_threshold 0.05 `
  --scale_threshold_log 2.0 `
  --assign_full `
  --save_npz
```

输出：`<OUT_DIR>\kmeans_8\class_0.ply ... class_7.ply`

## 2) 反投票关联 + 语义重命名 + 语义内 DBSCAN 实例分割

```powershell
conda run -n <your_env> python scripts/semantic_instance_pipeline.py `
  --scene_path <SCENE_PATH> `
  --cluster_dir <OUT_DIR>\kmeans_8 `
  --output_dir <OUT_DIR>\sem_ins `
  --mask_level 0 `
  --vote_stride 2 `
  --min_votes 2000 `
  --min_top1_ratio 0.55 `
  --id2label "0:background,1:vehicle,2:person,3:bicycle,4:vegetation,5:road,6:traffic_facility,7:other" `
  --stuff_ids "0,4,5,6,7,255" `
  --dbscan_eps -1 `
  --dbscan_eps_quantile 0.92 `
  --dbscan_min_samples 20 `
  --min_instance_points 300
```

说明：
- `--dbscan_eps -1` 表示自动估计 eps（基于 kNN 距离分位数）
- `stuff_ids` 会跳过实例分割（通常道路/植被/背景这类）
- 如果你的 `transforms_test.json` 不存在，脚本会自动只用 `transforms_train.json`

## 3) 输出目录说明

- `sem_ins/semantic_models/`
  - `XXX_<label>/class_*.ply`：按投票语义归档后的子模型
  - `XXX_<label>/merged_semantic.ply`：同语义聚合模型
- `sem_ins/semantic_instance/`
  - `XXX_<label>_sem_ins.ply`：带 `semantic_id` + `instance_id`
  - `XXX_<label>_instances/instance_*.ply`：实例拆分单体
- `sem_ins/semantic_instance_report.json`
  - 每个 cluster 的投票统计、置信度、映射结果
  - 每个语义的实例数量和 DBSCAN 参数

## 4) 常用调参

- 语义映射不稳定：提高 `--min_top1_ratio`（如 0.6~0.7）  
- 投票样本太少：减小 `--vote_stride`、增大 `--core_ratio`  
- 实例粘连：降低 `--dbscan_eps_quantile` 或增大 `--dbscan_min_samples`  
- 实例过分裂：提高 `--dbscan_eps_quantile` 或降低 `--dbscan_min_samples`

## 5) 车辆粘连增强（推荐）

当相邻车辆在 `XYZ-only DBSCAN` 中粘连时，建议只对车辆启用 `ins_feat + xyz` 联合聚类：

```bash
python scripts/semantic_instance_pipeline.py \
  --scene_path /mnt/d/Scene_roi \
  --cluster_dir /mnt/d/Scene/out/kmeans_20260307_1948 \
  --output_dir /mnt/d/Scene_roi/sem_ins_vehicle_refine \
  --transforms_files "" \
  --core_mode centroid \
  --vote_ignore_ids "0,7" \
  --min_top1_ratio 0.40 \
  --instance_only_ids "1" \
  --instance_space xyz_feat \
  --instance_spatial_weight 1.0 \
  --instance_feature_weight 1.6 \
  --instance_z_weight 0.5 \
  --vehicle_eps_quantile 0.86 \
  --vehicle_min_samples 24 \
  --vehicle_min_instance_points 180 \
  --id2label "0:background,1:vehicle,2:person,3:bicycle,4:vegetation,5:road,6:traffic_facility,7:other,255:unknown"
```

说明：
- `--instance_only_ids "1"`：仅处理 vehicle 实例，避免其他类干扰。
- `--instance_space xyz_feat`：在实例聚类时联合使用几何和 `ins_feat`。
- `--instance_feature_weight` 越大，越倾向按特征拆分（可缓解“挨着车粘连”）。

## 6) Use seg3dmodel 2D instance masks (proj2d)

If you already have per-view instance masks exported by `D:\seg3dmodel-master` (for example `sam_mask_instance_view_0000.png`),
you can switch instance segmentation from `dbscan` to `proj2d`.

```powershell
conda run -n <your_env> python scripts/semantic_instance_pipeline.py `
  --scene_path <SCENE_PATH> `
  --cluster_dir <OUT_DIR>\kmeans_8 `
  --output_dir <OUT_DIR>\sem_ins_proj2d `
  --mask_level 0 `
  --vote_stride 2 `
  --min_votes 2000 `
  --min_top1_ratio 0.55 `
  --id2label "0:background,1:vehicle,2:person,3:bicycle,4:vegetation,5:road,6:traffic_facility,7:other" `
  --stuff_ids "0,4,5,6,7,255" `
  --instance_method proj2d `
  --instance_mask_dir "D:\seg3dmodel-master\multicam" `
  --instance_mask_mode index `
  --instance_mask_index_pattern "sam_mask_instance_view_{index:04d}.png" `
  --instance_mask_ignore_ids "0" `
  --instance_min_mask_points 40 `
  --instance_match_iou 0.2 `
  --instance_min_point_votes 2 `
  --min_instance_points 300
```

Notes:
- `--instance_mask_mode index` + `--instance_mask_index_pattern` is designed for seg3dmodel naming.
- If your masks are named by frame file, use `--instance_mask_mode name --instance_mask_suffix "_instance.png"`.
- `proj2d` keeps semantic assignment by 2D semantic voting, then does instance assignment by multi-view 2D instance-mask voting.
