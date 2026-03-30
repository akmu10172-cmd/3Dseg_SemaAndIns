# OpenGaussian 语义 + 实例（4步流程）

本项目是基于 OpenGaussian 的缝合流程，当前只使用 OpenGaussian 的 Stage1 特征，再做语义与实例分割。

## 1. 环境与路径约定

- 项目目录：`/mnt/d/3Dseg_SemaAndIns`
- 场景目录（示例）：`/mnt/d/Scene_huijin_0318_q4`
- 2D 语义 mask：`/mnt/d/Scene_huijin_0318_q4/language_features`（`*_s.npy`）
- 2D 实例 mask：`/mnt/d/Scene_huijin_0318_q4/language_features_instance`（`*_inst.npy`）

```bash
cd /mnt/d/3Dseg_SemaAndIns
```

## 2. 流程一：OpenGaussian Stage1 特征训练（到 40000）

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

## 3. 流程二：聚类语义（KMeans 8 类）

```bash
python scripts/cluster_semantic_kmeans.py \
  --input_ply /mnt/d/Scene_huijin_0318_q4/out_stage1/point_cloud/iteration_40000/point_cloud.ply \
  --output_dir /mnt/d/Scene_huijin_0318_q4/out_stage1/kmeans_8 \
  --n_clusters 8 \
  --assign_full \
  --save_npz
```

关键输出：
- `/mnt/d/Scene_huijin_0318_q4/out_stage1/kmeans_8/class_0.ply ... class_7.ply`

## 4. 流程三：确定语义标签并整合模型（只做语义，不做实例）

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
- 这里把 0~7 全部放进 `stuff_ids`，因此只进行语义投票与语义模型整合，不做实例切分。

关键输出：
- `/mnt/d/Scene_huijin_0318_q4/sem_stage/semantic_models/<id_label>/merged_semantic.ply`
- `/mnt/d/Scene_huijin_0318_q4/sem_stage/semantic_instance_report.json`

## 5. 流程四：2D 投影实例分割（proj2d）

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

## 6. 快速验收

重点看：
- `semantic_models_seg3d_instance_report.json` 里 vehicle 的 `num_instances`
- `001_vehicle_sem_ins.ply` 是否语义/实例都合理
