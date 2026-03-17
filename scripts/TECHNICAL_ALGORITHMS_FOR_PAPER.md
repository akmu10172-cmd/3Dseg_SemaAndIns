# OpenGaussian + Semantic Voting + seg3D-Style Instance: Technical Notes

This document summarizes the project as a paper-oriented technical reference, with algorithm details mapped to implementation files for future paper writing and method extension.

## 1. Problem Setting

Given:
- A trained OpenGaussian model with point-wise geometry and learned instance features (`ins_feat_*`).
- Multi-view 2D semantic masks (SAM-based, `*_s.npy`).
- Optional multi-view 2D instance masks (seg3D-style outputs, e.g. `*_inst.npy` or `sam_mask_instance_view_XXXX.png`).

Target:
- Produce point-wise `semantic_id` and `instance_id` for 3D Gaussians.

Core decomposition:
1. Train continuous/discrete instance features in OpenGaussian.
2. Cluster Gaussians into semantic candidate groups (`class_*.ply`).
3. Assign semantic class to each group via 3D-to-2D voting.
4. Run per-semantic instance segmentation (DBSCAN or projection voting from 2D instance masks).

---

## 2. Notation

- Gaussian points: \( \mathcal{P}=\{p_i\}_{i=1}^{N} \), \( p_i=(x_i,y_i,z_i) \)
- Learned point feature (Stage1): \( f_i \in \mathbb{R}^{d_f} \) (`ins_feat`, default \(d_f=6\))
- Camera set: \( \mathcal{C}=\{c_v\}_{v=1}^{V} \), each with intrinsics/extrinsics
- 2D semantic mask in view \(v\): \( M_v^{sem}(u,v) \in \{0,\dots,K\} \)
- 2D instance mask in view \(v\): \( M_v^{ins}(u,v) \in \{0,\dots\} \)

---

## 3. OpenGaussian Training Pipeline (Stage0-Stage3)

Implementation:
- [train.py](/d:/OpenGaussian-original-20260310/train.py)
- [scene/kmeans_quantize.py](/d:/OpenGaussian-original-20260310/scene/kmeans_quantize.py)
- [utils/opengs_utlis.py](/d:/OpenGaussian-original-20260310/utils/opengs_utlis.py)

### 3.1 Stage0: RGB Pretraining (3DGS)

Before `start_ins_feat_iter`, optimize standard rendering objectives (L1 + SSIM-style term used in repo) for geometry/appearance.

### 3.2 Stage1: Continuous Instance Feature Learning

Main losses:

1) Intra-mask cohesion (Eq.(1) in code comment):
\[
\mathcal{L}_{coh}=
\frac{1}{N_m}\sum_{m}
\frac{1}{|m|}\sum_{q\in m}\left\|f(q)-\bar f_m\right\|_2
\]
where \(\bar f_m\) is mask-wise mean feature.

2) Inter-mask separation (contrastive style, Eq.(2) in code comment):
\[
\mathcal{L}_{sep}\propto
\frac{1}{N_m(N_m-1)}
\sum_{a\neq b}\frac{w_{ab}}{\| \bar f_a-\bar f_b\|_2^2+\epsilon}
\]

Code hooks:
- `cohesion_loss(...)`
- `separation_loss(...)`

### 3.3 Stage2: Two-Level Codebook Discretization

`Quantize_kMeans` performs hierarchical quantization:
- Root level (coarse): cluster on `[ins_feat, xyz]` (dim=9 by default).
- Leaf level (fine): cluster on `ins_feat` (dim=6) inside each root cluster.

Straight-through style update is used:
\[
f_i^{q} = f_i - \text{stopgrad}(f_i) + c_{z_i}
\]
where \(c_{z_i}\) is assigned center.

### 3.4 Stage3: 2D Language Feature to 3D Cluster Association

For rendered leaf clusters and pseudo masks:
- IoU matrix from masks.
- Feature distance matrix from mask means.
- Joint score:
\[
S = \text{IoU}\cdot(1-d_{L1})
\]

Outputs:
- `cluster_lang.npz` containing `leaf_feat`, `leaf_score`, `occu_count`, `leaf_ind`.

---

## 4. Semantic Candidate Clustering (Post-training)

Implementation:
- [scripts/cluster_semantic_kmeans.py](/d:/OpenGaussian-original-20260310/scripts/cluster_semantic_kmeans.py)

Input:
- Point cloud PLY from trained model (often Stage1 snapshot).

Procedure:
1. Select valid points by opacity/scale thresholds.
2. Build clustering vector using `ins_feat` (optionally concat `xyz`).
3. KMeans to \(K\) clusters (default 8).
4. Save each cluster as `class_i.ply`.

This stage gives *unsupervised semantic candidates*.

---

## 5. Semantic Voting + Instance Segmentation (Unified Pipeline)

Implementation:
- [scripts/semantic_instance_pipeline.py](/d:/OpenGaussian-original-20260310/scripts/semantic_instance_pipeline.py)

### 5.1 Semantic Voting by 3D-to-2D Projection

For each cluster:
1. Sample core points (quality/centroid/mixed strategy).
2. Project core points to each selected camera.
3. Read 2D semantic IDs from `M_v^{sem}`.
4. Aggregate votes:
\[
\text{votes}(k)=\sum_{v}\sum_{p\in \text{core}} \mathbf{1}[M_v^{sem}(\Pi_v(p))=k]
\]
5. Compute top-1 ratio:
\[
r_{top1}=\frac{\max_k \text{votes}(k)}{\sum_k \text{votes}(k)}
\]
6. Accept class if:
- `total_votes >= min_votes`
- `top1_ratio >= min_top1_ratio`
otherwise assign `unknown_id` (default 255).

### 5.2 Instance Segmentation Methods

#### Method A: `dbscan`
Per semantic group, run DBSCAN in:
- `xyz`, or
- fused `xyz + ins_feat` space.

#### Method B: `proj2d` (seg3D-style projection voting)
1. Project semantic points to each view.
2. Read per-pixel instance IDs from `M_v^{ins}`.
3. Build per-view local instance point sets.
4. Merge local sets across views by IoU threshold (`instance_match_iou`).
5. For each point, choose global instance with max vote count.
6. Drop low-vote and tiny instances.

This is implemented by `proj2d_instances(...)`.

---

## 6. Instance-Only Pipeline on Existing Semantic Models

Implementation:
- [scripts/semantic_models_seg3d_instance.py](/d:/OpenGaussian-original-20260310/scripts/semantic_models_seg3d_instance.py)

Use case:
- You already have `semantic_models/*/merged_semantic.ply`.
- You only want to rerun seg3D-style instance segmentation (no semantic voting rerun).

Key controls:
- `--semantic_only_ids`: only selected semantic IDs
- `--stuff_ids`: skip non-instance classes

---

## 7. Complexity and Practical Behavior

### 7.1 Main Complexity Drivers

- Projection voting: \(O(N_{core}\cdot V)\) for semantic voting.
- proj2d instance voting: \(O(N_{sem}\cdot V)\) per semantic group + set-merging cost.
- DBSCAN: depends on neighborhood graph; practical bottleneck for large `N_sem`.

### 7.2 Typical Failure Modes

1. Semantic rejected as unknown (`255`):
- Cause: low `top1_ratio` or insufficient votes.
- Fix: tune `vote_ignore_ids`, `min_top1_ratio`, `vote_stride`, `core_ratio`.

2. Side surfaces of vehicles missing in proj2d:
- Cause: masks dominated by top-view evidence.
- Fix: add oblique synthetic views/masks; reduce `instance_min_point_votes`; lower `instance_match_iou`.

3. Over-merged nearby cars:
- Fix: lower `instance_match_iou`, increase `instance_min_point_votes`, or use feature-assisted DBSCAN baseline.

---

## 8. Parameter Groups for Paper Ablations

Recommended ablations:

1. Semantic voting robustness:
- `core_mode`: quality vs centroid vs mixed
- `vote_stride`
- `vote_ignore_ids`
- `min_top1_ratio`

2. Instance method comparison:
- `dbscan` vs `proj2d`
- `proj2d` with/without extra oblique mask views

3. proj2d merge policy:
- `instance_match_iou` sweep
- `instance_min_point_votes` sweep
- `min_instance_points` sweep

4. Representation ablation:
- KMeans on `ins_feat` only vs `ins_feat+xyz`

---

## 9. Reproducibility Checklist

For each experiment, persist:
- Git commit hash.
- Full CLI commands.
- `semantic_instance_report.json` / `semantic_models_seg3d_instance_report.json`.
- Input mask directory version (`language_features_instance` generation date).
- Camera source (`transforms` vs `colmap`).

---

## 10. Suggested Paper Section Mapping

Suggested mapping from this repository to paper sections:

1. Method: OpenGaussian multi-stage representation learning
- Stage0-Stage3 in `train.py`.

2. Post-hoc semantic partition and voting
- `cluster_semantic_kmeans.py` + semantic voting part in `semantic_instance_pipeline.py`.

3. seg3D-style projection instance reasoning
- `proj2d_instances(...)` and instance-only pipeline.

4. Experiments and Ablations
- Parameter sweeps in Section 8 and failure cases in Section 7.
