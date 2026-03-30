#!/usr/bin/env python3
"""PySide6 desktop launcher for OpenGaussian Stage1.

This script reuses backend logic from scripts/train_ui.py:
- preview_command
- start_training
- stop_training
- refresh_logs

So behavior stays aligned with the existing Gradio launcher.
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Any, Tuple

try:
    from PySide6.QtCore import Qt, QTimer
    from PySide6.QtGui import QPixmap
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:
    raise SystemExit(
        "Missing dependency: PySide6\n"
        "Install with: pip install PySide6\n"
        f"Detail: {exc}"
    )

try:
    import train_ui as backend
except Exception as exc:
    raise SystemExit(f"Cannot import train_ui backend: {exc}")


class TrainQtWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("OpenGaussian Stage1 Trainer (PySide6)")
        self.resize(1280, 860)

        self._queue: queue.Queue[Tuple[str, Any]] = queue.Queue()
        self._start_thread: threading.Thread | None = None

        self._build_ui()

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(1200)
        self.refresh_timer.timeout.connect(self._on_tick)
        self.refresh_timer.start()

        self._loading_dialog: QDialog | None = None
        self._loading_label: QLabel | None = None
        self._loading_depth = 0

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        body_layout = QVBoxLayout(body)

        top_group = QGroupBox("基础参数")
        top_form = QFormLayout(top_group)
        self.python_exec = QLineEdit("python")
        self.path_mode = QComboBox()
        self.path_mode.addItem("auto", "auto")
        self.path_mode.addItem("linux", "linux")
        self.path_mode.addItem("windows", "windows")
        self.data_device = QComboBox()
        self.data_device.addItems(["cuda", "cpu"])
        self.resolution = QComboBox()
        self.resolution.addItems(["default", "1", "2", "4", "8"])
        top_form.addRow("Python", self.python_exec)
        top_form.addRow("路径模式", self.path_mode)
        top_form.addRow("数据设备", self.data_device)
        top_form.addRow("分辨率倍率", self.resolution)
        body_layout.addWidget(top_group)

        train_group = QGroupBox("训练参数")
        train_grid = QGridLayout(train_group)
        self.source_path = QLineEdit("D:\\Scene_0320")
        self.model_path = QLineEdit("D:\\Scene_0320\\test0323")
        self.start_checkpoint = QLineEdit("D:\\Scene_0320\\chkpnt30000_from_point_cloud_clip_aligned.pth")

        self.iterations = QSpinBox()
        self.iterations.setRange(1, 10_000_000)
        self.iterations.setValue(40000)
        self.start_ins_feat_iter = QSpinBox()
        self.start_ins_feat_iter.setRange(0, 10_000_000)
        self.start_ins_feat_iter.setValue(30000)
        self.sam_level = QSpinBox()
        self.sam_level.setRange(0, 10)
        self.sam_level.setValue(0)
        self.checkpoint_max_points = QSpinBox()
        self.checkpoint_max_points.setRange(0, 100_000_000)
        self.checkpoint_max_points.setValue(0)
        self.save_iterations = QLineEdit("40000")
        self.checkpoint_iterations = QLineEdit("40000")
        self.eval_mode = QCheckBox("eval")
        self.save_memory = QCheckBox("save_memory")
        self.btn_pick_source_path = QPushButton("浏览...")
        self.btn_pick_model_path = QPushButton("浏览...")
        self.btn_pick_start_checkpoint = QPushButton("选择文件...")

        row = 0
        train_grid.addWidget(QLabel("场景路径"), row, 0)
        train_grid.addWidget(self.source_path, row, 1, 1, 2)
        train_grid.addWidget(self.btn_pick_source_path, row, 3)
        row += 1
        train_grid.addWidget(QLabel("输出路径"), row, 0)
        train_grid.addWidget(self.model_path, row, 1, 1, 2)
        train_grid.addWidget(self.btn_pick_model_path, row, 3)
        row += 1
        train_grid.addWidget(QLabel("起始 checkpoint"), row, 0)
        train_grid.addWidget(self.start_checkpoint, row, 1, 1, 2)
        train_grid.addWidget(self.btn_pick_start_checkpoint, row, 3)
        row += 1
        train_grid.addWidget(QLabel("iterations"), row, 0)
        train_grid.addWidget(self.iterations, row, 1)
        train_grid.addWidget(QLabel("start_ins_feat_iter"), row, 2)
        train_grid.addWidget(self.start_ins_feat_iter, row, 3)
        row += 1
        train_grid.addWidget(QLabel("sam_level"), row, 0)
        train_grid.addWidget(self.sam_level, row, 1)
        train_grid.addWidget(QLabel("ckpt点数上限"), row, 2)
        train_grid.addWidget(self.checkpoint_max_points, row, 3)
        row += 1
        train_grid.addWidget(QLabel("save_iterations"), row, 0)
        train_grid.addWidget(self.save_iterations, row, 1)
        train_grid.addWidget(QLabel("checkpoint_iterations"), row, 2)
        train_grid.addWidget(self.checkpoint_iterations, row, 3)
        row += 1
        train_grid.addWidget(self.eval_mode, row, 1)
        train_grid.addWidget(self.save_memory, row, 2)

        body_layout.addWidget(train_group)

        sam_group = QGroupBox("SAM3（可选）")
        sam_grid = QGridLayout(sam_group)
        self.run_sam3 = QCheckBox("跑 SAM3")
        self.run_training = QCheckBox("跑训练")
        self.run_training.setChecked(True)
        self.sam3_export = QCheckBox("SAM3结果转 language_features")
        self.sam3_export.setChecked(True)
        self.sam3_input_dir = QLineEdit("D:\\Scene_0320\\images_8")
        self.sam3_output_dir = QLineEdit("")
        self.sam3_checkpoint = QLineEdit(str(backend.SAM3_DEFAULT_CKPT))
        self.sam3_device = QComboBox()
        self.sam3_device.addItems(["cuda", "cpu"])
        self.btn_pick_sam3_input_dir = QPushButton("浏览...")
        self.btn_pick_sam3_output_dir = QPushButton("浏览...")
        self.btn_pick_sam3_checkpoint = QPushButton("选择文件...")

        row = 0
        sam_grid.addWidget(self.run_sam3, row, 0)
        sam_grid.addWidget(self.run_training, row, 1)
        sam_grid.addWidget(self.sam3_export, row, 2, 1, 2)
        row += 1
        sam_grid.addWidget(QLabel("SAM3输入目录"), row, 0)
        sam_grid.addWidget(self.sam3_input_dir, row, 1, 1, 2)
        sam_grid.addWidget(self.btn_pick_sam3_input_dir, row, 3)
        row += 1
        sam_grid.addWidget(QLabel("SAM3输出目录"), row, 0)
        sam_grid.addWidget(self.sam3_output_dir, row, 1, 1, 2)
        sam_grid.addWidget(self.btn_pick_sam3_output_dir, row, 3)
        row += 1
        sam_grid.addWidget(QLabel("SAM3权重"), row, 0)
        sam_grid.addWidget(self.sam3_checkpoint, row, 1, 1, 1)
        sam_grid.addWidget(self.btn_pick_sam3_checkpoint, row, 2)
        sam_grid.addWidget(self.sam3_device, row, 3)
        body_layout.addWidget(sam_group)

        post_group = QGroupBox("后处理（可选）")
        post_grid = QGridLayout(post_group)
        self.run_postprocess = QCheckBox("跑后处理")
        self.post_auto_sync_id2label = QCheckBox("自动同步 id2label")
        self.post_auto_sync_id2label.setChecked(True)
        self.run_post_mask2inst = QCheckBox("语义mask转实例mask")
        self.run_post_mask2inst.setChecked(True)
        self.run_post_cluster = QCheckBox("聚类")
        self.run_post_cluster.setChecked(True)
        self.run_post_semantic = QCheckBox("语义聚合")
        self.run_post_semantic.setChecked(True)
        self.run_post_instance = QCheckBox("实例分割")
        self.run_post_instance.setChecked(True)
        self.post_iteration = QSpinBox()
        self.post_iteration.setRange(1, 10_000_000)
        self.post_iteration.setValue(40000)
        self.post_n_clusters = QSpinBox()
        self.post_n_clusters.setRange(1, 4096)
        self.post_n_clusters.setValue(8)
        self.post_vote_ignore_ids = QLineEdit("0,7")
        self.post_min_votes = QSpinBox()
        self.post_min_votes.setRange(0, 10_000_000)
        self.post_min_votes.setValue(800)
        self.post_min_top1_ratio = QDoubleSpinBox()
        self.post_min_top1_ratio.setRange(0.0, 1.0)
        self.post_min_top1_ratio.setSingleStep(0.05)
        self.post_min_top1_ratio.setValue(0.4)
        self.post_input_ply = QLineEdit("")
        self.post_cluster_output_dir = QLineEdit("")
        self.post_sem_output_dir = QLineEdit("")
        self.post_instance_output_dir = QLineEdit("")
        self.post_mask2inst_output_dir = QLineEdit("")
        self.post_mask2inst_ignore_ids = QLineEdit("0")
        self.post_mask2inst_min_area = QSpinBox()
        self.post_mask2inst_min_area.setRange(1, 1_000_000)
        self.post_mask2inst_min_area.setValue(20)
        self.post_sem_stuff_ids = QLineEdit("0,1,2,3,4,5,6,7,255")
        self.post_semantic_only_ids = QLineEdit("1")
        self.post_instance_stuff_ids = QLineEdit("0,4,5,6,7,255")
        self.post_instance_single_label = QLineEdit("")
        self.post_instance_mask_dir = QLineEdit("")
        self.post_instance_mask_mode = QComboBox()
        self.post_instance_mask_mode.addItems(["name", "index"])
        self.post_instance_mask_suffix = QLineEdit("_inst.npy")
        self.post_instance_mask_level = QSpinBox()
        self.post_instance_mask_level.setRange(0, 10)
        self.post_instance_mask_level.setValue(0)
        self.post_instance_mask_ignore_ids = QLineEdit("0")
        self.post_instance_vote_stride = QSpinBox()
        self.post_instance_vote_stride.setRange(1, 9999)
        self.post_instance_vote_stride.setValue(1)
        self.post_instance_min_mask_points = QSpinBox()
        self.post_instance_min_mask_points.setRange(1, 10_000_000)
        self.post_instance_min_mask_points.setValue(30)
        self.post_instance_match_iou = QDoubleSpinBox()
        self.post_instance_match_iou.setRange(0.0, 1.0)
        self.post_instance_match_iou.setSingleStep(0.05)
        self.post_instance_match_iou.setValue(0.2)
        self.post_instance_min_point_votes = QSpinBox()
        self.post_instance_min_point_votes.setRange(1, 10_000_000)
        self.post_instance_min_point_votes.setValue(2)
        self.post_min_instance_points = QSpinBox()
        self.post_min_instance_points.setRange(1, 10_000_000)
        self.post_min_instance_points.setValue(180)
        self.post_save_instance_parts = QCheckBox("保存每个实例子PLY")
        self.post_save_instance_parts.setChecked(True)
        self.post_id2label = QLineEdit(
            "0:background,1:vehicle,2:person,3:bicycle,4:vegetation,5:road,6:traffic_facility,7:other"
        )
        self.btn_load_id2label = QPushButton("从mask读取")
        self._set_small_path_button(self.btn_load_id2label)
        self.post_id2label_status = QLabel("可手动编辑，也可从 language_features 自动读取")
        self.btn_pick_post_input_ply = QPushButton("选择文件...")
        self.btn_pick_cluster_output_dir = QPushButton("浏览...")
        self.btn_pick_sem_output_dir = QPushButton("浏览...")
        self.btn_pick_instance_output_dir = QPushButton("浏览...")
        self.btn_pick_mask2inst_out_dir = QPushButton("浏览...")
        self.btn_pick_instance_mask_dir = QPushButton("浏览...")
        self._set_small_path_button(self.btn_pick_source_path)
        self._set_small_path_button(self.btn_pick_model_path)
        self._set_small_path_button(self.btn_pick_start_checkpoint)
        self._set_small_path_button(self.btn_pick_sam3_input_dir)
        self._set_small_path_button(self.btn_pick_sam3_output_dir)
        self._set_small_path_button(self.btn_pick_sam3_checkpoint)
        self._set_small_path_button(self.btn_pick_post_input_ply)
        self._set_small_path_button(self.btn_pick_cluster_output_dir)
        self._set_small_path_button(self.btn_pick_sem_output_dir)
        self._set_small_path_button(self.btn_pick_instance_output_dir)
        self._set_small_path_button(self.btn_pick_mask2inst_out_dir)
        self._set_small_path_button(self.btn_pick_instance_mask_dir)

        row = 0
        post_grid.addWidget(self.run_postprocess, row, 0)
        post_grid.addWidget(self.run_post_mask2inst, row, 1)
        post_grid.addWidget(self.run_post_cluster, row, 2)
        post_grid.addWidget(self.run_post_semantic, row, 3)
        row += 1
        post_grid.addWidget(self.run_post_instance, row, 1)
        row += 1
        post_grid.addWidget(QLabel("后处理迭代步"), row, 0)
        post_grid.addWidget(self.post_iteration, row, 1)
        post_grid.addWidget(QLabel("聚类数K"), row, 2)
        post_grid.addWidget(self.post_n_clusters, row, 3)
        row += 1
        post_grid.addWidget(QLabel("后处理输入PLY"), row, 0)
        post_grid.addWidget(self.post_input_ply, row, 1, 1, 2)
        post_grid.addWidget(self.btn_pick_post_input_ply, row, 3)
        row += 1
        post_grid.addWidget(QLabel("聚类输出目录"), row, 0)
        post_grid.addWidget(self.post_cluster_output_dir, row, 1, 1, 2)
        post_grid.addWidget(self.btn_pick_cluster_output_dir, row, 3)
        row += 1
        post_grid.addWidget(QLabel("语义输出目录"), row, 0)
        post_grid.addWidget(self.post_sem_output_dir, row, 1, 1, 2)
        post_grid.addWidget(self.btn_pick_sem_output_dir, row, 3)
        row += 1
        post_grid.addWidget(QLabel("实例输出目录"), row, 0)
        post_grid.addWidget(self.post_instance_output_dir, row, 1, 1, 2)
        post_grid.addWidget(self.btn_pick_instance_output_dir, row, 3)
        row += 1
        post_grid.addWidget(QLabel("语义投票忽略ID"), row, 0)
        post_grid.addWidget(self.post_vote_ignore_ids, row, 1)
        post_grid.addWidget(QLabel("语义最小票数"), row, 2)
        post_grid.addWidget(self.post_min_votes, row, 3)
        row += 1
        post_grid.addWidget(QLabel("语义Top1阈值"), row, 0)
        post_grid.addWidget(self.post_min_top1_ratio, row, 1)
        post_grid.addWidget(self.post_auto_sync_id2label, row, 2, 1, 2)
        row += 1
        post_grid.addWidget(QLabel("mask2inst输出目录"), row, 0)
        post_grid.addWidget(self.post_mask2inst_output_dir, row, 1, 1, 2)
        post_grid.addWidget(self.btn_pick_mask2inst_out_dir, row, 3)
        row += 1
        post_grid.addWidget(QLabel("mask2inst忽略ID"), row, 0)
        post_grid.addWidget(self.post_mask2inst_ignore_ids, row, 1)
        post_grid.addWidget(QLabel("mask2inst最小面积"), row, 2)
        post_grid.addWidget(self.post_mask2inst_min_area, row, 3)
        row += 1
        post_grid.addWidget(QLabel("id2label"), row, 0)
        post_grid.addWidget(self.post_id2label, row, 1, 1, 2)
        post_grid.addWidget(self.btn_load_id2label, row, 3)
        row += 1
        post_grid.addWidget(QLabel("读取状态"), row, 0)
        post_grid.addWidget(self.post_id2label_status, row, 1, 1, 3)
        row += 1
        post_grid.addWidget(QLabel("语义stuff_ids"), row, 0)
        post_grid.addWidget(self.post_sem_stuff_ids, row, 1, 1, 3)
        row += 1
        post_grid.addWidget(QLabel("实例仅处理语义ID"), row, 0)
        post_grid.addWidget(self.post_semantic_only_ids, row, 1)
        post_grid.addWidget(QLabel("实例阶段stuff_ids"), row, 2)
        post_grid.addWidget(self.post_instance_stuff_ids, row, 3)
        row += 1
        post_grid.addWidget(QLabel("实例单标签"), row, 0)
        post_grid.addWidget(self.post_instance_single_label, row, 1, 1, 3)
        row += 1
        post_grid.addWidget(QLabel("实例mask目录"), row, 0)
        post_grid.addWidget(self.post_instance_mask_dir, row, 1, 1, 2)
        post_grid.addWidget(self.btn_pick_instance_mask_dir, row, 3)
        row += 1
        post_grid.addWidget(QLabel("实例mask模式"), row, 0)
        post_grid.addWidget(self.post_instance_mask_mode, row, 1)
        post_grid.addWidget(QLabel("实例mask后缀"), row, 2)
        post_grid.addWidget(self.post_instance_mask_suffix, row, 3)
        row += 1
        post_grid.addWidget(QLabel("实例mask层级"), row, 0)
        post_grid.addWidget(self.post_instance_mask_level, row, 1)
        post_grid.addWidget(QLabel("实例mask忽略ID"), row, 2)
        post_grid.addWidget(self.post_instance_mask_ignore_ids, row, 3)
        row += 1
        post_grid.addWidget(QLabel("实例投票步长"), row, 0)
        post_grid.addWidget(self.post_instance_vote_stride, row, 1)
        post_grid.addWidget(QLabel("实例最小mask点数"), row, 2)
        post_grid.addWidget(self.post_instance_min_mask_points, row, 3)
        row += 1
        post_grid.addWidget(QLabel("实例匹配IoU"), row, 0)
        post_grid.addWidget(self.post_instance_match_iou, row, 1)
        post_grid.addWidget(QLabel("实例最小点票数"), row, 2)
        post_grid.addWidget(self.post_instance_min_point_votes, row, 3)
        row += 1
        post_grid.addWidget(QLabel("实例最小点数"), row, 0)
        post_grid.addWidget(self.post_min_instance_points, row, 1)
        post_grid.addWidget(self.post_save_instance_parts, row, 2, 1, 2)
        body_layout.addWidget(post_group)

        action_group = QGroupBox("操作")
        action_layout = QHBoxLayout(action_group)
        self.btn_preview = QPushButton("预览命令")
        self.btn_start = QPushButton("开始")
        self.btn_stop = QPushButton("停止")
        self.btn_refresh = QPushButton("手动刷新")
        action_layout.addWidget(self.btn_preview)
        action_layout.addWidget(self.btn_start)
        action_layout.addWidget(self.btn_stop)
        action_layout.addWidget(self.btn_refresh)
        body_layout.addWidget(action_group)

        status_group = QGroupBox("状态")
        status_layout = QGridLayout(status_group)
        self.status_label = QLabel("空闲")
        self.iter_label = QLabel("0")
        self.monitor_label = QLabel("-")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)  # 0.1% granularity
        self.progress.setValue(0)
        status_layout.addWidget(QLabel("状态"), 0, 0)
        status_layout.addWidget(self.status_label, 0, 1)
        status_layout.addWidget(QLabel("当前迭代"), 0, 2)
        status_layout.addWidget(self.iter_label, 0, 3)
        status_layout.addWidget(QLabel("资源监控"), 1, 0)
        status_layout.addWidget(self.monitor_label, 1, 1, 1, 3)
        status_layout.addWidget(QLabel("训练进度"), 2, 0)
        status_layout.addWidget(self.progress, 2, 1, 1, 3)
        body_layout.addWidget(status_group)

        cmd_group = QGroupBox("命令预览")
        cmd_layout = QVBoxLayout(cmd_group)
        self.command_text = QPlainTextEdit()
        self.command_text.setReadOnly(True)
        cmd_layout.addWidget(self.command_text)
        body_layout.addWidget(cmd_group)

        log_group = QGroupBox("日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        body_layout.addWidget(log_group)

        feat_group = QGroupBox("Stage1 特征渲染对比")
        feat_layout = QVBoxLayout(feat_group)
        feat_img_row = QHBoxLayout()
        self.stage1_gt_preview = QLabel("GT 预览（点击刷新）")
        self.stage1_gt_preview.setAlignment(Qt.AlignCenter)
        self.stage1_gt_preview.setMinimumSize(420, 280)
        self.stage1_gt_preview.setStyleSheet("border: 1px solid #D4D7DE; background: #F7F9FC;")
        self.stage1_ins_feat_preview = QLabel("INS_FEAT 预览（点击刷新）")
        self.stage1_ins_feat_preview.setAlignment(Qt.AlignCenter)
        self.stage1_ins_feat_preview.setMinimumSize(420, 280)
        self.stage1_ins_feat_preview.setStyleSheet("border: 1px solid #D4D7DE; background: #F7F9FC;")
        feat_img_row.addWidget(self.stage1_gt_preview)
        feat_img_row.addWidget(self.stage1_ins_feat_preview)
        feat_layout.addLayout(feat_img_row)
        self.stage1_ins_feat_status = QLabel("点击“刷新特征图”加载最新 GT/INS_FEAT 对比")
        self.btn_refresh_ins_feat = QPushButton("刷新特征图")
        self._set_small_path_button(self.btn_refresh_ins_feat)
        feat_layout.addWidget(self.stage1_ins_feat_status)
        feat_layout.addWidget(self.btn_refresh_ins_feat, alignment=Qt.AlignLeft)
        body_layout.addWidget(feat_group)

        scroll.setWidget(body)
        root_layout.addWidget(scroll)
        self.setCentralWidget(root)

        self.btn_preview.clicked.connect(self.on_preview)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_refresh.clicked.connect(self.on_manual_refresh)
        self.btn_pick_source_path.clicked.connect(
            lambda: self._pick_directory(self.source_path, "选择场景目录")
        )
        self.btn_pick_model_path.clicked.connect(
            lambda: self._pick_directory(self.model_path, "选择输出目录")
        )
        self.btn_pick_start_checkpoint.clicked.connect(
            lambda: self._pick_file(self.start_checkpoint, "选择起始 checkpoint", "PyTorch Checkpoint (*.pth);;All Files (*)")
        )
        self.btn_pick_sam3_input_dir.clicked.connect(
            lambda: self._pick_directory(self.sam3_input_dir, "选择 SAM3 输入目录")
        )
        self.btn_pick_sam3_output_dir.clicked.connect(
            lambda: self._pick_directory(self.sam3_output_dir, "选择 SAM3 输出目录")
        )
        self.btn_pick_sam3_checkpoint.clicked.connect(
            lambda: self._pick_file(self.sam3_checkpoint, "选择 SAM3 权重", "Checkpoint (*.pt *.pth);;All Files (*)")
        )
        self.btn_pick_post_input_ply.clicked.connect(
            lambda: self._pick_file(self.post_input_ply, "选择后处理输入 PLY", "PLY Files (*.ply);;All Files (*)")
        )
        self.btn_pick_cluster_output_dir.clicked.connect(
            lambda: self._pick_directory(self.post_cluster_output_dir, "选择聚类输出目录")
        )
        self.btn_pick_sem_output_dir.clicked.connect(
            lambda: self._pick_directory(self.post_sem_output_dir, "选择语义输出目录")
        )
        self.btn_pick_instance_output_dir.clicked.connect(
            lambda: self._pick_directory(self.post_instance_output_dir, "选择实例输出目录")
        )
        self.btn_pick_mask2inst_out_dir.clicked.connect(
            lambda: self._pick_directory(self.post_mask2inst_output_dir, "选择 mask2inst 输出目录")
        )
        self.btn_pick_instance_mask_dir.clicked.connect(
            lambda: self._pick_directory(self.post_instance_mask_dir, "选择实例mask目录")
        )
        self.btn_refresh_ins_feat.clicked.connect(self.on_refresh_ins_feat)
        self.btn_load_id2label.clicked.connect(self.on_load_id2label)

        self.setStyleSheet(
            """
            QWidget { font-size: 13px; }
            QGroupBox {
                font-weight: 600;
                border: 1px solid #D4D7DE;
                border-radius: 8px;
                margin-top: 10px;
                padding-top: 12px;
                background: #FAFBFD;
            }
            QPlainTextEdit {
                background: #0F1720;
                color: #D7E0EA;
                border-radius: 6px;
                border: 1px solid #2A3645;
                font-family: Consolas, Menlo, monospace;
            }
            QPushButton {
                background: #1F6FEB;
                color: white;
                border: none;
                border-radius: 6px;
                padding: 7px 12px;
                font-weight: 600;
            }
            QPushButton:disabled { background: #8AA2C8; color: #E8EDF5; }
            """
        )

    def _current_path_mode(self) -> str:
        return self.path_mode.currentData() or "auto"

    def _set_small_path_button(self, btn: QPushButton) -> None:
        btn.setFixedWidth(84)
        btn.setFixedHeight(26)

    def _pick_directory(self, target: QLineEdit, title: str) -> None:
        start = target.text().strip() or "."
        path = QFileDialog.getExistingDirectory(self, title, start)
        if path:
            target.setText(path)

    def _pick_file(self, target: QLineEdit, title: str, file_filter: str) -> None:
        start = target.text().strip() or "."
        path, _ = QFileDialog.getOpenFileName(self, title, start, file_filter)
        if path:
            target.setText(path)

    def _collect_args(self) -> Tuple[Any, ...]:
        run_post = bool(self.run_postprocess.isChecked())
        return (
            self.python_exec.text().strip(),
            self._current_path_mode(),
            self.source_path.text().strip(),
            self.model_path.text().strip(),
            self.start_checkpoint.text().strip(),
            int(self.iterations.value()),
            int(self.start_ins_feat_iter.value()),
            int(self.sam_level.value()),
            self.save_iterations.text().strip(),
            self.checkpoint_iterations.text().strip(),
            self.resolution.currentText().strip(),
            int(self.checkpoint_max_points.value()),
            self.data_device.currentText().strip(),
            bool(self.eval_mode.isChecked()),
            bool(self.save_memory.isChecked()),
            bool(self.run_sam3.isChecked()),
            bool(self.run_training.isChecked()),
            self.sam3_input_dir.text().strip(),
            self.sam3_output_dir.text().strip(),
            self.sam3_checkpoint.text().strip(),
            self.sam3_device.currentText().strip(),
            bool(self.sam3_export.isChecked()),
            bool(self.post_auto_sync_id2label.isChecked()),
            run_post,  # run_postprocess
            bool(self.run_post_mask2inst.isChecked()),
            self.post_mask2inst_output_dir.text().strip(),
            int(self.post_mask2inst_min_area.value()),
            self.post_mask2inst_ignore_ids.text().strip(),
            bool(self.run_post_cluster.isChecked()),
            bool(self.run_post_semantic.isChecked()),
            bool(self.run_post_instance.isChecked()),
            int(self.post_iteration.value()),
            self.post_input_ply.text().strip(),
            int(self.post_n_clusters.value()),
            self.post_cluster_output_dir.text().strip(),
            self.post_sem_output_dir.text().strip(),
            self.post_instance_output_dir.text().strip(),
            self.post_vote_ignore_ids.text().strip(),
            int(self.post_min_votes.value()),
            float(self.post_min_top1_ratio.value()),
            self.post_id2label.text().strip(),
            self.post_sem_stuff_ids.text().strip(),
            self.post_semantic_only_ids.text().strip(),
            self.post_instance_single_label.text().strip(),
            self.post_instance_stuff_ids.text().strip(),
            self.post_instance_mask_dir.text().strip(),
            self.post_instance_mask_mode.currentText().strip(),
            self.post_instance_mask_suffix.text().strip(),
            int(self.post_instance_mask_level.value()),
            self.post_instance_mask_ignore_ids.text().strip(),
            int(self.post_instance_vote_stride.value()),
            int(self.post_instance_min_mask_points.value()),
            float(self.post_instance_match_iou.value()),
            int(self.post_instance_min_point_votes.value()),
            int(self.post_min_instance_points.value()),
            bool(self.post_save_instance_parts.isChecked()),
        )

    def _show_loading(self, text: str) -> None:
        return

    def _hide_loading(self) -> None:
        return

    def _begin_loading(self, text: str) -> None:
        return

    def _end_loading(self) -> None:
        return

    def _run_async_with_loading(self, name: str, text: str, func) -> None:
        def _worker() -> None:
            try:
                ret = func()
                self._queue.put((f"{name}_ok", ret))
            except Exception as exc:
                self._queue.put((f"{name}_err", exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _set_image_preview(self, label: QLabel, img_path: str | None, empty_text: str) -> None:
        if not img_path:
            label.setPixmap(QPixmap())
            label.setText(empty_text)
            return
        pix = QPixmap(img_path)
        if pix.isNull():
            label.setPixmap(QPixmap())
            label.setText(empty_text)
            return
        scaled = pix.scaled(
            label.width(),
            label.height(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        label.setText("")
        label.setPixmap(scaled)

    def _set_text_preserve_scroll(self, widget: QPlainTextEdit, text: str) -> None:
        bar = widget.verticalScrollBar()
        old_max = bar.maximum()
        old_val = bar.value()
        at_bottom = old_val >= max(0, old_max - 2)
        widget.setPlainText(text)
        new_max = bar.maximum()
        if at_bottom:
            bar.setValue(new_max)
            return
        if old_max <= 0:
            bar.setValue(old_val)
            return
        ratio = float(old_val) / float(old_max)
        bar.setValue(int(round(ratio * new_max)))

    def _apply_refresh(
        self, status: str, logs: str, progress: float, current_iter: int, monitor: str
    ) -> None:
        self.status_label.setText(status)
        self._set_text_preserve_scroll(self.log_text, logs)
        self.progress.setValue(max(0, min(1000, int(float(progress) * 10))))
        self.iter_label.setText(str(int(current_iter)))
        self.monitor_label.setText(monitor.replace("\n", " | "))

    def on_preview(self) -> None:
        args = self._collect_args()
        self._run_async_with_loading(
            "preview",
            "正在生成命令预览...",
            lambda: backend.preview_command(*args),
        )

    def on_start(self) -> None:
        if self._start_thread is not None and self._start_thread.is_alive():
            return

        self.btn_start.setEnabled(False)

        def _worker() -> None:
            try:
                ret = backend.start_training(*self._collect_args())
                self._queue.put(("start_ok", ret))
            except Exception as exc:
                self._queue.put(("start_err", exc))

        self._start_thread = threading.Thread(target=_worker, daemon=True)
        self._start_thread.start()

    def on_stop(self) -> None:
        try:
            status, logs, progress, current_iter, monitor = backend.stop_training()
            self._apply_refresh(status, logs, progress, current_iter, monitor)
        except Exception as exc:
            QMessageBox.critical(self, "停止失败", str(exc))

    def on_manual_refresh(self) -> None:
        self._run_async_with_loading(
            "manual_refresh",
            "正在刷新日志与状态...",
            backend.refresh_logs,
        )

    def on_refresh_ins_feat(self) -> None:
        path_mode = self._current_path_mode()
        model_path = self.model_path.text().strip()
        self._run_async_with_loading(
            "refresh_ins_feat",
            "正在读取特征图预览...",
            lambda: backend.refresh_stage1_ins_feat_preview(path_mode, model_path),
        )

    def on_load_id2label(self) -> None:
        path_mode = self._current_path_mode()
        source_path = self.source_path.text().strip()
        self._run_async_with_loading(
            "load_id2label",
            "正在从 mask 读取 id2label...",
            lambda: backend.detect_id2label_from_language_features(path_mode, source_path),
        )

    def _on_tick(self) -> None:
        while True:
            try:
                kind, data = self._queue.get_nowait()
            except queue.Empty:
                break
            if kind == "start_ok":
                status, logs, cmd, progress, current_iter, monitor = data
                self.command_text.setPlainText(cmd)
                self._apply_refresh(status, logs, progress, current_iter, monitor)
                self.btn_start.setEnabled(True)
            elif kind == "start_err":
                self.btn_start.setEnabled(True)
                QMessageBox.critical(self, "启动失败", str(data))
            elif kind == "preview_ok":
                self._end_loading()
                self._set_text_preserve_scroll(self.command_text, str(data))
            elif kind == "preview_err":
                self._end_loading()
                QMessageBox.critical(self, "预览失败", str(data))
            elif kind == "manual_refresh_ok":
                self._end_loading()
                status, logs, progress, current_iter, monitor = data
                self._apply_refresh(status, logs, progress, current_iter, monitor)
            elif kind == "manual_refresh_err":
                self._end_loading()
                QMessageBox.critical(self, "刷新失败", str(data))
            elif kind == "refresh_ins_feat_ok":
                self._end_loading()
                gt_path, ins_path, status = data
                self._set_image_preview(self.stage1_gt_preview, gt_path, "GT 预览不可用")
                self._set_image_preview(self.stage1_ins_feat_preview, ins_path, "INS_FEAT 预览不可用")
                self.stage1_ins_feat_status.setText(str(status))
            elif kind == "refresh_ins_feat_err":
                self._end_loading()
                self.stage1_ins_feat_status.setText(f"刷新失败: {data}")
            elif kind == "load_id2label_ok":
                self._end_loading()
                auto_text = data
                if auto_text:
                    self.post_id2label.setText(str(auto_text))
                    self.post_id2label_status.setText("已从 language_features 读取并更新")
                else:
                    self.post_id2label_status.setText("未读取到有效映射，保留当前值")
            elif kind == "load_id2label_err":
                self._end_loading()
                self.post_id2label_status.setText(f"读取失败: {data}")

        try:
            status, logs, progress, current_iter, monitor = backend.refresh_logs()
            self._apply_refresh(status, logs, progress, current_iter, monitor)
        except Exception:
            pass


def main() -> None:
    app = QApplication(sys.argv)
    win = TrainQtWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
