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
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
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

        row = 0
        train_grid.addWidget(QLabel("场景路径"), row, 0)
        train_grid.addWidget(self.source_path, row, 1, 1, 3)
        row += 1
        train_grid.addWidget(QLabel("输出路径"), row, 0)
        train_grid.addWidget(self.model_path, row, 1, 1, 3)
        row += 1
        train_grid.addWidget(QLabel("起始 checkpoint"), row, 0)
        train_grid.addWidget(self.start_checkpoint, row, 1, 1, 3)
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

        row = 0
        sam_grid.addWidget(self.run_sam3, row, 0)
        sam_grid.addWidget(self.run_training, row, 1)
        sam_grid.addWidget(self.sam3_export, row, 2, 1, 2)
        row += 1
        sam_grid.addWidget(QLabel("SAM3输入目录"), row, 0)
        sam_grid.addWidget(self.sam3_input_dir, row, 1, 1, 3)
        row += 1
        sam_grid.addWidget(QLabel("SAM3输出目录"), row, 0)
        sam_grid.addWidget(self.sam3_output_dir, row, 1, 1, 3)
        row += 1
        sam_grid.addWidget(QLabel("SAM3权重"), row, 0)
        sam_grid.addWidget(self.sam3_checkpoint, row, 1, 1, 2)
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
        self.post_input_ply = QLineEdit("")
        self.post_mask2inst_output_dir = QLineEdit("")
        self.post_mask2inst_ignore_ids = QLineEdit("0")
        self.post_mask2inst_min_area = QSpinBox()
        self.post_mask2inst_min_area.setRange(1, 1_000_000)
        self.post_mask2inst_min_area.setValue(20)
        self.post_instance_single_label = QLineEdit("")

        row = 0
        post_grid.addWidget(self.run_postprocess, row, 0)
        post_grid.addWidget(self.post_auto_sync_id2label, row, 1)
        post_grid.addWidget(self.run_post_mask2inst, row, 2)
        post_grid.addWidget(self.run_post_cluster, row, 3)
        row += 1
        post_grid.addWidget(self.run_post_semantic, row, 2)
        post_grid.addWidget(self.run_post_instance, row, 3)
        row += 1
        post_grid.addWidget(QLabel("后处理输入PLY"), row, 0)
        post_grid.addWidget(self.post_input_ply, row, 1, 1, 3)
        row += 1
        post_grid.addWidget(QLabel("mask2inst输出目录"), row, 0)
        post_grid.addWidget(self.post_mask2inst_output_dir, row, 1, 1, 3)
        row += 1
        post_grid.addWidget(QLabel("mask2inst忽略ID"), row, 0)
        post_grid.addWidget(self.post_mask2inst_ignore_ids, row, 1)
        post_grid.addWidget(QLabel("mask2inst最小面积"), row, 2)
        post_grid.addWidget(self.post_mask2inst_min_area, row, 3)
        row += 1
        post_grid.addWidget(QLabel("实例单标签"), row, 0)
        post_grid.addWidget(self.post_instance_single_label, row, 1, 1, 3)
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

        scroll.setWidget(body)
        root_layout.addWidget(scroll)
        self.setCentralWidget(root)

        self.btn_preview.clicked.connect(self.on_preview)
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_refresh.clicked.connect(self.on_manual_refresh)

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
            40000,  # post_iteration
            self.post_input_ply.text().strip(),
            8,      # post_n_clusters
            "",     # post_cluster_output_dir
            "",     # post_sem_output_dir
            "",     # post_instance_output_dir
            "0,7",  # post_vote_ignore_ids
            800,    # post_min_votes
            0.4,    # post_min_top1_ratio
            "0:background,1:vehicle,2:person,3:bicycle,4:vegetation,5:road,6:traffic_facility,7:other",
            "0,1,2,3,4,5,6,7,255",  # post_sem_stuff_ids
            "1",    # post_semantic_only_ids
            self.post_instance_single_label.text().strip(),
            "0,4,5,6,7,255",  # post_instance_stuff_ids
            "D:\\Scene_0320\\language_features_instance",
            "name",
            "_inst.npy",
            0,      # post_instance_mask_level
            "0",    # post_instance_mask_ignore_ids
            1,      # post_instance_vote_stride
            30,     # post_instance_min_mask_points
            0.2,    # post_instance_match_iou
            2,      # post_instance_min_point_votes
            180,    # post_min_instance_points
            True,   # post_save_instance_parts
        )

    def _apply_refresh(
        self, status: str, logs: str, progress: float, current_iter: int, monitor: str
    ) -> None:
        self.status_label.setText(status)
        self.log_text.setPlainText(logs)
        self.progress.setValue(max(0, min(1000, int(float(progress) * 10))))
        self.iter_label.setText(str(int(current_iter)))
        self.monitor_label.setText(monitor.replace("\n", " | "))

    def on_preview(self) -> None:
        try:
            cmd_txt = backend.preview_command(*self._collect_args())
            self.command_text.setPlainText(cmd_txt)
        except Exception as exc:
            QMessageBox.critical(self, "预览失败", str(exc))

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
        try:
            status, logs, progress, current_iter, monitor = backend.refresh_logs()
            self._apply_refresh(status, logs, progress, current_iter, monitor)
        except Exception as exc:
            QMessageBox.critical(self, "刷新失败", str(exc))

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
