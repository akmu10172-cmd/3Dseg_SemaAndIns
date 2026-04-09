#!/usr/bin/env python3
"""Standalone workflow hub UI (no dependency on train_qt runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from PySide6.QtCore import QSize, Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMainWindow,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSpinBox,
        QStackedWidget,
        QStyle,
        QVBoxLayout,
        QWidget,
    )
except Exception as exc:
    raise SystemExit(
        "Missing dependency: PySide6\n"
        "Install with: pip install PySide6\n"
        f"Detail: {exc}"
    )


TASK_ITEMS: list[tuple[str, str, QStyle.StandardPixmap, str]] = [
    ("classic", "\u7ecf\u5178\u5168\u6d41\u7a0b", QStyle.SP_ComputerIcon, "\u4ece Stage1 \u8bad\u7ec3\u5230 SAM3/\u540e\u5904\u7406\u7684\u5b8c\u6574\u6d41\u7a0b\u9762\u677f\u3002"),
    ("sam3", "SAM3", QStyle.SP_DriveNetIcon, "\u805a\u7126 SAM3 \u76f8\u5173\u53c2\u6570\u4e0e\u8f93\u5165\u8f93\u51fa\u7ba1\u7406\u3002"),
    ("feature_train", "\u7279\u5f81\u8bad\u7ec3", QStyle.SP_FileDialogDetailedView, "\u7528\u4e8e Stage1 \u7279\u5f81\u8bad\u7ec3\u7684\u8f7b\u91cf\u5316\u4e13\u7528\u754c\u9762\u3002"),
    ("cluster", "\u805a\u7c7b", QStyle.SP_DirOpenIcon, "\u914d\u7f6e\u7279\u5f81\u805a\u7c7b\u53c2\u6570\u4e0e\u7ed3\u679c\u8f93\u51fa\u3002"),
    ("post_v1", "\u540e\u5904\u7406 V1", QStyle.SP_DialogApplyButton, "\u89c4\u5219\u5f0f\u540e\u5904\u7406\u7248\u672c\uff0c\u9002\u5408\u5feb\u901f\u9a8c\u8bc1\u3002"),
    ("post_v2", "\u540e\u5904\u7406 V2", QStyle.SP_DialogSaveButton, "\u589e\u5f3a\u578b\u540e\u5904\u7406\u7248\u672c\uff0c\u9884\u7559\u66f4\u591a\u7ec4\u5408\u7b56\u7565\u3002"),
]

ICON_DIR = Path(__file__).resolve().parent.parent / "assets" / "ui_icons"


def make_path_field(default_text: str = "", button_text: str = "\u6d4f\u89c8...") -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(8)
    line = QLineEdit(default_text)
    btn = QPushButton(button_text)
    btn.setMinimumWidth(92)
    layout.addWidget(line, 1)
    layout.addWidget(btn, 0)
    return row


class TaskTile(QFrame):
    def __init__(self, text: str, desc: str, icon: QIcon, on_click) -> None:
        super().__init__()
        self._desc_text = desc
        self._on_click = on_click
        self.setObjectName("taskTile")
        self.setCursor(Qt.PointingHandCursor)
        self.setProperty("hovered", False)
        self.setFixedWidth(220)
        self.setFixedHeight(332)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 10)
        layout.setSpacing(8)

        layout.addStretch(0)
        self.icon_label = QLabel()
        self.icon_label.setObjectName("taskIcon")
        self.icon_label.setAlignment(Qt.AlignCenter)
        self.icon_label.setPixmap(icon.pixmap(QSize(82, 82)))
        layout.addWidget(self.icon_label)

        self.title_label = QLabel(text)
        self.title_label.setObjectName("taskTitle")
        self.title_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        layout.addWidget(self.title_label)

        self.desc_label = QLabel("")
        self.desc_label.setObjectName("taskDesc")
        self.desc_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.desc_label.setWordWrap(True)
        self.desc_label.setFixedHeight(46)
        layout.addWidget(self.desc_label)
        layout.addStretch(0)

        # Let parent tile receive mouse events even when clicking inner labels.
        self.icon_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.title_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.desc_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self.setProperty("hovered", True)
        self.style().unpolish(self)
        self.style().polish(self)
        self.desc_label.setText(self._desc_text)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self.setProperty("hovered", False)
        self.style().unpolish(self)
        self.style().polish(self)
        self.desc_label.setText("")
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton and self.rect().contains(event.position().toPoint()):
            self._on_click()
        super().mouseReleaseEvent(event)


class MainPage(QWidget):
    def __init__(self, on_open_panel) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(22, 0, 22, 0)
        root.setSpacing(1)

        intro = QLabel(
            "\u672c\u9879\u76ee\u7528\u4e8e 3D \u8bed\u4e49/\u5b9e\u4f8b\u5206\u5272\u5168\u6d41\u7a0b\u8bd5\u9a8c\uff0c"
            "\u6db5\u76d6 Stage1 \u7279\u5f81\u8bad\u7ec3\u3001SAM3 \u8f85\u52a9\u5206\u5272\u3001\u805a\u7c7b\u4e0e\u540e\u5904\u7406\u6a21\u5757\u3002"
        )
        intro.setObjectName("projectIntro")
        intro.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        intro.setWordWrap(True)
        root.addWidget(intro)

        title = QLabel("\u4f60\u60f3\u505a\u4ec0\u4e48\uff1f")
        title.setObjectName("mainTitle")
        title.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        root.addWidget(title)

        row = QHBoxLayout()
        row.setSpacing(12)
        row.addStretch(1)
        for key, text, icon_type, desc in TASK_ITEMS:
            tile = TaskTile(
                text=text,
                desc=desc,
                icon=self._resolve_task_icon(key, icon_type),
                on_click=(lambda k=key: on_open_panel(k)),
            )
            row.addWidget(tile, 0)
        row.addStretch(1)

        root.addLayout(row)

    def _resolve_task_icon(self, key: str, fallback: QStyle.StandardPixmap) -> QIcon:
        icon_path = ICON_DIR / f"{key}.png"
        if icon_path.exists():
            icon = QIcon(str(icon_path))
            if not icon.isNull():
                return icon
        return self.style().standardIcon(fallback)


class BasicPanel(QWidget):
    def __init__(self, title: str, subtitle: str, on_back) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(20, 18, 20, 18)
        root.setSpacing(12)

        top = QHBoxLayout()
        btn_back = QPushButton("\u8fd4\u56de\u4e3b\u754c\u9762")
        btn_back.setObjectName("backMainBtn")
        btn_back.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        btn_back.setIconSize(QSize(14, 14))
        btn_back.clicked.connect(on_back)
        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")
        top.addWidget(btn_back)
        top.addWidget(title_label, 1)
        root.addLayout(top)

        sub = QLabel(subtitle)
        sub.setObjectName("hint")
        sub.setWordWrap(True)
        root.addWidget(sub)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(10)
        self.scroll.setWidget(content)
        root.addWidget(self.scroll, 1)

    def add_form_group(self, title: str) -> QFormLayout:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setHorizontalSpacing(14)
        form.setVerticalSpacing(10)
        self.content_layout.addWidget(group)
        return form

    def add_grid_group(self, title: str) -> QGridLayout:
        group = QGroupBox(title)
        grid = QGridLayout(group)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(8)
        self.content_layout.addWidget(group)
        return grid

    def add_log_placeholder(self) -> None:
        group = QGroupBox("\u65e5\u5fd7\uff08\u5360\u4f4d\uff09")
        layout = QVBoxLayout(group)
        log = QPlainTextEdit()
        log.setPlaceholderText("\u8fd9\u91cc\u540e\u7eed\u63a5\u5b9e\u65f6\u65e5\u5fd7...")
        log.setMinimumHeight(130)
        layout.addWidget(log)
        self.content_layout.addWidget(group)


class TitleBar(QFrame):
    def __init__(self, host_window: QMainWindow) -> None:
        super().__init__()
        self.host_window = host_window
        self._drag_offset = None
        self.setObjectName("titleBar")
        self.setFixedHeight(48)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 6, 10, 6)
        layout.setSpacing(6)

        self.title_label = QLabel(host_window.windowTitle())
        self.title_label.setObjectName("titleBarTitle")
        self.title_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.title_label, 1)

        self.btn_min = QPushButton("\u2013")
        self.btn_min.setObjectName("titleBarBtn")
        self.btn_min.setFixedSize(34, 28)
        self.btn_min.clicked.connect(self.host_window.showMinimized)
        layout.addWidget(self.btn_min)

        self.btn_max = QPushButton("\u25a1")
        self.btn_max.setObjectName("titleBarBtn")
        self.btn_max.setFixedSize(34, 28)
        self.btn_max.clicked.connect(self._toggle_max_restore)
        layout.addWidget(self.btn_max)

        self.btn_close = QPushButton("\u2715")
        self.btn_close.setObjectName("titleBarCloseBtn")
        self.btn_close.setFixedSize(34, 28)
        self.btn_close.clicked.connect(self.host_window.close)
        layout.addWidget(self.btn_close)

    def _toggle_max_restore(self) -> None:
        if self.host_window.isMaximized():
            self.host_window.showNormal()
            self.btn_max.setText("\u25a1")
        else:
            self.host_window.showMaximized()
            self.btn_max.setText("\u2752")

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            if not self.host_window.isMaximized():
                handle = self.window().windowHandle()
                if handle is not None and handle.startSystemMove():
                    event.accept()
                    return
            self._drag_offset = event.globalPosition().toPoint() - self.host_window.frameGeometry().topLeft()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.buttons() & Qt.LeftButton and self._drag_offset is not None and not self.host_window.isMaximized():
            self.host_window.move(event.globalPosition().toPoint() - self._drag_offset)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        self._drag_offset = None
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.LeftButton:
            self._toggle_max_restore()
        super().mouseDoubleClickEvent(event)


class FullPipelinePanel(BasicPanel):
    """Pure UI clone of train_qt's full workflow panel (no backend bindings)."""

    def __init__(self, on_back) -> None:
        super().__init__(
            title="\u7ecf\u5178\u5168\u6d41\u7a0b",
            subtitle="train_qt \u754c\u9762\u590d\u5236\u7248\uff08\u7eafUI\uff0c\u4e0d\u5173\u8054 train_qt \u6216\u540e\u7aef\uff09",
            on_back=on_back,
        )

        top_form = self.add_form_group("\u57fa\u7840\u53c2\u6570")
        path_mode = QComboBox()
        path_mode.addItems(["auto", "linux", "windows"])
        data_device = QComboBox()
        data_device.addItems(["cuda", "cpu"])
        resolution = QComboBox()
        resolution.addItems(["default", "1", "2", "4", "8"])
        top_form.addRow("Python", QLineEdit("python"))
        top_form.addRow("\u8def\u5f84\u6a21\u5f0f", path_mode)
        top_form.addRow("\u6570\u636e\u8bbe\u5907", data_device)
        top_form.addRow("\u5206\u8fa8\u7387\u500d\u7387", resolution)

        train_grid = self.add_grid_group("\u8bad\u7ec3\u53c2\u6570")
        row = 0
        train_grid.addWidget(QLabel("\u573a\u666f\u8def\u5f84"), row, 0)
        train_grid.addWidget(make_path_field("D:/Scene_0320"), row, 1, 1, 3)
        row += 1
        train_grid.addWidget(QLabel("\u8f93\u51fa\u8def\u5f84"), row, 0)
        train_grid.addWidget(make_path_field("D:/Scene_0320/test0323"), row, 1, 1, 3)
        row += 1
        train_grid.addWidget(QLabel("\u8d77\u59cb checkpoint"), row, 0)
        train_grid.addWidget(make_path_field("D:/Scene_0320/chkpnt30000_from_point_cloud_clip_aligned.pth", "\u9009\u62e9\u6587\u4ef6..."), row, 1, 1, 3)
        row += 1
        iters = QSpinBox()
        iters.setRange(1, 10_000_000)
        iters.setValue(40000)
        start_feat = QSpinBox()
        start_feat.setRange(0, 10_000_000)
        start_feat.setValue(30000)
        train_grid.addWidget(QLabel("iterations"), row, 0)
        train_grid.addWidget(iters, row, 1)
        train_grid.addWidget(QLabel("start_ins_feat_iter"), row, 2)
        train_grid.addWidget(start_feat, row, 3)
        row += 1
        sam_level = QSpinBox()
        sam_level.setRange(0, 10)
        sam_level.setValue(0)
        ckpt_points = QSpinBox()
        ckpt_points.setRange(0, 100_000_000)
        train_grid.addWidget(QLabel("sam_level"), row, 0)
        train_grid.addWidget(sam_level, row, 1)
        train_grid.addWidget(QLabel("ckpt\u70b9\u6570\u4e0a\u9650"), row, 2)
        train_grid.addWidget(ckpt_points, row, 3)
        row += 1
        train_grid.addWidget(QLabel("save_iterations"), row, 0)
        train_grid.addWidget(QLineEdit("40000"), row, 1)
        train_grid.addWidget(QLabel("checkpoint_iterations"), row, 2)
        train_grid.addWidget(QLineEdit("40000"), row, 3)
        row += 1
        train_grid.addWidget(QCheckBox("eval"), row, 1)
        train_grid.addWidget(QCheckBox("save_memory"), row, 2)

        sam_grid = self.add_grid_group("SAM3\uff08\u53ef\u9009\uff09")
        row = 0
        run_sam3 = QCheckBox("\u8dd1 SAM3")
        run_train = QCheckBox("\u8dd1\u8bad\u7ec3")
        run_train.setChecked(True)
        sam_export = QCheckBox("SAM3 \u7ed3\u679c\u8f6c language_features")
        sam_export.setChecked(True)
        sam_grid.addWidget(run_sam3, row, 0)
        sam_grid.addWidget(run_train, row, 1)
        sam_grid.addWidget(sam_export, row, 2, 1, 2)
        row += 1
        sam_grid.addWidget(QLabel("SAM3 \u8f93\u5165\u76ee\u5f55"), row, 0)
        sam_grid.addWidget(make_path_field("D:/Scene_0320/images_8"), row, 1, 1, 3)
        row += 1
        sam_grid.addWidget(QLabel("SAM3 \u8f93\u51fa\u76ee\u5f55"), row, 0)
        sam_grid.addWidget(make_path_field(""), row, 1, 1, 3)
        row += 1
        sam_device = QComboBox()
        sam_device.addItems(["cuda", "cpu"])
        sam_grid.addWidget(QLabel("SAM3 \u6743\u91cd"), row, 0)
        sam_grid.addWidget(make_path_field("sam3_large.pt", "\u9009\u62e9\u6587\u4ef6..."), row, 1, 1, 2)
        sam_grid.addWidget(sam_device, row, 3)

        post_grid = self.add_grid_group("\u540e\u5904\u7406\uff08\u53ef\u9009\uff09")
        row = 0
        run_post = QCheckBox("\u8dd1\u540e\u5904\u7406")
        run_mask2inst = QCheckBox("\u8bed\u4e49 mask \u8f6c\u5b9e\u4f8b mask")
        run_mask2inst.setChecked(True)
        run_cluster = QCheckBox("\u805a\u7c7b")
        run_cluster.setChecked(True)
        run_sem = QCheckBox("\u8bed\u4e49\u805a\u5408")
        run_sem.setChecked(True)
        run_ins = QCheckBox("\u5b9e\u4f8b\u5206\u5272")
        run_ins.setChecked(True)
        post_grid.addWidget(run_post, row, 0)
        post_grid.addWidget(run_mask2inst, row, 1)
        post_grid.addWidget(run_cluster, row, 2)
        post_grid.addWidget(run_sem, row, 3)
        row += 1
        post_grid.addWidget(run_ins, row, 1)
        row += 1
        post_iter = QSpinBox()
        post_iter.setRange(1, 10_000_000)
        post_iter.setValue(40000)
        n_clusters = QSpinBox()
        n_clusters.setRange(1, 4096)
        n_clusters.setValue(8)
        post_grid.addWidget(QLabel("\u540e\u5904\u7406\u8fed\u4ee3\u6b65"), row, 0)
        post_grid.addWidget(post_iter, row, 1)
        post_grid.addWidget(QLabel("\u805a\u7c7b\u6570 K"), row, 2)
        post_grid.addWidget(n_clusters, row, 3)
        row += 1
        post_grid.addWidget(QLabel("\u540e\u5904\u7406\u8f93\u5165 PLY"), row, 0)
        post_grid.addWidget(make_path_field("", "\u9009\u62e9\u6587\u4ef6..."), row, 1, 1, 3)
        row += 1
        post_grid.addWidget(QLabel("\u805a\u7c7b\u8f93\u51fa\u76ee\u5f55"), row, 0)
        post_grid.addWidget(make_path_field(""), row, 1, 1, 3)
        row += 1
        post_grid.addWidget(QLabel("\u8bed\u4e49\u8f93\u51fa\u76ee\u5f55"), row, 0)
        post_grid.addWidget(make_path_field(""), row, 1, 1, 3)
        row += 1
        post_grid.addWidget(QLabel("\u5b9e\u4f8b\u8f93\u51fa\u76ee\u5f55"), row, 0)
        post_grid.addWidget(make_path_field(""), row, 1, 1, 3)
        row += 1
        post_grid.addWidget(QLabel("\u8bed\u4e49\u6295\u7968\u5ffd\u7565 ID"), row, 0)
        post_grid.addWidget(QLineEdit("0,7"), row, 1)
        post_min_votes = QSpinBox()
        post_min_votes.setRange(0, 10_000_000)
        post_min_votes.setValue(800)
        post_grid.addWidget(QLabel("\u8bed\u4e49\u6700\u5c0f\u7968\u6570"), row, 2)
        post_grid.addWidget(post_min_votes, row, 3)
        row += 1
        top1 = QDoubleSpinBox()
        top1.setRange(0.0, 1.0)
        top1.setSingleStep(0.05)
        top1.setValue(0.4)
        post_grid.addWidget(QLabel("\u8bed\u4e49 Top1 \u9608\u503c"), row, 0)
        post_grid.addWidget(top1, row, 1)
        auto_id2label = QCheckBox("\u81ea\u52a8\u540c\u6b65 id2label")
        auto_id2label.setChecked(True)
        post_grid.addWidget(auto_id2label, row, 2, 1, 2)
        row += 1
        post_grid.addWidget(QLabel("mask2inst \u8f93\u51fa\u76ee\u5f55"), row, 0)
        post_grid.addWidget(make_path_field(""), row, 1, 1, 3)
        row += 1
        post_grid.addWidget(QLabel("mask2inst \u5ffd\u7565 ID"), row, 0)
        post_grid.addWidget(QLineEdit("0"), row, 1)
        min_area = QSpinBox()
        min_area.setRange(1, 1_000_000)
        min_area.setValue(20)
        post_grid.addWidget(QLabel("mask2inst \u6700\u5c0f\u9762\u79ef"), row, 2)
        post_grid.addWidget(min_area, row, 3)
        row += 1
        post_grid.addWidget(QLabel("id2label"), row, 0)
        post_grid.addWidget(QLineEdit("0:background,1:vehicle,2:person,3:bicycle,4:vegetation,5:road,6:traffic_facility,7:other"), row, 1, 1, 2)
        post_grid.addWidget(QPushButton("\u4ece task \u8bfb\u53d6"), row, 3)
        row += 1
        post_grid.addWidget(QLabel("\u8bed\u4e49 stuff_ids"), row, 0)
        post_grid.addWidget(QLineEdit("0,1,2,3,4,5,6,7,255"), row, 1, 1, 3)
        row += 1
        post_grid.addWidget(QLabel("\u5b9e\u4f8b\u53ea\u5904\u7406\u8bed\u4e49 ID"), row, 0)
        post_grid.addWidget(QLineEdit("1"), row, 1)
        post_grid.addWidget(QLabel("\u5b9e\u4f8b\u9636\u6bb5 stuff_ids"), row, 2)
        post_grid.addWidget(QLineEdit("0,4,5,6,7,255"), row, 3)
        row += 1
        post_grid.addWidget(QLabel("\u5b9e\u4f8b\u5355\u6807\u7b7e"), row, 0)
        post_grid.addWidget(QLineEdit(""), row, 1, 1, 3)
        row += 1
        post_grid.addWidget(QLabel("\u5b9e\u4f8b mask \u76ee\u5f55"), row, 0)
        post_grid.addWidget(make_path_field(""), row, 1, 1, 3)
        row += 1
        mode = QComboBox()
        mode.addItems(["name", "index"])
        level = QSpinBox()
        level.setRange(0, 10)
        post_grid.addWidget(QLabel("\u5b9e\u4f8b mask \u6a21\u5f0f"), row, 0)
        post_grid.addWidget(mode, row, 1)
        post_grid.addWidget(QLabel("\u5b9e\u4f8b mask \u5c42\u7ea7"), row, 2)
        post_grid.addWidget(level, row, 3)
        row += 1
        stride = QSpinBox()
        stride.setRange(1, 9999)
        stride.setValue(1)
        min_mask_points = QSpinBox()
        min_mask_points.setRange(1, 10_000_000)
        min_mask_points.setValue(30)
        post_grid.addWidget(QLabel("\u5b9e\u4f8b\u6295\u7968\u6b65\u957f"), row, 0)
        post_grid.addWidget(stride, row, 1)
        post_grid.addWidget(QLabel("\u5b9e\u4f8b\u6700\u5c0f mask \u70b9\u6570"), row, 2)
        post_grid.addWidget(min_mask_points, row, 3)
        row += 1
        match_iou = QDoubleSpinBox()
        match_iou.setRange(0.0, 1.0)
        match_iou.setSingleStep(0.05)
        match_iou.setValue(0.2)
        min_votes_inst = QSpinBox()
        min_votes_inst.setRange(1, 10_000_000)
        min_votes_inst.setValue(2)
        post_grid.addWidget(QLabel("\u5b9e\u4f8b\u5339\u914d IoU"), row, 0)
        post_grid.addWidget(match_iou, row, 1)
        post_grid.addWidget(QLabel("\u5b9e\u4f8b\u6700\u5c0f\u70b9\u7968\u6570"), row, 2)
        post_grid.addWidget(min_votes_inst, row, 3)
        row += 1
        min_inst_points = QSpinBox()
        min_inst_points.setRange(1, 10_000_000)
        min_inst_points.setValue(180)
        post_grid.addWidget(QLabel("\u5b9e\u4f8b\u6700\u5c0f\u70b9\u6570"), row, 0)
        post_grid.addWidget(min_inst_points, row, 1)
        save_parts = QCheckBox("\u4fdd\u5b58\u6bcf\u4e2a\u5b9e\u4f8b\u5b50 PLY")
        save_parts.setChecked(True)
        post_grid.addWidget(save_parts, row, 2, 1, 2)

        lite_group = QGroupBox("\u540e\u5904\u7406\u7cbe\u7b80\u7248\uff08Step1 \u8bed\u4e49\u547d\u540d + Step2 \u5b9e\u4f8b\u5206\u5272\uff09")
        lite_layout = QVBoxLayout(lite_group)
        lite_layout.addWidget(QCheckBox("\u8dd1\u540e\u5904\u7406\u7cbe\u7b80\u7248"))
        lite_hint = QLabel("\u4f7f\u7528\u4e0a\u65b9 SAM3 \u6743\u91cd/\u8bbe\u5907\uff1bscene_path \u4f7f\u7528\u573a\u666f\u8def\u5f84\u3002")
        lite_hint.setWordWrap(True)
        lite_layout.addWidget(lite_hint)

        step1_group = QGroupBox("STEP1\uff1akmeans \u8bed\u4e49\u91cd\u547d\u540d")
        step1_grid = QGridLayout(step1_group)
        r = 0
        step1_grid.addWidget(QLabel("Step1 \u8f93\u5165 kmeans \u76ee\u5f55"), r, 0)
        step1_grid.addWidget(make_path_field(""), r, 1, 1, 3)
        r += 1
        step1_grid.addWidget(QLabel("Step1 \u8f93\u51fa\u8bed\u4e49\u76ee\u5f55"), r, 0)
        step1_grid.addWidget(make_path_field(""), r, 1, 1, 3)
        r += 1
        step1_grid.addWidget(QLabel("Step1 \u8bed\u4e49\u5019\u9009"), r, 0)
        step1_grid.addWidget(QLineEdit("building,vehicle,person,bicycle,vegetation,road,traffic_facility,other"), r, 1, 1, 2)
        step1_grid.addWidget(QPushButton("\u4ece SAM3 \u8bfb\u53d6"), r, 3)
        r += 1
        probe = QSpinBox()
        probe.setRange(1, 128)
        probe.setValue(6)
        views = QSpinBox()
        views.setRange(1, 128)
        views.setValue(6)
        step1_grid.addWidget(QLabel("\u8bed\u4e49\u63a2\u6d4b\u89c6\u89d2\u6570"), r, 0)
        step1_grid.addWidget(probe, r, 1)
        step1_grid.addWidget(QLabel("\u4fef\u89c6\u89c6\u89d2\u6570"), r, 2)
        step1_grid.addWidget(views, r, 3)
        r += 1
        img_size = QSpinBox()
        img_size.setRange(256, 4096)
        img_size.setSingleStep(128)
        img_size.setValue(1024)
        fov = QDoubleSpinBox()
        fov.setRange(10.0, 170.0)
        fov.setValue(58.0)
        step1_grid.addWidget(QLabel("\u6e32\u67d3\u5206\u8fa8\u7387"), r, 0)
        step1_grid.addWidget(img_size, r, 1)
        step1_grid.addWidget(QLabel("FOV(\u5ea6)"), r, 2)
        step1_grid.addWidget(fov, r, 3)
        r += 1
        xoy = QDoubleSpinBox()
        xoy.setRange(0.2, 5.0)
        xoy.setSingleStep(0.1)
        xoy.setValue(1.0)
        step1_grid.addWidget(QLabel("XOY \u6b65\u957f\u500d\u7387"), r, 0)
        step1_grid.addWidget(xoy, r, 1)
        step1_grid.addWidget(QPushButton("Step1: kmeans \u8bed\u4e49\u91cd\u547d\u540d"), r, 2, 1, 2)
        lite_layout.addWidget(step1_group)

        step2_group = QGroupBox("STEP2\uff1a\u4fef\u89c6\u6e32\u67d3 -> SAM3 \u5b9e\u4f8b -> \u7eb5\u5411\u5207\u5272")
        step2_grid = QGridLayout(step2_group)
        r = 0
        step2_grid.addWidget(QLabel("\u8f93\u5165 PLY"), r, 0)
        step2_grid.addWidget(make_path_field("", "\u9009\u62e9\u6587\u4ef6..."), r, 1, 1, 3)
        r += 1
        step2_grid.addWidget(QLabel("\u8f93\u51fa\u5b50\u76ee\u5f55"), r, 0)
        step2_grid.addWidget(QLineEdit("topdown_instance_pipeline"), r, 1)
        step2_grid.addWidget(QLabel("\u4fef\u89c6\u89c6\u89d2\u6570"), r, 2)
        step2_grid.addWidget(QSpinBox(), r, 3)
        r += 1
        step2_grid.addWidget(QLabel("\u6e32\u67d3\u5206\u8fa8\u7387"), r, 0)
        step2_grid.addWidget(QSpinBox(), r, 1)
        step2_grid.addWidget(QLabel("FOV(\u5ea6)"), r, 2)
        step2_grid.addWidget(QDoubleSpinBox(), r, 3)
        r += 1
        step2_grid.addWidget(QLabel("XOY \u6b65\u957f\u500d\u7387"), r, 0)
        step2_grid.addWidget(QDoubleSpinBox(), r, 1)
        step2_grid.addWidget(QLabel("SAM \u63d0\u793a\u8bcd"), r, 2)
        step2_grid.addWidget(QLineEdit("building"), r, 3)
        r += 1
        sem_id = QSpinBox()
        sem_id.setRange(0, 255)
        sem_id.setValue(3)
        min_pts = QSpinBox()
        min_pts.setRange(1, 10_000_000)
        min_pts.setValue(3000)
        step2_grid.addWidget(QLabel("\u8bed\u4e49 ID"), r, 0)
        step2_grid.addWidget(sem_id, r, 1)
        step2_grid.addWidget(QLabel("\u6700\u5c0f\u5b9e\u4f8b\u70b9\u6570"), r, 2)
        step2_grid.addWidget(min_pts, r, 3)
        r += 1
        save_parts2 = QCheckBox("\u4fdd\u5b58\u6bcf\u4e2a\u5b9e\u4f8b\u5b50 PLY")
        save_parts2.setChecked(True)
        step2_grid.addWidget(save_parts2, r, 0, 1, 2)
        step2_grid.addWidget(QPushButton("\u586b\u5145\u5efa\u7b51\u9ed8\u8ba4\u53c2\u6570"), r, 2, 1, 2)
        r += 1
        step2_grid.addWidget(QPushButton("\u751f\u6210\u6e32\u67d3\u56fe"), r, 0, 1, 2)
        step2_grid.addWidget(QPushButton("\u57fa\u4e8e\u5f53\u524d\u6e32\u67d3\u505a\u5b9e\u4f8b\u5206\u5272"), r, 2, 1, 2)
        lite_layout.addWidget(step2_group)
        self.content_layout.addWidget(lite_group)

        preview_group = QGroupBox("\u4fef\u89c6\u6e32\u67d3\u56fe\u9884\u89c8\uff08\u7cbe\u7b80\u540e\u5904\u7406\uff09")
        preview_layout = QVBoxLayout(preview_group)
        top_btns = QHBoxLayout()
        top_btns.addWidget(QPushButton("\u5237\u65b0\u6e32\u67d3\u5217\u8868"))
        top_btns.addWidget(QPushButton("\u4e0a\u4e00\u5f20"))
        top_btns.addWidget(QPushButton("\u4e0b\u4e00\u5f20"))
        top_btns.addStretch(1)
        preview_layout.addLayout(top_btns)
        preview_layout.addWidget(QLabel("\u70b9\u51fb\u201c\u5237\u65b0\u6e32\u67d3\u5217\u8868\u201d\u8bfb\u53d6 renders_topdown"))
        preview_img = QLabel("\u6e32\u67d3\u56fe\u9884\u89c8")
        preview_img.setAlignment(Qt.AlignCenter)
        preview_img.setMinimumSize(520, 320)
        preview_img.setStyleSheet("border: 1px solid #D4D7DE; background: #F7F9FC;")
        preview_layout.addWidget(preview_img)
        self.content_layout.addWidget(preview_group)

        action_group = QGroupBox("\u64cd\u4f5c")
        action_layout = QHBoxLayout(action_group)
        action_layout.addWidget(QPushButton("\u9884\u89c8\u547d\u4ee4"))
        action_layout.addWidget(QPushButton("\u5f00\u59cb"))
        action_layout.addWidget(QPushButton("\u505c\u6b62"))
        action_layout.addWidget(QPushButton("\u624b\u52a8\u5237\u65b0"))
        self.content_layout.addWidget(action_group)

        status_group = QGroupBox("\u72b6\u6001")
        status_layout = QGridLayout(status_group)
        status_layout.addWidget(QLabel("\u72b6\u6001"), 0, 0)
        status_layout.addWidget(QLabel("\u7a7a\u95f2"), 0, 1)
        status_layout.addWidget(QLabel("\u5f53\u524d\u8fed\u4ee3"), 0, 2)
        status_layout.addWidget(QLabel("0"), 0, 3)
        status_layout.addWidget(QLabel("\u8d44\u6e90\u76d1\u63a7"), 1, 0)
        status_layout.addWidget(QLabel("-"), 1, 1, 1, 3)
        status_layout.addWidget(QLabel("\u8bad\u7ec3\u8fdb\u5ea6"), 2, 0)
        progress = QProgressBar()
        progress.setRange(0, 1000)
        progress.setValue(0)
        status_layout.addWidget(progress, 2, 1, 1, 3)
        self.content_layout.addWidget(status_group)

        cmd_group = QGroupBox("\u547d\u4ee4\u9884\u89c8")
        cmd_layout = QVBoxLayout(cmd_group)
        cmd_box = QPlainTextEdit()
        cmd_box.setReadOnly(True)
        cmd_layout.addWidget(cmd_box)
        self.content_layout.addWidget(cmd_group)

        log_group = QGroupBox("\u65e5\u5fd7")
        log_layout = QVBoxLayout(log_group)
        log_box = QPlainTextEdit()
        log_box.setReadOnly(True)
        log_layout.addWidget(log_box)
        self.content_layout.addWidget(log_group)

        feat_group = QGroupBox("Stage1 \u7279\u5f81\u6e32\u67d3\u5bf9\u6bd4")
        feat_layout = QVBoxLayout(feat_group)
        feat_imgs = QHBoxLayout()
        gt = QLabel("GT \u9884\u89c8\uff08\u70b9\u51fb\u5237\u65b0\uff09")
        gt.setAlignment(Qt.AlignCenter)
        gt.setMinimumSize(420, 280)
        gt.setStyleSheet("border: 1px solid #D4D7DE; background: #F7F9FC;")
        ins = QLabel("INS_FEAT \u9884\u89c8\uff08\u70b9\u51fb\u5237\u65b0\uff09")
        ins.setAlignment(Qt.AlignCenter)
        ins.setMinimumSize(420, 280)
        ins.setStyleSheet("border: 1px solid #D4D7DE; background: #F7F9FC;")
        feat_imgs.addWidget(gt)
        feat_imgs.addWidget(ins)
        feat_layout.addLayout(feat_imgs)
        feat_layout.addWidget(QLabel("\u70b9\u51fb\u201c\u5237\u65b0\u7279\u5f81\u56fe\u201d\u52a0\u8f7d\u6700\u65b0 GT/INS_FEAT \u5bf9\u6bd4"))
        feat_layout.addWidget(QPushButton("\u5237\u65b0\u7279\u5f81\u56fe"), alignment=Qt.AlignLeft)
        self.content_layout.addWidget(feat_group)
        self.content_layout.addStretch(1)


class FeatureTrainPanel(BasicPanel):
    def __init__(self, on_back) -> None:
        super().__init__(
            title="\u7279\u5f81\u8bad\u7ec3\uff08Stage1\uff09",
            subtitle="\u805a\u7126 Stage1\uff0c\u5e03\u5c40\u98ce\u683c\u4e0e\u5168\u6d41\u7a0b\u4e00\u81f4\uff08\u7eaf UI \u7248\uff09",
            on_back=on_back,
        )

        basic = self.add_form_group("\u57fa\u7840\u53c2\u6570")
        mode = QComboBox()
        mode.addItems(["auto", "linux", "windows"])
        device = QComboBox()
        device.addItems(["cuda", "cpu"])
        res = QComboBox()
        res.addItems(["default", "1", "2", "4", "8"])
        basic.addRow("Python", QLineEdit("python"))
        basic.addRow("\u8def\u5f84\u6a21\u5f0f", mode)
        basic.addRow("\u6570\u636e\u8bbe\u5907", device)
        basic.addRow("\u5206\u8fa8\u7387\u500d\u7387", res)

        train = self.add_grid_group("Stage1 \u8bad\u7ec3\u53c2\u6570")
        r = 0
        train.addWidget(QLabel("\u573a\u666f\u8def\u5f84"), r, 0)
        train.addWidget(make_path_field("D:/Scene_0320"), r, 1, 1, 3)
        r += 1
        train.addWidget(QLabel("\u8f93\u51fa\u8def\u5f84"), r, 0)
        train.addWidget(make_path_field("D:/Scene_0320/test0323"), r, 1, 1, 3)
        r += 1
        train.addWidget(QLabel("\u8d77\u59cb checkpoint"), r, 0)
        train.addWidget(make_path_field("D:/Scene_0320/chkpnt30000_from_point_cloud_clip_aligned.pth", "\u9009\u62e9\u6587\u4ef6..."), r, 1, 1, 3)
        r += 1
        iter_spin = QSpinBox()
        iter_spin.setRange(1, 10_000_000)
        iter_spin.setValue(40000)
        feat_spin = QSpinBox()
        feat_spin.setRange(0, 10_000_000)
        feat_spin.setValue(30000)
        train.addWidget(QLabel("iterations"), r, 0)
        train.addWidget(iter_spin, r, 1)
        train.addWidget(QLabel("start_ins_feat_iter"), r, 2)
        train.addWidget(feat_spin, r, 3)
        r += 1
        train.addWidget(QLabel("save_iterations"), r, 0)
        train.addWidget(QLineEdit("40000"), r, 1)
        train.addWidget(QLabel("checkpoint_iterations"), r, 2)
        train.addWidget(QLineEdit("40000"), r, 3)
        r += 1
        train.addWidget(QCheckBox("eval"), r, 1)
        train.addWidget(QCheckBox("save_memory"), r, 2)

        action_group = QGroupBox("\u64cd\u4f5c")
        action_layout = QHBoxLayout(action_group)
        action_layout.addWidget(QPushButton("\u9884\u89c8\u547d\u4ee4"))
        action_layout.addWidget(QPushButton("\u5f00\u59cb"))
        action_layout.addWidget(QPushButton("\u505c\u6b62"))
        self.content_layout.addWidget(action_group)

        self.add_log_placeholder()
        self.content_layout.addStretch(1)


class PlaceholderPanel(BasicPanel):
    def __init__(self, title: str, on_back) -> None:
        super().__init__(
            title=title,
            subtitle="\u8be5\u9875\u6682\u4e3a\u5360\u4f4d\uff0c\u540e\u7eed\u53ef\u6309\u4f60\u7684\u529f\u80fd\u5355\u72ec\u8bbe\u8ba1\u3002",
            on_back=on_back,
        )
        form = self.add_form_group("\u5f85\u5b9a\u5236")
        form.addRow("\u5f53\u524d\u72b6\u6001", QLabel("\u53ea\u5b8c\u6210 UI \u5360\u4f4d"))
        form.addRow("\u5907\u6ce8", QLabel("\u4fdd\u7559\u4e3a\u72ec\u7acb\u9875\u9762\uff0c\u65b9\u4fbf\u540e\u7eed\u5f00\u53d1"))
        self.add_log_placeholder()
        self.content_layout.addStretch(1)


class TaskHubWindow(QMainWindow):
    MAIN_SIZE = (1650, 660)
    PANEL_SIZE = (1520, 900)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("3D Seg Workflow Hub")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.resize(*self.MAIN_SIZE)

        root = QWidget()
        root.setObjectName("windowRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)
        root_layout.setSpacing(0)

        self.app_frame = QFrame()
        self.app_frame.setObjectName("appFrame")
        app_layout = QVBoxLayout(self.app_frame)
        app_layout.setContentsMargins(0, 0, 0, 0)
        app_layout.setSpacing(0)

        self.title_bar = TitleBar(self)
        app_layout.addWidget(self.title_bar, 0)

        self.stack = QStackedWidget()
        self.stack.setObjectName("contentStack")
        app_layout.addWidget(self.stack, 1)

        root_layout.addWidget(self.app_frame)
        self.setCentralWidget(root)

        self.main_page = MainPage(on_open_panel=self.show_panel)
        self.stack.addWidget(self.main_page)

        self._panels: dict[str, QWidget] = {
            "classic": FullPipelinePanel(on_back=self.show_main_page),
            "sam3": PlaceholderPanel(title="SAM3", on_back=self.show_main_page),
            "feature_train": FeatureTrainPanel(on_back=self.show_main_page),
            "cluster": PlaceholderPanel(title="\u805a\u7c7b", on_back=self.show_main_page),
            "post_v1": PlaceholderPanel(title="\u540e\u5904\u7406 V1", on_back=self.show_main_page),
            "post_v2": PlaceholderPanel(title="\u540e\u5904\u7406 V2", on_back=self.show_main_page),
        }
        for panel in self._panels.values():
            self.stack.addWidget(panel)

        self.show_main_page()
        self._apply_style()

    def show_main_page(self) -> None:
        self.stack.setCurrentWidget(self.main_page)
        self.resize(*self.MAIN_SIZE)
        self.setMinimumSize(1400, 620)

    def show_panel(self, key: str) -> None:
        self.stack.setCurrentWidget(self._panels[key])
        self.resize(*self.PANEL_SIZE)
        self.setMinimumSize(1200, 760)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { font-size: 13px; background: #F5F7FB; }
            QWidget#windowRoot { background: transparent; }
            QFrame#appFrame {
                background: #F5F7FB;
                border: 1px solid #C9D3E6;
                border-radius: 14px;
            }
            QFrame#titleBar {
                background: #EAF0FB;
                border-top-left-radius: 14px;
                border-top-right-radius: 14px;
                border-bottom: 1px solid #D5DEEE;
            }
            QLabel#titleBarTitle {
                font-size: 14px;
                font-weight: 700;
                color: #1E2A3A;
                background: transparent;
                border: none;
                padding-left: 4px;
            }
            QPushButton#titleBarBtn {
                background: #F4F7FD;
                border: 1px solid #CAD4E6;
                border-radius: 10px;
                color: #213049;
                padding: 0;
            }
            QPushButton#titleBarBtn:hover { background: #E6EEFC; }
            QPushButton#titleBarCloseBtn {
                background: #FFF1F1;
                border: 1px solid #EDC3C3;
                border-radius: 10px;
                color: #912F2F;
                padding: 0;
            }
            QPushButton#titleBarCloseBtn:hover { background: #FFDCDC; }
            QStackedWidget#contentStack {
                border-bottom-left-radius: 14px;
                border-bottom-right-radius: 14px;
                background: #F5F7FB;
            }
            QLabel#mainTitle {
                font-size: 46px;
                font-weight: 800;
                color: #16202B;
                padding: 0;
            }
            QLabel#projectIntro {
                font-size: 14px;
                color: #4A5568;
                padding: 0 140px 0 140px;
            }
            QLabel#panelTitle {
                font-size: 24px;
                font-weight: 700;
                color: #1D2530;
            }
            QLabel#hint {
                color: #384151;
                font-size: 16px;
                font-weight: 600;
            }
            QFrame#taskTile {
                border: 1px solid #CFD5DF;
                border-radius: 26px;
                background: #FFFFFF;
            }
            QFrame#taskTile[hovered="true"] {
                background: #EFF4FF;
                border-color: #B6C4E6;
            }
            QLabel#taskTitle {
                font-size: 30px;
                font-weight: 700;
                color: #1C2530;
                background: transparent;
                border: none;
            }
            QLabel#taskDesc {
                font-size: 13px;
                color: #4A5568;
                background: transparent;
                border: none;
            }
            QLabel#taskIcon {
                background: transparent;
                border: none;
            }
            QGroupBox {
                border: 1px solid #D2D9E5;
                border-radius: 20px;
                margin-top: 10px;
                padding-top: 14px;
                background: #F9FBFF;
                font-weight: 600;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 16px;
                padding: 0 6px;
                color: #2D3A4F;
            }
            QPushButton {
                border-radius: 12px;
                padding: 7px 12px;
            }
            QScrollArea QPushButton {
                max-width: 260px;
            }
            QPushButton#backMainBtn {
                padding: 8px 14px;
                font-weight: 700;
                border: 1px solid #9FB3D9;
                background: #E9F0FF;
                color: #1A3566;
                max-width: 170px;
            }
            QPushButton#backMainBtn:hover {
                background: #DCE8FF;
                border-color: #89A2D4;
            }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                border-radius: 12px;
            }
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    window = TaskHubWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
