#!/usr/bin/env python3
"""Workflow hub with train_qt transplanted functionality.

No modifications to train_ui.py / train_qt.py are required.
"""

from __future__ import annotations

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from PySide6.QtCore import QSize, Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import (
        QApplication,
        QFrame,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QPushButton,
        QScrollArea,
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
    ("classic", "经典全流程", QStyle.SP_ComputerIcon, "完整训练与后处理流程。"),
    ("sam3", "SAM3", QStyle.SP_DriveNetIcon, "仅展示 SAM3 相关参数与执行。"),
    ("feature_train", "特征训练", QStyle.SP_FileDialogDetailedView, "仅展示基础参数 + 训练参数。"),
    ("cluster", "聚类", QStyle.SP_DirOpenIcon, "后处理第一步，仅聚类。"),
    ("post_v2", "后处理", QStyle.SP_DialogSaveButton, "后处理后两步（Step1 + Step2）。"),
]

ICON_DIR = Path(__file__).resolve().parent.parent / "assets" / "ui_icons"


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

        icon_label = QLabel()
        icon_label.setObjectName("taskIcon")
        icon_label.setAlignment(Qt.AlignCenter)
        icon_label.setPixmap(icon.pixmap(QSize(82, 82)))
        icon_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(icon_label)

        title_label = QLabel(text)
        title_label.setObjectName("taskTitle")
        title_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        title_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(title_label)

        self.desc_label = QLabel("")
        self.desc_label.setObjectName("taskDesc")
        self.desc_label.setAlignment(Qt.AlignHCenter | Qt.AlignTop)
        self.desc_label.setWordWrap(True)
        self.desc_label.setFixedHeight(46)
        self.desc_label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.desc_label)

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
            "本项目用于 3D 语义/实例分割全流程试验，"
            "涵盖 Stage1 特征训练、SAM3 辅助分割、聚类与后处理模块。"
        )
        intro.setObjectName("projectIntro")
        intro.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        intro.setWordWrap(True)
        root.addWidget(intro)

        title = QLabel("你想做什么？")
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

        self.btn_min = QPushButton("–")
        self.btn_min.setObjectName("titleBarBtn")
        self.btn_min.setFixedSize(34, 28)
        self.btn_min.clicked.connect(self.host_window.showMinimized)
        layout.addWidget(self.btn_min)

        self.btn_max = QPushButton("□")
        self.btn_max.setObjectName("titleBarBtn")
        self.btn_max.setFixedSize(34, 28)
        self.btn_max.clicked.connect(self._toggle_max_restore)
        layout.addWidget(self.btn_max)

        self.btn_close = QPushButton("✕")
        self.btn_close.setObjectName("titleBarCloseBtn")
        self.btn_close.setFixedSize(34, 28)
        self.btn_close.clicked.connect(self.host_window.close)
        layout.addWidget(self.btn_close)

    def _toggle_max_restore(self) -> None:
        if self.host_window.isMaximized():
            self.host_window.showNormal()
            self.btn_max.setText("□")
        else:
            self.host_window.showMaximized()
            self.btn_max.setText("❒")

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


class TransplantedTrainQtPanel(QWidget):
    """Embed train_qt and show only relevant sections per mode."""

    GROUP_ORDER_KEYS = [
        "basic",
        "train",
        "sam3",
        "cluster_step",
        "post_lite",
        "post_step1",
        "post_step2",
        "post_preview",
        "action",
        "status",
        "cmd",
        "log",
        "feat_preview",
    ]

    SHARED_RUNTIME_GROUPS = {"action", "status", "cmd", "log"}

    MODE_VISIBLE_GROUPS = {
        "classic": "ALL",
        "feature_train": {"basic", "train"} | SHARED_RUNTIME_GROUPS,
        "sam3": {"sam3"} | SHARED_RUNTIME_GROUPS,
        "cluster": {"cluster_step"} | SHARED_RUNTIME_GROUPS,
        "post_v2": {"post_lite", "post_step1", "post_step2", "post_preview"} | SHARED_RUNTIME_GROUPS,
    }

    MODE_FOCUS_GROUP = {
        "classic": "basic",
        "feature_train": "basic",
        "sam3": "sam3",
        "cluster": "cluster_step",
        "post_v2": "post_lite",
    }

    def __init__(self, on_back) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        top = QHBoxLayout()
        self.btn_back = QPushButton("返回主界面")
        self.btn_back.setObjectName("backMainBtn")
        self.btn_back.setIcon(self.style().standardIcon(QStyle.SP_ArrowBack))
        self.btn_back.setIconSize(QSize(14, 14))
        self.btn_back.clicked.connect(on_back)
        self.title_label = QLabel("经典全流程")
        self.title_label.setObjectName("panelTitle")
        self.mode_hint = QLabel("")
        self.mode_hint.setObjectName("hint")
        top.addWidget(self.btn_back, 0)
        top.addWidget(self.title_label, 1)
        root.addLayout(top)
        root.addWidget(self.mode_hint)

        self._import_error: Exception | None = None
        self.train_window = None
        self._group_map: dict[str, QGroupBox] = {}

        try:
            import train_qt as train_qt_module

            self.train_window = train_qt_module.TrainQtWindow()
            self.train_window.setWindowFlags(Qt.Widget)
            root.addWidget(self.train_window, 1)
            self._rebuild_group_map()
        except Exception as exc:
            self._import_error = exc
            fallback = QLabel(
                "无法载入 train_qt 功能面板。\n"
                f"错误：{exc}\n"
                "请确认当前环境可运行 scripts/train_qt.py。"
            )
            fallback.setWordWrap(True)
            root.addWidget(fallback, 1)

    def set_mode(self, key: str, title: str) -> None:
        self.title_label.setText(title)
        self.mode_hint.setText(f"当前模式：{title}")
        if self.train_window is None:
            return

        self._apply_mode_visibility(key)
        self._apply_mode_defaults(key)
        self._focus_related_group(key)

    def _rebuild_group_map(self) -> None:
        if self.train_window is None:
            self._group_map = {}
            return
        groups = self.train_window.findChildren(QGroupBox)
        self._group_map = {}
        for i, group in enumerate(groups):
            if i < len(self.GROUP_ORDER_KEYS):
                self._group_map[self.GROUP_ORDER_KEYS[i]] = group

    def _apply_mode_visibility(self, key: str) -> None:
        if not self._group_map:
            return
        rule = self.MODE_VISIBLE_GROUPS.get(key, "ALL")
        visible_names = set(self._group_map.keys()) if rule == "ALL" else set(rule)
        for name, group in self._group_map.items():
            group.setVisible(name in visible_names)

    def _set_checked_if_exists(self, attr: str, value: bool) -> None:
        widget = getattr(self.train_window, attr, None)
        if widget is not None and hasattr(widget, "setChecked"):
            widget.setChecked(value)

    def _apply_mode_defaults(self, key: str) -> None:
        if key == "classic":
            return
        if key == "sam3":
            self._set_checked_if_exists("run_sam3", True)
            self._set_checked_if_exists("run_training", False)
            self._set_checked_if_exists("run_postprocess", False)
            self._set_checked_if_exists("run_postprocess_lite", False)
            self._set_checked_if_exists("run_post_lite_step1", False)
            self._set_checked_if_exists("run_post_lite_step2", False)
            return
        if key == "feature_train":
            self._set_checked_if_exists("run_training", True)
            self._set_checked_if_exists("run_sam3", False)
            self._set_checked_if_exists("run_postprocess", False)
            self._set_checked_if_exists("run_postprocess_lite", False)
            self._set_checked_if_exists("run_post_lite_step1", False)
            self._set_checked_if_exists("run_post_lite_step2", False)
            return
        if key == "cluster":
            self._set_checked_if_exists("run_postprocess", True)
            self._set_checked_if_exists("run_postprocess_lite", False)
            self._set_checked_if_exists("run_post_cluster", True)
            self._set_checked_if_exists("run_post_semantic", False)
            self._set_checked_if_exists("run_post_instance", False)
            self._set_checked_if_exists("run_post_mask2inst", False)
            self._set_checked_if_exists("run_post_lite_step1", False)
            self._set_checked_if_exists("run_post_lite_step2", False)
            return
        if key == "post_v2":
            self._set_checked_if_exists("run_postprocess_lite", True)
            self._set_checked_if_exists("run_postprocess", False)
            self._set_checked_if_exists("run_post_cluster", False)
            self._set_checked_if_exists("run_post_semantic", True)
            self._set_checked_if_exists("run_post_instance", True)
            self._set_checked_if_exists("run_post_mask2inst", False)
            self._set_checked_if_exists("run_post_lite_step1", True)
            self._set_checked_if_exists("run_post_lite_step2", True)

    def _focus_related_group(self, key: str) -> None:
        if self.train_window is None:
            return
        scroll = self.train_window.findChild(QScrollArea)
        if scroll is None:
            return
        group_key = self.MODE_FOCUS_GROUP.get(key)
        if group_key is None:
            return
        target = self._group_map.get(group_key)
        if target is not None and target.isVisible():
            scroll.ensureWidgetVisible(target, 0, 60)


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

        self.main_page = MainPage(on_open_panel=self.show_mode_panel)
        self.stack.addWidget(self.main_page)

        self.functional_panel = TransplantedTrainQtPanel(on_back=self.show_main_page)
        self.stack.addWidget(self.functional_panel)

        self.show_main_page()
        self._apply_style()

    def show_main_page(self) -> None:
        self.stack.setCurrentWidget(self.main_page)
        self.resize(*self.MAIN_SIZE)
        self.setMinimumSize(1400, 620)

    def show_mode_panel(self, key: str) -> None:
        title = next((name for k, name, _icon, _desc in TASK_ITEMS if k == key), key)
        self.functional_panel.set_mode(key, title)
        self.stack.setCurrentWidget(self.functional_panel)
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
                font-size: 14px;
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
            QLabel#taskIcon, QLabel#taskTitle, QLabel#taskDesc {
                background: transparent;
                border: none;
            }
            QLabel#taskTitle {
                font-size: 30px;
                font-weight: 700;
                color: #1C2530;
            }
            QLabel#taskDesc {
                font-size: 13px;
                color: #4A5568;
            }
            QPushButton {
                border-radius: 12px;
                padding: 7px 12px;
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
            """
        )


def main() -> int:
    app = QApplication(sys.argv)
    window = TaskHubWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
