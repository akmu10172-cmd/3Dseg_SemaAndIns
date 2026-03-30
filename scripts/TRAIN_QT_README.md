# PySide6 训练面板（不改 train_ui 后端）

这个方案是“只换界面层”，训练逻辑仍复用 `scripts/train_ui.py` 的后端函数。

## 1. 已准备文件

- `scripts/train_qt.py`
- `scripts/install_train_qt_deps.sh`
- `scripts/run_train_qt.sh`
- `scripts/run_train_qt_windows.cmd`

## 2. 在 WSL 安装依赖

```bash
cd /mnt/d/3Dseg_SemaAndIns
bash scripts/install_train_qt_deps.sh
```

如果你的 Python 路径不同：

```bash
bash scripts/install_train_qt_deps.sh /path/to/python
```

## 3. 在 WSL 启动 Qt 界面

```bash
cd /mnt/d/3Dseg_SemaAndIns
bash scripts/run_train_qt.sh
```

## 4. 在 Windows 侧一键启动（调用 WSL）

```bat
scripts\run_train_qt_windows.cmd
```

可选指定 Python（WSL 路径）：

```bat
scripts\run_train_qt_windows.cmd /home/ysy/miniconda3/envs/opengaussian/bin/python
```

## 5. Qt Designer 可视化编辑（建议工作流）

你可以走这条流程：

1. 用 Qt Designer 画 `.ui` 文件。  
2. 用 `pyside6-uic` 转成 Python。  
3. 在 `train_qt.py` 里把控件信号连接到后端函数（`preview/start/stop/refresh`）。  

示例命令：

```bash
/home/ysy/miniconda3/envs/opengaussian/bin/pyside6-uic your_form.ui -o your_form_ui.py
```

Windows 侧等价流程（推荐）：

1. 在 Windows 虚拟环境安装：

```bat
pip install PySide6
```

2. 打开 Designer（常见路径）：

```bat
%USERPROFILE%\AppData\Local\Programs\Python\Python3x\Lib\site-packages\PySide6\designer.exe
```

3. 生成 Python 界面代码：

```bat
pyside6-uic your_form.ui -o scripts\your_form_ui.py
```

4. 在 `scripts/train_qt.py` 里替换布局层，保留后端调用（`preview/start/stop/refresh`）。

当前 `train_qt.py` 已经把后端对接打通，后续你只需要替换界面布局层。
