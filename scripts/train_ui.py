#!/usr/bin/env python3
"""Lightweight web UI for launching OpenGaussian Stage1 training.

This script does not modify existing training code. It only builds CLI args
and runs `train.py` as a subprocess.
"""

from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
import torch
import numpy as np

try:
    import gradio as gr
except ImportError as exc:
    raise SystemExit(
        "Missing dependency: gradio\n"
        "Install with: pip install gradio\n"
        f"Detail: {exc}"
    )

try:
    import gradio_client.utils as _gradio_client_utils
except Exception:
    _gradio_client_utils = None

try:
    import psutil

    PSUTIL_AVAILABLE = True
except Exception:
    psutil = None
    PSUTIL_AVAILABLE = False


def _patch_gradio_schema_parser() -> None:
    """Work around gradio_client schema parser crash on boolean additionalProperties.

    Some gradio/gradio_client combos may raise:
      TypeError: argument of type 'bool' is not iterable
    when parsing JSON schema that contains additionalProperties=true/false.
    """
    if _gradio_client_utils is None:
        return
    if getattr(_gradio_client_utils, "_opengaussian_schema_patch", False):
        return

    orig = _gradio_client_utils._json_schema_to_python_type

    def _safe_json_schema_to_python_type(schema, defs):
        if isinstance(schema, bool):
            # additionalProperties: true/false
            return "Any" if schema else "None"
        return orig(schema, defs)

    _gradio_client_utils._json_schema_to_python_type = _safe_json_schema_to_python_type
    _gradio_client_utils._opengaussian_schema_patch = True


_patch_gradio_schema_parser()


ROOT_DIR = Path(__file__).resolve().parents[1]
TRAIN_PY = ROOT_DIR / "train.py"
KMEANS_SCRIPT = ROOT_DIR / "scripts" / "cluster_semantic_kmeans.py"
SEMANTIC_PIPELINE_SCRIPT = ROOT_DIR / "scripts" / "semantic_instance_pipeline.py"
SEMANTIC_INSTANCE_SCRIPT = ROOT_DIR / "scripts" / "semantic_models_seg3d_instance.py"
SEMANTIC_TO_INSTANCE_CC_SCRIPT = ROOT_DIR / "scripts" / "semantic_to_instance_cc.py"
TOPDOWN_PIPELINE_SCRIPT = ROOT_DIR / "scripts" / "topdown_sam3_instance_pipeline.py"
TOPDOWN_COLUMNCUT_SCRIPT = ROOT_DIR / "scripts" / "topdown_column_cut_instance.py"
WSL_POWERSHELL = Path("/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe")
SAM3_PY = Path("/mnt/d/sam3/.conda/envs/sam3/bin/python")
SAM3_SCRIPT = Path("/mnt/d/sam3/sam3/scripts/batch_sam3_dji_masks.py")
SAM3_IMAGE_SCRIPT = ROOT_DIR / "scripts" / "sam3_instance_from_images.py"
STEP1_KMEANS_SEM_SCRIPT = ROOT_DIR / "scripts" / "step1_kmeans_semantic_rename.py"
SAM3_DEFAULT_CKPT = Path("/mnt/c/Users/ysy/.cache/modelscope/hub/models/facebook/sam3/sam3.pt")
DEFAULT_LITE_XY_JITTER_BASE = 0.22

DEFAULT_ID2LABEL: Dict[int, str] = {
    0: "background",
    1: "vehicle",
    2: "person",
    3: "bicycle",
    4: "vegetation",
    5: "road",
    6: "traffic_facility",
    7: "other",
}


def parse_id2label_text(text: str) -> Dict[int, str]:
    text = (text or "").strip()
    if not text:
        return {}
    out: Dict[int, str] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid id2label item: {item}")
        k, v = item.split(":", 1)
        out[int(k.strip())] = v.strip()
    return out


def resolve_semantic_id_from_text(text: str, id2label_text: str) -> Optional[int]:
    s = (text or "").strip()
    if not s:
        return None
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    try:
        mapping = parse_id2label_text(id2label_text)
    except Exception:
        mapping = {}
    s_low = s.lower()
    for k, v in mapping.items():
        if str(v).strip().lower() == s_low:
            return int(k)
    for k, v in mapping.items():
        if s_low in str(v).strip().lower():
            return int(k)
    return None


def sanitize_name(name: str) -> str:
    s = (name or "").strip()
    if not s:
        return "unknown"
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_\-]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown"


def format_id2label_text(mapping: Dict[int, str]) -> str:
    items = sorted(((int(k), str(v)) for k, v in mapping.items()), key=lambda x: x[0])
    return ",".join(f"{k}:{v}" for k, v in items)


def _load_id2label_candidates(scene_path: Path) -> Optional[Dict[int, str]]:
    candidates = [
        scene_path / "language_features" / "id2label.json",
        scene_path / "language_features" / "id2label.txt",
        scene_path / "id2label.json",
        scene_path / "id2label.txt",
        scene_path / "language_features" / "labels.json",
        scene_path / "labels.json",
    ]
    for p in candidates:
        if not p.exists():
            continue
        try:
            if p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(obj, dict):
                    src = obj.get("id2label", obj)
                    out = {int(k): str(v) for k, v in src.items()}
                    if out:
                        return out
            else:
                out = parse_id2label_text(p.read_text(encoding="utf-8"))
                if out:
                    return out
        except Exception:
            continue
    return None


def _scan_semantic_ids(language_features_dir: Path, suffix: str = "_s.npy") -> List[int]:
    files = sorted(language_features_dir.glob(f"*{suffix}"))
    ids: set[int] = set()
    for p in files:
        try:
            arr = np.load(p)
            if arr.ndim == 3:
                arr = arr[0] if arr.shape[0] > 0 else arr
            if arr.ndim != 2:
                continue
            uniq = np.unique(arr)
            ids.update(int(x) for x in uniq.tolist())
        except Exception:
            continue
    return sorted(ids)


def _load_sam3_priority(scene_path: Path) -> List[str]:
    run_cfg = scene_path / "sam3_masks" / "run_config.json"
    if not run_cfg.exists():
        return []
    try:
        obj = json.loads(run_cfg.read_text(encoding="utf-8"))
    except Exception:
        return []
    pri = obj.get("priority", [])
    if not isinstance(pri, list):
        return []
    out: List[str] = []
    for x in pri:
        s = str(x).strip()
        if s:
            out.append(s)
    return out


def detect_id2label_from_language_features(path_mode: str, source_path: str) -> str:
    scene_path = Path(normalize_path(source_path, mode=path_mode))
    language_features_dir = scene_path / "language_features"
    if not language_features_dir.exists():
        return ""

    meta_mapping = _load_id2label_candidates(scene_path)
    if meta_mapping:
        return format_id2label_text(meta_mapping)

    ids = _scan_semantic_ids(language_features_dir, suffix="_s.npy")
    if not ids:
        return ""

    # Prefer SAM3 class priority if available.
    # In SAM3 outputs, id=0 is background and id=1..N follow priority order.
    sam3_priority = _load_sam3_priority(scene_path)
    if sam3_priority:
        mapping: Dict[int, str] = {}
        for cid in ids:
            if cid == 0:
                mapping[cid] = "background"
                continue
            idx = cid - 1
            if 0 <= idx < len(sam3_priority):
                mapping[cid] = sam3_priority[idx]
            else:
                mapping[cid] = DEFAULT_ID2LABEL.get(cid, f"class_{cid}")
        return format_id2label_text(mapping)

    mapping: Dict[int, str] = {}
    for cid in ids:
        mapping[cid] = DEFAULT_ID2LABEL.get(cid, f"class_{cid}")
    return format_id2label_text(mapping)


def detect_semantic_prompts_for_lite(path_mode: str, source_path: str, fallback_prompt: str = "building") -> str:
    scene_path = Path(normalize_path(source_path, mode=path_mode))

    prompts: List[str] = []
    try:
        txt = detect_id2label_from_language_features(path_mode, source_path)
        if txt:
            mapping = parse_id2label_text(txt)
            for k in sorted(mapping.keys()):
                name = str(mapping[k]).strip()
                if not name:
                    continue
                if int(k) == 0 or name.lower() in {"background", "bg"}:
                    continue
                prompts.append(name)
    except Exception:
        pass

    if not prompts:
        pri = _load_sam3_priority(scene_path)
        for s in pri:
            t = str(s).strip()
            if t and t.lower() not in {"background", "bg"}:
                prompts.append(t)

    # de-duplicate while preserving order
    dedup: List[str] = []
    seen: set[str] = set()
    for p in prompts:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        dedup.append(p)

    if not dedup:
        fb = (fallback_prompt or "building").strip()
        if fb:
            dedup = [fb]
    return ",".join(dedup)


def sync_id2label_for_ui(path_mode: str, source_path: str, current_id2label: str, auto_sync: bool) -> str:
    if not auto_sync:
        return current_id2label
    try:
        auto = detect_id2label_from_language_features(path_mode, source_path)
        if auto:
            return auto
    except Exception:
        pass
    return current_id2label


def normalize_path(p: str, mode: str = "auto") -> str:
    """Normalize input path according to mode.

    mode:
      - auto: detect Windows drive path and convert to WSL style
      - windows: force Windows-drive-to-WSL conversion when possible
      - linux: keep Linux path style (no drive conversion)
    """
    p = (p or "").strip()
    if not p:
        return p

    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m and mode in {"auto", "windows"}:
        drive = m.group(1).lower()
        rest = m.group(2).replace("\\", "/")
        return f"/mnt/{drive}/{rest}"
    return p.replace("\\", "/")


def parse_int_list(text: str) -> List[int]:
    text = (text or "").strip()
    if not text:
        return []
    out: List[int] = []
    for token in re.split(r"[\s,]+", text):
        if not token:
            continue
        out.append(int(token))
    return out


def shell_join(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _slice_if_point_tensor(x: Any, idx: torch.Tensor, n_points: int) -> Any:
    if torch.is_tensor(x) and x.ndim >= 1 and x.shape[0] == n_points:
        use_idx = idx.to(x.device) if x.device != idx.device else idx
        y = x.index_select(0, use_idx).contiguous()
        if isinstance(x, torch.nn.Parameter):
            return torch.nn.Parameter(y.detach(), requires_grad=x.requires_grad)
        if y.requires_grad != x.requires_grad:
            return y.detach().requires_grad_(x.requires_grad)
        return y
    return x


def _to_device_keep_kind(x: Any, device: torch.device) -> Any:
    if not torch.is_tensor(x):
        return x
    y = x.to(device=device, non_blocking=False)
    if isinstance(x, torch.nn.Parameter):
        return torch.nn.Parameter(y, requires_grad=x.requires_grad)
    if y.requires_grad != x.requires_grad:
        return y.detach().requires_grad_(x.requires_grad)
    return y


def _downsample_model_params(
    model_params: Any, max_points: int, target_device: torch.device
) -> Tuple[Any, int, int]:
    if not isinstance(model_params, (list, tuple)) or len(model_params) < 2:
        raise ValueError("Unsupported checkpoint model_params format.")
    if not torch.is_tensor(model_params[1]):
        raise ValueError("Cannot infer point count from model_params[1].")

    n_points = int(model_params[1].shape[0])
    if max_points <= 0 or n_points <= max_points:
        return model_params, n_points, n_points

    g = torch.Generator(device="cpu")
    g.manual_seed(42)
    idx = torch.randperm(n_points, generator=g)[: int(max_points)]
    idx, _ = torch.sort(idx)

    data = list(model_params)
    for i, x in enumerate(data):
        if torch.is_tensor(x):
            data[i] = _to_device_keep_kind(
                _slice_if_point_tensor(x, idx, n_points), target_device
            )

    # Keep optimizer structure but downsample per-point state tensors.
    # For this repo: index 12 (new 14-field) / index 10 (old 12-field).
    opt_idx = 12 if len(data) >= 14 else (10 if len(data) >= 11 else None)
    if opt_idx is not None and isinstance(data[opt_idx], dict):
        opt = data[opt_idx]
        new_opt: dict = {
            "state": {},
            "param_groups": opt.get("param_groups", []),
        }
        for k, st in opt.get("state", {}).items():
            if isinstance(st, dict):
                st_new = {}
                for sk, sv in st.items():
                    sliced = _slice_if_point_tensor(sv, idx, n_points)
                    st_new[sk] = _to_device_keep_kind(sliced, target_device)
                new_opt["state"][k] = st_new
            else:
                new_opt["state"][k] = st
        data[opt_idx] = new_opt

    out = tuple(data) if isinstance(model_params, tuple) else data
    return out, n_points, int(idx.numel())


def maybe_prepare_downsampled_checkpoint(
    path_mode: str,
    start_checkpoint: str,
    model_path: str,
    max_points: int,
) -> Tuple[str, str]:
    max_points = int(max_points or 0)
    ckpt_path = normalize_path(start_checkpoint, mode=path_mode)
    model_path_norm = normalize_path(model_path, mode=path_mode)
    if max_points <= 0 or not ckpt_path:
        return ckpt_path, ""
    p = Path(ckpt_path)
    if not p.exists():
        raise FileNotFoundError(f"start_checkpoint 不存在: {ckpt_path}")

    raw = torch.load(str(p), map_location="cpu")
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_out: Any
    iter_idx = 0
    if isinstance(raw, (list, tuple)) and len(raw) == 2 and isinstance(raw[1], int):
        mp, iter_idx = raw
        mp_ds, old_n, new_n = _downsample_model_params(mp, max_points, target_device)
        ckpt_out = (mp_ds, int(iter_idx))
    elif isinstance(raw, dict) and "model_params" in raw:
        mp = raw["model_params"]
        iter_idx = int(raw.get("iteration", 0))
        mp_ds, old_n, new_n = _downsample_model_params(mp, max_points, target_device)
        ckpt_out = dict(raw)
        ckpt_out["model_params"] = mp_ds
        ckpt_out["iteration"] = int(iter_idx)
    else:
        raise ValueError("不支持的 checkpoint 格式，无法下采样。")

    if new_n >= old_n:
        return ckpt_path, f"[launcher] 下采样未启用：点数 {old_n} <= 上限 {max_points}"

    cache_dir = Path(model_path_norm) / "_launcher_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{p.stem}.ds{new_n}.iter{iter_idx}.pth"
    torch.save(ckpt_out, str(out_path))
    msg = (
        f"[launcher] start_checkpoint 下采样完成: {old_n} -> {new_n} 点\n"
        f"[launcher] downsampled_ckpt={out_path}"
    )
    return str(out_path), msg


def open_browser_windows_from_wsl(url: str) -> bool:
    """Open URL in Windows default browser when running from WSL."""
    if not WSL_POWERSHELL.exists():
        return False
    try:
        subprocess.Popen(
            [
                str(WSL_POWERSHELL),
                "-NoProfile",
                "-Command",
                f"Start-Process '{url}'",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return True
    except Exception:
        return False


def _format_process_metrics(pid: Optional[int]) -> str:
    if pid is None:
        return "CPU: - | RAM: -"
    if not PSUTIL_AVAILABLE:
        return "CPU: psutil未安装 | RAM: psutil未安装"
    try:
        proc = psutil.Process(pid)
        cpu_pct = proc.cpu_percent(interval=None)
        rss_gb = proc.memory_info().rss / (1024**3)
        return f"CPU: {cpu_pct:.1f}% | RAM: {rss_gb:.2f} GB"
    except Exception:
        return "CPU: N/A | RAM: N/A"


def _format_gpu_metrics() -> str:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=1.5,
        )
        rows = [x.strip() for x in out.splitlines() if x.strip()]
        if not rows:
            return "GPU: 未检测到设备"
        parts: List[str] = []
        for i, row in enumerate(rows):
            seg = [s.strip() for s in row.split(",")]
            if len(seg) < 3:
                continue
            util, used, total = seg[0], seg[1], seg[2]
            parts.append(f"GPU{i}: {util}% | 显存 {used}/{total} MiB")
        return " ; ".join(parts) if parts else "GPU: N/A"
    except Exception:
        return "GPU: N/A"


@dataclass
class TrainState:
    process: Optional[subprocess.Popen] = None
    side_process: Optional[subprocess.Popen] = None
    post_running: bool = False
    command: List[str] = field(default_factory=list)
    started_at: float = 0.0
    target_iterations: int = 0
    current_iteration: int = 0
    progress_percent: float = 0.0
    lines: Deque[str] = field(default_factory=lambda: deque(maxlen=6000))
    lock: threading.Lock = field(default_factory=threading.Lock)

    def is_running(self) -> bool:
        train_running = self.process is not None and self.process.poll() is None
        side_running = self.side_process is not None and self.side_process.poll() is None
        return train_running or side_running or self.post_running

    def append(self, line: str) -> None:
        with self.lock:
            self.lines.append(line.rstrip("\n"))

    def tail(self, max_lines: int = 500) -> str:
        with self.lock:
            data = list(self.lines)[-max_lines:]
        return "\n".join(data)

    def set_progress(self, progress: Optional[float] = None, current_iter: Optional[int] = None) -> None:
        with self.lock:
            if current_iter is not None:
                self.current_iteration = max(self.current_iteration, int(current_iter))
            if progress is not None:
                self.progress_percent = max(self.progress_percent, float(progress))
                self.progress_percent = min(self.progress_percent, 100.0)
            if self.target_iterations > 0 and self.current_iteration > 0:
                by_iter = min(100.0, 100.0 * self.current_iteration / float(self.target_iterations))
                self.progress_percent = max(self.progress_percent, by_iter)

    def get_progress(self) -> float:
        with self.lock:
            return float(self.progress_percent)

    def get_current_iter(self) -> int:
        with self.lock:
            return int(self.current_iteration)

    def status_text(self) -> str:
        if self.side_process is not None and self.side_process.poll() is None:
            tag = "后处理" if self.post_running else "子流程"
            return f"{tag}运行中 | PID={self.side_process.pid}"
        if self.post_running:
            return "后处理运行中"
        if self.process is None:
            return "空闲"
        code = self.process.poll()
        if code is None:
            elapsed = int(time.time() - self.started_at)
            return f"运行中 | PID={self.process.pid} | 已运行 {elapsed}s"
        return f"已结束 | 退出码={code}"

    def set_post_running(self, running: bool) -> None:
        with self.lock:
            self.post_running = bool(running)


STATE = TrainState()


def build_command(
    python_exec: str,
    path_mode: str,
    source_path: str,
    model_path: str,
    start_checkpoint: str,
    iterations: int,
    start_ins_feat_iter: int,
    sam_level: int,
    save_iterations_text: str,
    checkpoint_iterations_text: str,
    resolution: str,
    data_device: str,
    eval_mode: bool,
    save_memory: bool,
) -> List[str]:
    py = (python_exec or "").strip() or "python"
    source_path = normalize_path(source_path, mode=path_mode)
    model_path = normalize_path(model_path, mode=path_mode)
    start_checkpoint = normalize_path(start_checkpoint, mode=path_mode)

    cmd = [
        py,
        str(TRAIN_PY),
        "-s",
        source_path,
        "-m",
        model_path,
        "--iterations",
        str(int(iterations)),
        "--start_ins_feat_iter",
        str(int(start_ins_feat_iter)),
        "--sam_level",
        str(int(sam_level)),
    ]

    if start_checkpoint:
        cmd.extend(["--start_checkpoint", start_checkpoint])

    save_iterations = parse_int_list(save_iterations_text)
    if save_iterations:
        cmd.extend(["--save_iterations", *[str(x) for x in save_iterations]])

    checkpoint_iterations = parse_int_list(checkpoint_iterations_text)
    if checkpoint_iterations:
        cmd.extend(["--checkpoint_iterations", *[str(x) for x in checkpoint_iterations]])

    if resolution != "default":
        cmd.extend(["--resolution", str(int(resolution))])
    if data_device in {"cuda", "cpu"}:
        cmd.extend(["--data_device", data_device])
    if eval_mode:
        cmd.append("--eval")
    if save_memory:
        cmd.append("--save_memory")

    return cmd


def build_sam3_command(
    path_mode: str,
    sam3_input_dir: str,
    sam3_output_dir: str,
    sam3_checkpoint: str,
    sam3_device: str,
    source_path: str,
) -> List[str]:
    scene_path = normalize_path(source_path, mode=path_mode)
    input_dir = normalize_path(sam3_input_dir, mode=path_mode)
    output_dir = normalize_path(sam3_output_dir, mode=path_mode) if (sam3_output_dir or "").strip() else ""
    ckpt = normalize_path(sam3_checkpoint, mode=path_mode)

    if not output_dir:
        output_dir = str(Path(scene_path) / "sam3_masks")

    cmd = [
        str(SAM3_PY),
        str(SAM3_SCRIPT),
        "--input-dir",
        input_dir,
        "--output-dir",
        output_dir,
        "--device",
        sam3_device if sam3_device in {"cuda", "cpu"} else "cuda",
    ]
    if ckpt:
        cmd.extend(["--checkpoint-path", ckpt, "--no-hf-download"])
    return cmd


def build_postprocess_commands(
    python_exec: str,
    path_mode: str,
    source_path: str,
    model_path: str,
    post_input_ply: str,
    post_iteration: int,
    post_n_clusters: int,
    post_cluster_output_dir: str,
    post_sem_output_dir: str,
    post_instance_output_dir: str,
    post_vote_ignore_ids: str,
    post_min_votes: int,
    post_min_top1_ratio: float,
    post_id2label: str,
    post_sem_stuff_ids: str,
    post_semantic_only_ids: str,
    post_instance_single_label: str,
    post_instance_stuff_ids: str,
    post_instance_mask_dir: str,
    post_instance_mask_mode: str,
    post_instance_mask_suffix: str,
    post_instance_mask_level: int,
    post_instance_mask_ignore_ids: str,
    run_post_mask2inst: bool,
    post_mask2inst_output_dir: str,
    post_mask2inst_min_area: int,
    post_mask2inst_ignore_ids: str,
    post_instance_vote_stride: int,
    post_instance_min_mask_points: int,
    post_instance_match_iou: float,
    post_instance_min_point_votes: int,
    post_min_instance_points: int,
    post_save_instance_parts: bool,
    run_post_cluster: bool,
    run_post_semantic: bool,
    run_post_instance: bool,
) -> Tuple[List[Tuple[str, List[str]]], Dict[str, str]]:
    py = (python_exec or "").strip() or "python"
    scene_path = normalize_path(source_path, mode=path_mode)
    model_path_norm = normalize_path(model_path, mode=path_mode)

    post_iter = int(post_iteration)
    n_clusters = int(post_n_clusters)

    input_ply = (
        normalize_path(post_input_ply, mode=path_mode)
        if (post_input_ply or "").strip()
        else str(Path(model_path_norm) / "point_cloud" / f"iteration_{post_iter}" / "point_cloud.ply")
    )
    input_ply_dir = str(Path(input_ply).parent)
    cluster_dir = (
        normalize_path(post_cluster_output_dir, mode=path_mode)
        if (post_cluster_output_dir or "").strip()
        else str(Path(input_ply_dir) / f"kmeans_{n_clusters}")
    )
    sem_dir = (
        normalize_path(post_sem_output_dir, mode=path_mode)
        if (post_sem_output_dir or "").strip()
        else str(Path(input_ply_dir) / "sem_stage")
    )
    ins_dir = (
        normalize_path(post_instance_output_dir, mode=path_mode)
        if (post_instance_output_dir or "").strip()
        else str(Path(input_ply_dir) / "sem_ins_proj2d")
    )
    mask2inst_out_dir = (
        normalize_path(post_mask2inst_output_dir, mode=path_mode)
        if (post_mask2inst_output_dir or "").strip()
        else str(Path(input_ply_dir) / "language_features_instance")
    )
    instance_mask_dir = (
        mask2inst_out_dir
        if run_post_mask2inst
        else (
            normalize_path(post_instance_mask_dir, mode=path_mode)
            if (post_instance_mask_dir or "").strip()
            else str(Path(input_ply_dir) / "language_features_instance")
        )
    )
    language_features_dir = str(Path(scene_path) / "language_features")
    effective_semantic_only_ids = (post_semantic_only_ids or "").strip()

    steps: List[Tuple[str, List[str]]] = []

    if run_post_mask2inst:
        steps.append(
            (
                "post-mask2inst",
                [
                    py,
                    str(SEMANTIC_TO_INSTANCE_CC_SCRIPT),
                    "--src_dir",
                    language_features_dir,
                    "--dst_dir",
                    mask2inst_out_dir,
                    "--suffix",
                    "_s.npy",
                    "--out_suffix",
                    "_inst.npy",
                    "--ignore_ids",
                    (post_mask2inst_ignore_ids or "0").strip(),
                    "--min_area",
                    str(int(post_mask2inst_min_area)),
                ],
            )
        )

    if run_post_cluster:
        steps.append(
            (
                "post-kmeans",
                [
                    py,
                    str(KMEANS_SCRIPT),
                    "--input_ply",
                    input_ply,
                    "--output_dir",
                    cluster_dir,
                    "--n_clusters",
                    str(n_clusters),
                    "--assign_full",
                    "--save_npz",
                ],
            )
        )

    if run_post_semantic:
        steps.append(
            (
                "post-semantic",
                [
                    py,
                    str(SEMANTIC_PIPELINE_SCRIPT),
                    "--scene_path",
                    scene_path,
                    "--cluster_dir",
                    cluster_dir,
                    "--output_dir",
                    sem_dir,
                    "--mask_level",
                    "0",
                    "--vote_stride",
                    "1",
                    "--vote_ignore_ids",
                    (post_vote_ignore_ids or "").strip(),
                    "--min_votes",
                    str(int(post_min_votes)),
                    "--min_top1_ratio",
                    str(float(post_min_top1_ratio)),
                    "--id2label",
                    (post_id2label or "").strip(),
                    "--stuff_ids",
                    (post_sem_stuff_ids or "").strip(),
                ],
            )
        )

    if run_post_instance:
        sem_models_dir = str(Path(sem_dir) / "semantic_models")
        single_mode = False
        single_label_text = (post_instance_single_label or "").strip()
        if single_label_text and (post_input_ply or "").strip():
            sem_id = resolve_semantic_id_from_text(single_label_text, post_id2label)
            if sem_id is not None:
                try:
                    id2label_map = parse_id2label_text(post_id2label)
                except Exception:
                    id2label_map = {}
                sem_label = id2label_map.get(sem_id, single_label_text)
                sem_safe = sanitize_name(sem_label)
                single_root = Path(ins_dir) / "_single_semantic_models"
                single_dir = single_root / f"{int(sem_id):03d}_{sem_safe}"
                dst_ply = single_dir / "merged_semantic.ply"
                prep_py = (
                    "from pathlib import Path; import shutil; "
                    f"src=Path({input_ply!r}); dst=Path({str(dst_ply)!r}); "
                    "dst.parent.mkdir(parents=True, exist_ok=True); "
                    "shutil.copy2(src, dst); "
                    "print(f'[single-semantic] copied: {src} -> {dst}')"
                )
                steps.append(("post-single-semantic-prepare", [py, "-c", prep_py]))
                sem_models_dir = str(single_root)
                effective_semantic_only_ids = str(int(sem_id))
                single_mode = True

        cmd = [
            py,
            str(SEMANTIC_INSTANCE_SCRIPT),
            "--scene_path",
            scene_path,
            "--semantic_models_dir",
            sem_models_dir,
            "--output_dir",
            ins_dir,
            "--semantic_only_ids",
            effective_semantic_only_ids,
            "--stuff_ids",
            (post_instance_stuff_ids or "").strip(),
            "--instance_mask_dir",
            instance_mask_dir,
            "--instance_mask_mode",
            post_instance_mask_mode if post_instance_mask_mode in {"name", "index"} else "name",
            "--instance_mask_suffix",
            (post_instance_mask_suffix or "_inst.npy").strip(),
            "--instance_mask_level",
            str(int(post_instance_mask_level)),
            "--instance_mask_ignore_ids",
            (post_instance_mask_ignore_ids or "0").strip(),
            "--vote_stride",
            str(int(post_instance_vote_stride)),
            "--instance_min_mask_points",
            str(int(post_instance_min_mask_points)),
            "--instance_match_iou",
            str(float(post_instance_match_iou)),
            "--instance_min_point_votes",
            str(int(post_instance_min_point_votes)),
            "--min_instance_points",
            str(int(post_min_instance_points)),
        ]
        if post_save_instance_parts:
            cmd.append("--save_instance_parts")
        if single_mode:
            cmd.extend(["--merged_name", "merged_semantic.ply"])
        steps.append(("post-instance", cmd))

    meta = {
        "scene_path": scene_path,
        "model_path": model_path_norm,
        "input_ply": input_ply,
        "language_features_dir": language_features_dir,
        "mask2inst_out_dir": mask2inst_out_dir,
        "cluster_dir": cluster_dir,
        "sem_dir": sem_dir,
        "ins_dir": ins_dir,
        "instance_mask_dir": instance_mask_dir,
    }
    return steps, meta


def build_postprocess_lite_commands(
    python_exec: str,
    path_mode: str,
    source_path: str,
    model_path: str,
    post_iteration: int,
    post_lite_input_ply: str,
    post_lite_output_subdir: str,
    post_lite_num_views: int,
    post_lite_image_size: int,
    post_lite_fov_deg: float,
    post_lite_xoy_step_multiplier: float,
    post_lite_semantic_prompts: str,
    post_lite_semantic_probe_views: int,
    post_lite_sam_prompt: str,
    post_lite_semantic_id: int,
    post_lite_min_instance_points: int,
    post_lite_save_instance_parts: bool,
    sam3_checkpoint: str,
    sam3_device: str,
    post_lite_mask_image: str = "auto",
) -> Tuple[List[Tuple[str, List[str]]], Dict[str, str]]:
    py = (python_exec or "").strip() or "python"
    scene_path = normalize_path(source_path, mode=path_mode)
    model_path_norm = normalize_path(model_path, mode=path_mode)

    input_ply = (
        normalize_path(post_lite_input_ply, mode=path_mode)
        if (post_lite_input_ply or "").strip()
        else str(Path(model_path_norm) / "point_cloud" / f"iteration_{int(post_iteration)}" / "point_cloud.ply")
    )
    output_subdir = (post_lite_output_subdir or "topdown_instance_pipeline").strip()
    sam3_ckpt_norm = (
        normalize_path(sam3_checkpoint, mode=path_mode)
        if (sam3_checkpoint or "").strip()
        else str(SAM3_DEFAULT_CKPT)
    )
    pipe_root = str(Path(scene_path) / output_subdir)
    render_dir = str(Path(pipe_root) / "renders_topdown")
    semantic_probe_dir = str(Path(pipe_root) / "semantic_probe")
    semantic_probe_summary = str(Path(semantic_probe_dir) / "summary.json")
    pose_json = str(Path(scene_path) / "render_view_poses_topdown" / "topdown_camera_poses.json")
    instance_mask_dir = str(Path(pipe_root) / "sam3_instance" / "instance_index_npy")
    column_out_dir = str(Path(pipe_root) / "instance_3d_columncut")

    xy_jitter_ratio = max(0.0, min(1.0, float(DEFAULT_LITE_XY_JITTER_BASE) * float(post_lite_xoy_step_multiplier)))

    step1_cmd = [
        py,
        str(TOPDOWN_PIPELINE_SCRIPT),
        "--scene_path",
        scene_path,
        "--input_ply",
        input_ply,
        "--output_subdir",
        output_subdir,
        "--num_views",
        str(int(post_lite_num_views)),
        "--image_size",
        str(int(post_lite_image_size)),
        "--fov_deg",
        str(float(post_lite_fov_deg)),
        "--xy_jitter_ratio",
        str(float(xy_jitter_ratio)),
        "--sam3_python",
        str(SAM3_PY),
        "--sam3_checkpoint",
        sam3_ckpt_norm,
        "--sam3_device",
        sam3_device if sam3_device in {"cuda", "cpu"} else "cuda",
        "--sam3_prompts",
        (post_lite_sam_prompt or "building").strip(),
        "--semantic_id",
        str(int(post_lite_semantic_id)),
        "--render_only",
    ]

    step2_cmd = [
        py,
        str(TOPDOWN_PIPELINE_SCRIPT),
        "--scene_path",
        scene_path,
        "--input_ply",
        input_ply,
        "--output_subdir",
        output_subdir,
        "--num_views",
        str(int(post_lite_num_views)),
        "--image_size",
        str(int(post_lite_image_size)),
        "--fov_deg",
        str(float(post_lite_fov_deg)),
        "--xy_jitter_ratio",
        str(float(xy_jitter_ratio)),
        "--sam3_python",
        str(SAM3_PY),
        "--sam3_checkpoint",
        sam3_ckpt_norm,
        "--sam3_device",
        sam3_device if sam3_device in {"cuda", "cpu"} else "cuda",
        "--sam3_prompts",
        (post_lite_sam_prompt or "building").strip(),
        "--semantic_id",
        str(int(post_lite_semantic_id)),
        "--skip_render",
        "--semantic_probe_summary",
        semantic_probe_summary,
        "--semantic_probe_fallback_prompts",
        (post_lite_sam_prompt or "building").strip(),
    ]
    if post_lite_save_instance_parts:
        step2_cmd.append("--save_instance_parts")

    semantic_prompts = (post_lite_semantic_prompts or "").strip()
    if (not semantic_prompts) or semantic_prompts.lower() == "auto":
        semantic_prompts = detect_semantic_prompts_for_lite(
            path_mode=path_mode,
            source_path=source_path,
            fallback_prompt=(post_lite_sam_prompt or "building").strip(),
        )
    step_probe_cmd = [
        str(SAM3_PY),
        str(SAM3_IMAGE_SCRIPT),
        "--input-dir",
        render_dir,
        "--output-dir",
        semantic_probe_dir,
        "--device",
        sam3_device if sam3_device in {"cuda", "cpu"} else "cuda",
        "--resolution",
        "1008",
        "--confidence-threshold",
        "0.35",
        "--prompts",
        semantic_prompts,
        "--min-mask-area",
        "40",
        "--checkpoint-path",
        sam3_ckpt_norm,
        "--no-hf-download",
    ]
    if int(post_lite_semantic_probe_views) > 0:
        step_probe_cmd.extend(["--max-images", str(int(post_lite_semantic_probe_views))])

    step3_cmd = [
        py,
        str(TOPDOWN_COLUMNCUT_SCRIPT),
        "--input_ply",
        input_ply,
        "--pose_json",
        pose_json,
        "--instance_mask_dir",
        instance_mask_dir,
        "--output_dir",
        column_out_dir,
        "--semantic_id",
        str(int(post_lite_semantic_id)),
        "--mode",
        "all_views_fused",
        "--fused_instance_npy",
        str(Path(pipe_root) / "instance_3d" / "point_instance_id.npy"),
        "--mask_image",
        (post_lite_mask_image or "auto").strip(),
        "--xoy_stride_multiplier",
        str(float(post_lite_xoy_step_multiplier)),
        "--min_instance_points",
        str(int(post_lite_min_instance_points)),
    ]
    if post_lite_save_instance_parts:
        step3_cmd.append("--save_instance_parts")

    steps = [
        ("post-lite-render-only", step1_cmd),
        ("post-lite-semantic-probe", step_probe_cmd),
        ("post-lite-instance-from-renders", step2_cmd),
        ("post-lite-column-cut", step3_cmd),
    ]
    meta = {
        "scene_path": scene_path,
        "model_path": model_path_norm,
        "input_ply": input_ply,
        "pipeline_root": pipe_root,
        "render_dir": render_dir,
        "semantic_probe_dir": semantic_probe_dir,
        "semantic_probe_summary": semantic_probe_summary,
        "pose_json": pose_json,
        "instance_mask_dir": instance_mask_dir,
        "column_out_dir": column_out_dir,
    }
    return steps, meta


def run_blocking_with_logs(cmd: List[str], prefix: str, track_side_process: bool = False) -> int:
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=os.environ.copy(),
    )
    if track_side_process:
        STATE.side_process = proc
    assert proc.stdout is not None
    for line in proc.stdout:
        STATE.append(f"[{prefix}] {line.rstrip()}")
    code = proc.wait()
    if track_side_process:
        STATE.side_process = None
    return code


def run_postprocess_steps(post_steps: List[Tuple[str, List[str]]]) -> bool:
    if not post_steps:
        return True
    STATE.set_post_running(True)
    ok = True
    try:
        for prefix, cmd in post_steps:
            STATE.append(f"[launcher] {prefix}_cmd={shell_join(cmd)}")
            code = run_blocking_with_logs(cmd, prefix, track_side_process=True)
            if code != 0:
                STATE.append(f"[launcher] {prefix} 失败，退出码={code}")
                ok = False
                break
    finally:
        STATE.set_post_running(False)
    if ok:
        STATE.append("[launcher] 后处理完成")
    return ok


def run_postprocess_lite_split(
    python_exec: str,
    path_mode: str,
    source_path: str,
    model_path: str,
    post_iteration: int,
    post_lite_input_ply: str,
    post_lite_output_subdir: str,
    post_lite_num_views: int,
    post_lite_image_size: int,
    post_lite_fov_deg: float,
    post_lite_xoy_step_multiplier: float,
    post_lite_semantic_prompts: str,
    post_lite_semantic_probe_views: int,
    post_lite_sam_prompt: str,
    post_lite_semantic_id: int,
    post_lite_min_instance_points: int,
    post_lite_save_instance_parts: bool,
    sam3_checkpoint: str,
    sam3_device: str,
    mode: str,
    post_lite_mask_image: str = "auto",
) -> Tuple[str, str]:
    if STATE.is_running():
        return "训练或后处理正在运行", STATE.tail()
    if mode not in {"render_only", "instance_from_renders"}:
        return f"未知模式: {mode}", STATE.tail()

    lite_steps, _ = build_postprocess_lite_commands(
        python_exec=python_exec,
        path_mode=path_mode,
        source_path=source_path,
        model_path=model_path,
        post_iteration=int(post_iteration),
        post_lite_input_ply=post_lite_input_ply,
        post_lite_output_subdir=post_lite_output_subdir,
        post_lite_num_views=int(post_lite_num_views),
        post_lite_image_size=int(post_lite_image_size),
        post_lite_fov_deg=float(post_lite_fov_deg),
        post_lite_xoy_step_multiplier=float(post_lite_xoy_step_multiplier),
        post_lite_semantic_prompts=post_lite_semantic_prompts,
        post_lite_semantic_probe_views=int(post_lite_semantic_probe_views),
        post_lite_sam_prompt=post_lite_sam_prompt,
        post_lite_semantic_id=int(post_lite_semantic_id),
        post_lite_min_instance_points=int(post_lite_min_instance_points),
        post_lite_save_instance_parts=bool(post_lite_save_instance_parts),
        sam3_checkpoint=sam3_checkpoint,
        sam3_device=sam3_device,
        post_lite_mask_image=post_lite_mask_image,
    )
    if mode == "render_only":
        steps = lite_steps[:1]
        STATE.append("[launcher] 运行精简后处理: 仅生成渲染图")
    else:
        steps = lite_steps[1:]
        STATE.append("[launcher] 运行精简后处理: 基于当前渲染继续实例分割")
    ok = run_postprocess_steps(steps)
    return ("完成" if ok else "失败"), STATE.tail()


def run_postprocess_lite_step1_kmeans_semantic(
    python_exec: str,
    path_mode: str,
    source_path: str,
    post_lite_kmeans_dir: str,
    post_lite_kmeans_sem_output_dir: str,
    post_lite_semantic_prompts: str,
    post_lite_semantic_probe_views: int,
    post_lite_num_views: int,
    post_lite_image_size: int,
    post_lite_fov_deg: float,
    post_lite_xoy_step_multiplier: float,
    sam3_checkpoint: str,
    sam3_device: str,
    fallback_prompt: str = "building",
) -> Tuple[str, str]:
    if STATE.is_running():
        return "训练或后处理正在运行", STATE.tail()

    scene_path = normalize_path(source_path, mode=path_mode)
    kmeans_dir = normalize_path(post_lite_kmeans_dir, mode=path_mode)
    out_dir = normalize_path(post_lite_kmeans_sem_output_dir, mode=path_mode)
    if not out_dir.strip():
        out_dir = str(Path(kmeans_dir) / "semantic_named")

    prompts = (post_lite_semantic_prompts or "").strip()
    if (not prompts) or prompts.lower() == "auto":
        prompts = detect_semantic_prompts_for_lite(path_mode, source_path, fallback_prompt=fallback_prompt)

    xy_jitter_ratio = max(0.0, min(1.0, float(DEFAULT_LITE_XY_JITTER_BASE) * float(post_lite_xoy_step_multiplier)))
    ckpt = normalize_path(sam3_checkpoint, mode=path_mode) if (sam3_checkpoint or "").strip() else str(SAM3_DEFAULT_CKPT)

    cmd = [
        (python_exec or "").strip() or "python",
        str(STEP1_KMEANS_SEM_SCRIPT),
        "--scene_path",
        scene_path,
        "--kmeans_dir",
        kmeans_dir,
        "--output_dir",
        out_dir,
        "--python_exec",
        (python_exec or "").strip() or "python",
        "--sam3_python",
        str(SAM3_PY),
        "--sam3_checkpoint",
        ckpt,
        "--sam3_device",
        sam3_device if sam3_device in {"cuda", "cpu"} else "cuda",
        "--semantic_prompts",
        prompts,
        "--semantic_probe_views",
        str(int(post_lite_semantic_probe_views)),
        "--num_views",
        str(int(post_lite_num_views)),
        "--image_size",
        str(int(post_lite_image_size)),
        "--fov_deg",
        str(float(post_lite_fov_deg)),
        "--xy_jitter_ratio",
        str(float(xy_jitter_ratio)),
    ]

    ok = run_postprocess_steps([("post-lite-step1-kmeans-semantic", cmd)])
    return ("完成" if ok else "失败"), STATE.tail()


def export_sam3_to_language_features(path_mode: str, source_path: str, sam3_output_dir: str) -> int:
    scene_path = Path(normalize_path(source_path, mode=path_mode))
    output_dir = Path(normalize_path(sam3_output_dir, mode=path_mode))
    src_dir = output_dir / "semantic_index_npy"
    dst_dir = scene_path / "language_features"
    if not src_dir.exists():
        raise FileNotFoundError(f"SAM3输出目录不存在: {src_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for p in sorted(src_dir.glob("*.npy")):
        dst = dst_dir / f"{p.stem}_s.npy"
        shutil.copy2(p, dst)
        copied += 1
    return copied


def _update_progress_from_text(text: str) -> None:
    # tqdm progress, e.g. "... 37%|..."
    m = re.search(r"(\d{1,3})%\|", text)
    if m:
        STATE.set_progress(progress=float(m.group(1)))

    # iteration marker from train.py, e.g. "[ITER 40000]"
    m = re.search(r"\[ITER\s+(\d+)\]", text)
    if m:
        STATE.set_progress(current_iter=int(m.group(1)))


def _spawn_reader(proc: subprocess.Popen, post_steps: Optional[List[Tuple[str, List[str]]]] = None) -> None:
    def _reader() -> None:
        assert proc.stdout is not None
        pending = ""
        while True:
            chunk = proc.stdout.read(256)
            if chunk == "":
                break
            text = pending + chunk
            parts = re.split(r"[\r\n]+", text)
            pending = parts.pop() if parts else ""
            for seg in parts:
                seg = seg.strip()
                if not seg:
                    continue
                STATE.append(seg)
                _update_progress_from_text(seg)
        if pending.strip():
            STATE.append(pending.strip())
            _update_progress_from_text(pending.strip())
        code = proc.wait()
        if code == 0:
            STATE.set_progress(progress=100.0)
        STATE.append(f"[train.py exited] code={code}")
        if code == 0 and post_steps:
            STATE.append("[launcher] 训练已完成，开始执行后处理流水线...")
            run_postprocess_steps(post_steps)

    t = threading.Thread(target=_reader, daemon=True)
    t.start()


def preview_command(
    python_exec: str,
    path_mode: str,
    source_path: str,
    model_path: str,
    start_checkpoint: str,
    iterations: int,
    start_ins_feat_iter: int,
    sam_level: int,
    save_iterations_text: str,
    checkpoint_iterations_text: str,
    resolution: str,
    checkpoint_max_points: int,
    data_device: str,
    eval_mode: bool,
    save_memory: bool,
    run_sam3: bool,
    run_training: bool,
    sam3_input_dir: str,
    sam3_output_dir: str,
    sam3_checkpoint: str,
    sam3_device: str,
    sam3_export_to_language_features: bool,
    post_auto_sync_id2label: bool,
    run_postprocess: bool,
    run_post_mask2inst: bool,
    post_mask2inst_output_dir: str,
    post_mask2inst_min_area: int,
    post_mask2inst_ignore_ids: str,
    run_post_cluster: bool,
    run_post_semantic: bool,
    run_post_instance: bool,
    post_iteration: int,
    post_input_ply: str,
    post_n_clusters: int,
    post_cluster_output_dir: str,
    post_sem_output_dir: str,
    post_instance_output_dir: str,
    post_vote_ignore_ids: str,
    post_min_votes: int,
    post_min_top1_ratio: float,
    post_id2label: str,
    post_sem_stuff_ids: str,
    post_semantic_only_ids: str,
    post_instance_single_label: str,
    post_instance_stuff_ids: str,
    post_instance_mask_dir: str,
    post_instance_mask_mode: str,
    post_instance_mask_suffix: str,
    post_instance_mask_level: int,
    post_instance_mask_ignore_ids: str,
    post_instance_vote_stride: int,
    post_instance_min_mask_points: int,
    post_instance_match_iou: float,
    post_instance_min_point_votes: int,
    post_min_instance_points: int,
    post_save_instance_parts: bool,
    run_postprocess_lite: bool = False,
    post_lite_input_ply: str = "",
    post_lite_output_subdir: str = "topdown_instance_pipeline",
    post_lite_num_views: int = 6,
    post_lite_image_size: int = 1024,
    post_lite_fov_deg: float = 58.0,
    post_lite_xoy_step_multiplier: float = 1.0,
    post_lite_semantic_prompts: str = "building,vehicle,person,bicycle,vegetation,road,traffic_facility,other",
    post_lite_semantic_probe_views: int = 6,
    post_lite_sam_prompt: str = "building",
    post_lite_semantic_id: int = 3,
    post_lite_min_instance_points: int = 3000,
    post_lite_save_instance_parts: bool = True,
):
    if not run_sam3 and not run_training and not run_postprocess and not run_postprocess_lite:
        return "请至少勾选一个流程：跑SAM3 / 跑训练 / 跑后处理 / 跑后处理精简版"

    train_cmd = build_command(
        python_exec,
        path_mode,
        source_path,
        model_path,
        start_checkpoint,
        iterations,
        start_ins_feat_iter,
        sam_level,
        save_iterations_text,
        checkpoint_iterations_text,
        resolution,
        data_device,
        eval_mode,
        save_memory,
    )

    ds_note = ""
    if run_training and (start_checkpoint or "").strip() and int(checkpoint_max_points or 0) > 0:
        ds_note = f"# 运行前会将 start_checkpoint 下采样到 <= {int(checkpoint_max_points)} 点\n"

    blocks: List[str] = []

    if run_sam3:
        sam3_cmd = build_sam3_command(
            path_mode=path_mode,
            sam3_input_dir=sam3_input_dir,
            sam3_output_dir=sam3_output_dir,
            sam3_checkpoint=sam3_checkpoint,
            sam3_device=sam3_device,
            source_path=source_path,
        )
        extra = ""
        if sam3_export_to_language_features:
            scene = normalize_path(source_path, mode=path_mode)
            out_dir = normalize_path(sam3_output_dir, mode=path_mode) if (sam3_output_dir or "").strip() else str(Path(scene) / "sam3_masks")
            extra = (
                "\n# SAM3输出转OpenGaussian格式\n"
                f"# copy {out_dir}/semantic_index_npy/*.npy -> {scene}/language_features/*_s.npy"
            )
        blocks.append(f"[SAM3]\n{shell_join(sam3_cmd)}{extra}")

    if run_training:
        blocks.append(f"[TRAIN]\n{ds_note}{shell_join(train_cmd)}")

    if run_postprocess:
        if post_auto_sync_id2label:
            auto_text = detect_id2label_from_language_features(path_mode, source_path)
            if auto_text:
                post_id2label = auto_text

        missing_scripts: List[str] = []
        if run_post_mask2inst and not SEMANTIC_TO_INSTANCE_CC_SCRIPT.exists():
            missing_scripts.append(str(SEMANTIC_TO_INSTANCE_CC_SCRIPT))
        if run_post_cluster and not KMEANS_SCRIPT.exists():
            missing_scripts.append(str(KMEANS_SCRIPT))
        if run_post_semantic and not SEMANTIC_PIPELINE_SCRIPT.exists():
            missing_scripts.append(str(SEMANTIC_PIPELINE_SCRIPT))
        if run_post_instance and not SEMANTIC_INSTANCE_SCRIPT.exists():
            missing_scripts.append(str(SEMANTIC_INSTANCE_SCRIPT))
        if missing_scripts:
            lines = ["[POST]", "# 后处理脚本缺失："]
            for p in missing_scripts:
                lines.append(f"# - {p}")
            blocks.append("\n".join(lines))
            return "\n\n".join(blocks)
        post_steps, post_meta = build_postprocess_commands(
            python_exec=python_exec,
            path_mode=path_mode,
            source_path=source_path,
            model_path=model_path,
            post_iteration=int(post_iteration),
            post_input_ply=post_input_ply,
            post_n_clusters=int(post_n_clusters),
            post_cluster_output_dir=post_cluster_output_dir,
            post_sem_output_dir=post_sem_output_dir,
            post_instance_output_dir=post_instance_output_dir,
            post_vote_ignore_ids=post_vote_ignore_ids,
            post_min_votes=int(post_min_votes),
            post_min_top1_ratio=float(post_min_top1_ratio),
            post_id2label=post_id2label,
            post_sem_stuff_ids=post_sem_stuff_ids,
            post_semantic_only_ids=post_semantic_only_ids,
            post_instance_single_label=post_instance_single_label,
            post_instance_stuff_ids=post_instance_stuff_ids,
            post_instance_mask_dir=post_instance_mask_dir,
            post_instance_mask_mode=post_instance_mask_mode,
            post_instance_mask_suffix=post_instance_mask_suffix,
            post_instance_mask_level=int(post_instance_mask_level),
            post_instance_mask_ignore_ids=post_instance_mask_ignore_ids,
            run_post_mask2inst=bool(run_post_mask2inst),
            post_mask2inst_output_dir=post_mask2inst_output_dir,
            post_mask2inst_min_area=int(post_mask2inst_min_area),
            post_mask2inst_ignore_ids=post_mask2inst_ignore_ids,
            post_instance_vote_stride=int(post_instance_vote_stride),
            post_instance_min_mask_points=int(post_instance_min_mask_points),
            post_instance_match_iou=float(post_instance_match_iou),
            post_instance_min_point_votes=int(post_instance_min_point_votes),
            post_min_instance_points=int(post_min_instance_points),
            post_save_instance_parts=bool(post_save_instance_parts),
            run_post_cluster=bool(run_post_cluster),
            run_post_semantic=bool(run_post_semantic),
            run_post_instance=bool(run_post_instance),
        )
        if not post_steps:
            blocks.append("[POST]\n# 未选择后处理子步骤（聚类/语义/实例）")
        else:
            cmd_lines = [f"# input_ply={post_meta['input_ply']}"]
            for name, c in post_steps:
                cmd_lines.append(f"# {name}")
                cmd_lines.append(shell_join(c))
            blocks.append("[POST]\n" + "\n".join(cmd_lines))

    if run_postprocess_lite:
        missing_scripts: List[str] = []
        if not TOPDOWN_PIPELINE_SCRIPT.exists():
            missing_scripts.append(str(TOPDOWN_PIPELINE_SCRIPT))
        if not TOPDOWN_COLUMNCUT_SCRIPT.exists():
            missing_scripts.append(str(TOPDOWN_COLUMNCUT_SCRIPT))
        if not SAM3_IMAGE_SCRIPT.exists():
            missing_scripts.append(str(SAM3_IMAGE_SCRIPT))
        if missing_scripts:
            lines = ["[POST-LITE]", "# 后处理精简版脚本缺失："]
            for p in missing_scripts:
                lines.append(f"# - {p}")
            blocks.append("\n".join(lines))
            return "\n\n".join(blocks)

        lite_steps, lite_meta = build_postprocess_lite_commands(
            python_exec=python_exec,
            path_mode=path_mode,
            source_path=source_path,
            model_path=model_path,
            post_iteration=int(post_iteration),
            post_lite_input_ply=post_lite_input_ply,
            post_lite_output_subdir=post_lite_output_subdir,
            post_lite_num_views=int(post_lite_num_views),
            post_lite_image_size=int(post_lite_image_size),
            post_lite_fov_deg=float(post_lite_fov_deg),
            post_lite_xoy_step_multiplier=float(post_lite_xoy_step_multiplier),
            post_lite_semantic_prompts=post_lite_semantic_prompts,
            post_lite_semantic_probe_views=int(post_lite_semantic_probe_views),
            post_lite_sam_prompt=post_lite_sam_prompt,
            post_lite_semantic_id=int(post_lite_semantic_id),
            post_lite_min_instance_points=int(post_lite_min_instance_points),
            post_lite_save_instance_parts=bool(post_lite_save_instance_parts),
            sam3_checkpoint=sam3_checkpoint,
            sam3_device=sam3_device,
        )
        cmd_lines = [
            f"# input_ply={lite_meta['input_ply']}",
            f"# pose_json={lite_meta['pose_json']}",
            f"# instance_mask_dir={lite_meta['instance_mask_dir']}",
        ]
        for name, c in lite_steps:
            cmd_lines.append(f"# {name}")
            cmd_lines.append(shell_join(c))
        blocks.append("[POST-LITE]\n" + "\n".join(cmd_lines))

    return "\n\n".join(blocks)


def start_training(
    python_exec: str,
    path_mode: str,
    source_path: str,
    model_path: str,
    start_checkpoint: str,
    iterations: int,
    start_ins_feat_iter: int,
    sam_level: int,
    save_iterations_text: str,
    checkpoint_iterations_text: str,
    resolution: str,
    checkpoint_max_points: int,
    data_device: str,
    eval_mode: bool,
    save_memory: bool,
    run_sam3: bool,
    run_training: bool,
    sam3_input_dir: str,
    sam3_output_dir: str,
    sam3_checkpoint: str,
    sam3_device: str,
    sam3_export_to_language_features: bool,
    post_auto_sync_id2label: bool,
    run_postprocess: bool,
    run_post_mask2inst: bool,
    post_mask2inst_output_dir: str,
    post_mask2inst_min_area: int,
    post_mask2inst_ignore_ids: str,
    run_post_cluster: bool,
    run_post_semantic: bool,
    run_post_instance: bool,
    post_iteration: int,
    post_input_ply: str,
    post_n_clusters: int,
    post_cluster_output_dir: str,
    post_sem_output_dir: str,
    post_instance_output_dir: str,
    post_vote_ignore_ids: str,
    post_min_votes: int,
    post_min_top1_ratio: float,
    post_id2label: str,
    post_sem_stuff_ids: str,
    post_semantic_only_ids: str,
    post_instance_single_label: str,
    post_instance_stuff_ids: str,
    post_instance_mask_dir: str,
    post_instance_mask_mode: str,
    post_instance_mask_suffix: str,
    post_instance_mask_level: int,
    post_instance_mask_ignore_ids: str,
    post_instance_vote_stride: int,
    post_instance_min_mask_points: int,
    post_instance_match_iou: float,
    post_instance_min_point_votes: int,
    post_min_instance_points: int,
    post_save_instance_parts: bool,
    run_postprocess_lite: bool = False,
    post_lite_input_ply: str = "",
    post_lite_output_subdir: str = "topdown_instance_pipeline",
    post_lite_num_views: int = 6,
    post_lite_image_size: int = 1024,
    post_lite_fov_deg: float = 58.0,
    post_lite_xoy_step_multiplier: float = 1.0,
    post_lite_semantic_prompts: str = "building,vehicle,person,bicycle,vegetation,road,traffic_facility,other",
    post_lite_semantic_probe_views: int = 6,
    post_lite_sam_prompt: str = "building",
    post_lite_semantic_id: int = 3,
    post_lite_min_instance_points: int = 3000,
    post_lite_save_instance_parts: bool = True,
):
    if STATE.is_running():
        return (
            STATE.status_text(),
            STATE.tail(),
            shell_join(STATE.command),
            STATE.get_progress(),
            STATE.get_current_iter(),
            get_monitor_text(),
        )

    if not run_sam3 and not run_training and not run_postprocess and not run_postprocess_lite:
        return (
            "空闲（未选择流程）",
            "请至少勾选一个流程：跑SAM3 / 跑训练 / 跑后处理 / 跑后处理精简版",
            "",
            0.0,
            0,
            get_monitor_text(),
        )

    with STATE.lock:
        STATE.lines.clear()

    effective_start_checkpoint = start_checkpoint
    if run_training and (start_checkpoint or "").strip() and int(checkpoint_max_points or 0) > 0:
        try:
            effective_start_checkpoint, ds_msg = maybe_prepare_downsampled_checkpoint(
                path_mode=path_mode,
                start_checkpoint=start_checkpoint,
                model_path=model_path,
                max_points=int(checkpoint_max_points),
            )
            if ds_msg:
                STATE.append(ds_msg)
        except Exception as exc:
            STATE.append(f"[launcher] start_checkpoint 下采样失败: {exc}")
            return (
                "空闲（下采样失败）",
                STATE.tail(),
                "",
                0.0,
                0,
                get_monitor_text(),
            )

    cmd = build_command(
        python_exec,
        path_mode,
        source_path,
        model_path,
        effective_start_checkpoint,
        iterations,
        start_ins_feat_iter,
        sam_level,
        save_iterations_text,
        checkpoint_iterations_text,
        resolution,
        data_device,
        eval_mode,
        save_memory,
    )
    post_steps: List[Tuple[str, List[str]]] = []
    post_meta: Dict[str, str] = {}
    if run_postprocess:
        if post_auto_sync_id2label:
            auto_text = detect_id2label_from_language_features(path_mode, source_path)
            if auto_text:
                post_id2label = auto_text

        missing_scripts: List[str] = []
        if run_post_mask2inst and not SEMANTIC_TO_INSTANCE_CC_SCRIPT.exists():
            missing_scripts.append(str(SEMANTIC_TO_INSTANCE_CC_SCRIPT))
        if run_post_cluster and not KMEANS_SCRIPT.exists():
            missing_scripts.append(str(KMEANS_SCRIPT))
        if run_post_semantic and not SEMANTIC_PIPELINE_SCRIPT.exists():
            missing_scripts.append(str(SEMANTIC_PIPELINE_SCRIPT))
        if run_post_instance and not SEMANTIC_INSTANCE_SCRIPT.exists():
            missing_scripts.append(str(SEMANTIC_INSTANCE_SCRIPT))
        if missing_scripts:
            STATE.append("[launcher] 后处理脚本缺失:")
            for p in missing_scripts:
                STATE.append(f"[launcher]   - {p}")
            return (
                "空闲（后处理脚本缺失）",
                STATE.tail(),
                shell_join(cmd) if run_training else "",
                0.0,
                0,
                get_monitor_text(),
            )

    post_lite_steps: List[Tuple[str, List[str]]] = []
    post_lite_meta: Dict[str, str] = {}
    if run_postprocess_lite:
        missing_scripts: List[str] = []
        if not TOPDOWN_PIPELINE_SCRIPT.exists():
            missing_scripts.append(str(TOPDOWN_PIPELINE_SCRIPT))
        if not TOPDOWN_COLUMNCUT_SCRIPT.exists():
            missing_scripts.append(str(TOPDOWN_COLUMNCUT_SCRIPT))
        if not SAM3_PY.exists():
            missing_scripts.append(str(SAM3_PY))
        lite_ckpt_norm = (
            normalize_path(sam3_checkpoint, mode=path_mode)
            if (sam3_checkpoint or "").strip()
            else str(SAM3_DEFAULT_CKPT)
        )
        if not Path(lite_ckpt_norm).exists():
            missing_scripts.append(str(lite_ckpt_norm))
        if missing_scripts:
            STATE.append("[launcher] 后处理精简版依赖缺失:")
            for p in missing_scripts:
                STATE.append(f"[launcher]   - {p}")
            return (
                "空闲（后处理精简版脚本缺失）",
                STATE.tail(),
                shell_join(cmd) if run_training else "",
                0.0,
                0,
                get_monitor_text(),
            )

        post_lite_steps, post_lite_meta = build_postprocess_lite_commands(
            python_exec=python_exec,
            path_mode=path_mode,
            source_path=source_path,
            model_path=model_path,
            post_iteration=int(post_iteration),
            post_lite_input_ply=post_lite_input_ply,
            post_lite_output_subdir=post_lite_output_subdir,
            post_lite_num_views=int(post_lite_num_views),
            post_lite_image_size=int(post_lite_image_size),
            post_lite_fov_deg=float(post_lite_fov_deg),
            post_lite_xoy_step_multiplier=float(post_lite_xoy_step_multiplier),
            post_lite_semantic_prompts=post_lite_semantic_prompts,
            post_lite_semantic_probe_views=int(post_lite_semantic_probe_views),
            post_lite_sam_prompt=post_lite_sam_prompt,
            post_lite_semantic_id=int(post_lite_semantic_id),
            post_lite_min_instance_points=int(post_lite_min_instance_points),
            post_lite_save_instance_parts=bool(post_lite_save_instance_parts),
            sam3_checkpoint=sam3_checkpoint,
            sam3_device=sam3_device,
        )
        if not run_training and not Path(post_lite_meta["input_ply"]).exists():
            STATE.append(f"[launcher] 精简版输入点云不存在: {post_lite_meta['input_ply']}")
            return (
                "空闲（后处理精简版输入缺失）",
                STATE.tail(),
                shell_join(cmd) if run_training else "",
                0.0,
                0,
                get_monitor_text(),
            )
        post_steps, post_meta = build_postprocess_commands(
            python_exec=python_exec,
            path_mode=path_mode,
            source_path=source_path,
            model_path=model_path,
            post_iteration=int(post_iteration),
            post_input_ply=post_input_ply,
            post_n_clusters=int(post_n_clusters),
            post_cluster_output_dir=post_cluster_output_dir,
            post_sem_output_dir=post_sem_output_dir,
            post_instance_output_dir=post_instance_output_dir,
            post_vote_ignore_ids=post_vote_ignore_ids,
            post_min_votes=int(post_min_votes),
            post_min_top1_ratio=float(post_min_top1_ratio),
            post_id2label=post_id2label,
            post_sem_stuff_ids=post_sem_stuff_ids,
            post_semantic_only_ids=post_semantic_only_ids,
            post_instance_single_label=post_instance_single_label,
            post_instance_stuff_ids=post_instance_stuff_ids,
            post_instance_mask_dir=post_instance_mask_dir,
            post_instance_mask_mode=post_instance_mask_mode,
            post_instance_mask_suffix=post_instance_mask_suffix,
            post_instance_mask_level=int(post_instance_mask_level),
            post_instance_mask_ignore_ids=post_instance_mask_ignore_ids,
            run_post_mask2inst=bool(run_post_mask2inst),
            post_mask2inst_output_dir=post_mask2inst_output_dir,
            post_mask2inst_min_area=int(post_mask2inst_min_area),
            post_mask2inst_ignore_ids=post_mask2inst_ignore_ids,
            post_instance_vote_stride=int(post_instance_vote_stride),
            post_instance_min_mask_points=int(post_instance_min_mask_points),
            post_instance_match_iou=float(post_instance_match_iou),
            post_instance_min_point_votes=int(post_instance_min_point_votes),
            post_min_instance_points=int(post_min_instance_points),
            post_save_instance_parts=bool(post_save_instance_parts),
            run_post_cluster=bool(run_post_cluster),
            run_post_semantic=bool(run_post_semantic),
            run_post_instance=bool(run_post_instance),
        )
        if not post_steps:
            STATE.append("[launcher] 后处理已启用，但未勾选任何子步骤。")
            return (
                "空闲（后处理配置无效）",
                STATE.tail(),
                shell_join(cmd) if run_training else "",
                0.0,
                0,
                get_monitor_text(),
            )
        if run_post_mask2inst and not Path(post_meta["language_features_dir"]).exists():
            STATE.append(f"[launcher] 语义mask目录不存在: {post_meta['language_features_dir']}")
            return (
                "空闲（后处理输入缺失）",
                STATE.tail(),
                shell_join(cmd) if run_training else "",
                0.0,
                0,
                get_monitor_text(),
            )
        if run_post_cluster and not Path(post_meta["input_ply"]).exists():
            STATE.append(f"[launcher] 聚类输入点云不存在: {post_meta['input_ply']}")
            return (
                "空闲（后处理输入缺失）",
                STATE.tail(),
                shell_join(cmd) if run_training else "",
                0.0,
                0,
                get_monitor_text(),
            )
        if run_post_instance and (not run_post_mask2inst) and not Path(post_meta["instance_mask_dir"]).exists():
            STATE.append(f"[launcher] 实例掩码目录不存在: {post_meta['instance_mask_dir']}")
            return (
                "空闲（后处理输入缺失）",
                STATE.tail(),
                shell_join(cmd) if run_training else "",
                0.0,
                0,
                get_monitor_text(),
            )

    sam3_cmd: Optional[List[str]] = None

    if run_sam3:
        if not SAM3_PY.exists():
            STATE.append(f"[launcher] SAM3环境不存在: {SAM3_PY}")
            return (
                "空闲（SAM3环境缺失）",
                STATE.tail(),
                shell_join(cmd),
                STATE.get_progress(),
                STATE.get_current_iter(),
                get_monitor_text(),
            )
        if not SAM3_SCRIPT.exists():
            STATE.append(f"[launcher] SAM3脚本不存在: {SAM3_SCRIPT}")
            return (
                "空闲（SAM3脚本缺失）",
                STATE.tail(),
                shell_join(cmd),
                STATE.get_progress(),
                STATE.get_current_iter(),
                get_monitor_text(),
            )
        sam3_input_norm = normalize_path(sam3_input_dir, mode=path_mode)
        if not Path(sam3_input_norm).exists():
            STATE.append(f"[launcher] SAM3输入目录不存在: {sam3_input_norm}")
            return (
                "空闲（SAM3输入缺失）",
                STATE.tail(),
                shell_join(cmd),
                STATE.get_progress(),
                STATE.get_current_iter(),
                get_monitor_text(),
            )
        sam3_ckpt_norm = normalize_path(sam3_checkpoint, mode=path_mode)
        if sam3_ckpt_norm and not Path(sam3_ckpt_norm).exists():
            STATE.append(f"[launcher] SAM3权重不存在: {sam3_ckpt_norm}")
            return (
                "空闲（SAM3权重缺失）",
                STATE.tail(),
                shell_join(cmd),
                STATE.get_progress(),
                STATE.get_current_iter(),
                get_monitor_text(),
            )

        sam3_cmd = build_sam3_command(
            path_mode=path_mode,
            sam3_input_dir=sam3_input_dir,
            sam3_output_dir=sam3_output_dir,
            sam3_checkpoint=sam3_checkpoint,
            sam3_device=sam3_device,
            source_path=source_path,
        )
        STATE.append("[launcher] SAM3预处理已启用")
        STATE.append(f"[launcher] sam3_cmd={shell_join(sam3_cmd)}")
        sam3_code = run_blocking_with_logs(sam3_cmd, "sam3")
        if sam3_code != 0:
            STATE.append(f"[launcher] SAM3失败，退出码={sam3_code}")
            return (
                "空闲（SAM3失败）",
                STATE.tail(),
                shell_join(cmd),
                STATE.get_progress(),
                STATE.get_current_iter(),
                get_monitor_text(),
            )

        if sam3_export_to_language_features:
            try:
                scene = normalize_path(source_path, mode=path_mode)
                out_dir = normalize_path(sam3_output_dir, mode=path_mode) if (sam3_output_dir or "").strip() else str(Path(scene) / "sam3_masks")
                copied = export_sam3_to_language_features(
                    path_mode=path_mode,
                    source_path=source_path,
                    sam3_output_dir=out_dir,
                )
                STATE.append(f"[launcher] SAM3 npy -> language_features 完成，复制 {copied} 个文件")
            except Exception as exc:
                STATE.append(f"[launcher] SAM3结果转换失败: {exc}")
                return (
                    "空闲（SAM3转换失败）",
                    STATE.tail(),
                    shell_join(cmd),
                    STATE.get_progress(),
                    STATE.get_current_iter(),
                    get_monitor_text(),
                )

    if not run_training:
        all_post_steps = []
        if run_postprocess:
            all_post_steps.extend(post_steps)
        if run_postprocess_lite:
            all_post_steps.extend(post_lite_steps)

        if all_post_steps:
            STATE.append("[launcher] 开始执行后处理流水线...")
            ok = run_postprocess_steps(all_post_steps)
            cmd_preview = "\n".join(shell_join(c) for _, c in all_post_steps)
            return (
                "空闲（后处理完成）" if ok else "空闲（后处理失败）",
                STATE.tail(),
                cmd_preview,
                100.0 if ok else STATE.get_progress(),
                STATE.get_current_iter(),
                get_monitor_text(),
            )
        return (
            "空闲（SAM3完成）",
            STATE.tail(),
            shell_join(sam3_cmd) if sam3_cmd is not None else "",
            0.0,
            0,
            get_monitor_text(),
        )

    STATE.command = cmd
    STATE.started_at = time.time()
    STATE.target_iterations = int(iterations)
    STATE.current_iteration = 0
    STATE.progress_percent = 0.0

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    STATE.process = proc
    STATE.append("[launcher] training started")
    STATE.append(f"[launcher] cwd={ROOT_DIR}")
    STATE.append(f"[launcher] cmd={shell_join(cmd)}")
    all_post_steps = []
    if run_postprocess and post_steps:
        all_post_steps.extend(post_steps)
    if run_postprocess_lite and post_lite_steps:
        all_post_steps.extend(post_lite_steps)
    if all_post_steps:
        STATE.append("[launcher] 已启用后处理：训练完成后自动执行聚合/实例步骤")
    _spawn_reader(proc, post_steps=all_post_steps if all_post_steps else None)

    # Prime CPU sampling baseline to avoid first read always 0.
    if PSUTIL_AVAILABLE:
        try:
            psutil.Process(proc.pid).cpu_percent(interval=None)
        except Exception:
            pass

    return (
        STATE.status_text(),
        STATE.tail(),
        shell_join(cmd),
        STATE.get_progress(),
        STATE.get_current_iter(),
        get_monitor_text(),
    )


def stop_training():
    if not STATE.is_running():
        return STATE.status_text(), STATE.tail(), STATE.get_progress(), STATE.get_current_iter(), get_monitor_text()

    if STATE.process is not None and STATE.process.poll() is None:
        STATE.append("[launcher] stopping train process...")
        STATE.process.terminate()
        try:
            STATE.process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            STATE.append("[launcher] train terminate timeout, killing process...")
            STATE.process.kill()
            STATE.process.wait(timeout=5)
        STATE.append("[launcher] train process stopped")

    if STATE.side_process is not None and STATE.side_process.poll() is None:
        STATE.append("[launcher] stopping side process...")
        STATE.side_process.terminate()
        try:
            STATE.side_process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            STATE.append("[launcher] side terminate timeout, killing process...")
            STATE.side_process.kill()
            STATE.side_process.wait(timeout=5)
        STATE.append("[launcher] side process stopped")

    STATE.set_post_running(False)
    return STATE.status_text(), STATE.tail(), STATE.get_progress(), STATE.get_current_iter(), get_monitor_text()


def refresh_logs():
    return STATE.status_text(), STATE.tail(), STATE.get_progress(), STATE.get_current_iter(), get_monitor_text()


def get_monitor_text() -> str:
    pid: Optional[int] = None
    if STATE.process is not None and STATE.process.poll() is None:
        pid = STATE.process.pid
    elif STATE.side_process is not None and STATE.side_process.poll() is None:
        pid = STATE.side_process.pid
    proc_line = _format_process_metrics(pid)
    gpu_line = _format_gpu_metrics()
    return f"{proc_line}\n{gpu_line}"


def _extract_iter_from_name(name: str) -> int:
    m = re.search(r"(\d+)", name)
    if not m:
        return -1
    try:
        return int(m.group(1))
    except Exception:
        return -1


def _collect_valid_images(img_dir: Path) -> List[Tuple[Path, float]]:
    files: List[Path] = []
    for pat in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        files.extend(img_dir.glob(pat))

    valid: List[Tuple[Path, float]] = []
    for p in files:
        try:
            if not p.is_file():
                continue
            st = p.stat()
            if st.st_size <= 0:
                continue
            valid.append((p, st.st_mtime))
        except Exception:
            continue
    return valid


def _pick_latest_image(valid: List[Tuple[Path, float]]) -> Optional[Path]:
    if not valid:
        return None
    return max(valid, key=lambda x: (_extract_iter_from_name(x[0].stem), x[1]))[0]


def refresh_stage1_ins_feat_preview(path_mode: str, model_path: str) -> Tuple[Optional[str], Optional[str], str]:
    try:
        model_path_norm = normalize_path(model_path, mode=path_mode)
        if not model_path_norm:
            return None, None, "未设置输出路径（model_path）"

        ins_feat_dir = Path(model_path_norm) / "train_process" / "stage1" / "ins_feat"
        gt_dir = Path(model_path_norm) / "train_process" / "gt"

        if not ins_feat_dir.exists() and not gt_dir.exists():
            return None, None, f"未找到目录: {ins_feat_dir} 或 {gt_dir}"

        ins_valid = _collect_valid_images(ins_feat_dir) if ins_feat_dir.exists() else []
        gt_valid = _collect_valid_images(gt_dir) if gt_dir.exists() else []
        if not ins_valid and not gt_valid:
            return None, None, f"目录存在但暂无可读图片: {ins_feat_dir} / {gt_dir}"

        latest_ins = _pick_latest_image(ins_valid)
        latest_gt = None
        iter_id = -1

        if latest_ins is not None:
            iter_id = _extract_iter_from_name(latest_ins.stem)
            same_iter = [x for x in gt_valid if _extract_iter_from_name(x[0].stem) == iter_id] if iter_id >= 0 else []
            latest_gt = _pick_latest_image(same_iter) if same_iter else _pick_latest_image(gt_valid)
        else:
            latest_gt = _pick_latest_image(gt_valid)

        gt_path = str(latest_gt) if latest_gt is not None else None
        ins_path = str(latest_ins) if latest_ins is not None else None

        gt_msg = f"GT: {latest_gt.name}" if latest_gt is not None else "GT: 未找到"
        if latest_ins is not None and iter_id >= 0:
            ins_msg = f"INS_FEAT: {latest_ins.name}（迭代 {iter_id}）"
        elif latest_ins is not None:
            ins_msg = f"INS_FEAT: {latest_ins.name}"
        else:
            ins_msg = "INS_FEAT: 未找到"
        return gt_path, ins_path, f"{gt_msg} | {ins_msg}"
    except Exception as exc:
        return None, None, f"特征图预览刷新失败: {type(exc).__name__}: {exc}"


def toggle_training_controls(run_training: bool):
    upd = gr.update(interactive=bool(run_training))
    # model_path, start_checkpoint, iterations, start_ins_feat_iter, sam_level,
    # save_iterations_text, checkpoint_iterations_text, data_device, resolution,
    # checkpoint_max_points, eval_mode, save_memory
    return [upd, upd, upd, upd, upd, upd, upd, upd, upd, upd, upd, upd]


def toggle_sam3_controls(run_sam3: bool):
    upd = gr.update(interactive=bool(run_sam3))
    # sam3_export_to_language_features, sam3_input_dir, sam3_output_dir, sam3_checkpoint, sam3_device
    return [upd, upd, upd, upd, upd]


def toggle_post_controls(run_postprocess: bool):
    upd = gr.update(interactive=bool(run_postprocess))
    # post controls
    return [upd] * 33


def toggle_start_button(run_sam3: bool, run_training: bool, run_postprocess: bool):
    return gr.update(interactive=bool(run_sam3 or run_training or run_postprocess))


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="OpenGaussian Stage1 Trainer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## OpenGaussian Stage1 训练面板")
        gr.Markdown("只负责拼参数并启动 `train.py`，不改训练逻辑。")

        with gr.Row():
            python_exec = gr.Textbox(
                label="Python 可执行文件（python）",
                value="python",
                placeholder="/home/xxx/miniconda3/envs/opengaussian/bin/python",
            )
            path_mode = gr.Dropdown(
                label="路径格式模式",
                choices=[
                    ("自动识别（推荐）", "auto"),
                    ("Linux 路径", "linux"),
                    ("Windows 路径", "windows"),
                ],
                value="auto",
            )
            data_device = gr.Dropdown(
                label="数据设备（--data_device）",
                choices=["cuda", "cpu"],
                value="cuda",
            )
            resolution = gr.Dropdown(
                label="分辨率倍率（--resolution / -r）",
                choices=["default", "1", "2", "4", "8"],
                value="default",
            )

        with gr.Row():
            source_path = gr.Textbox(label="场景路径（-s / --source_path）", value="D:\\Scene_0325")
            model_path = gr.Textbox(label="输出路径（-m / --model_path）", value="D:\\Scene_0325\\test0325")

        start_checkpoint = gr.Textbox(
            label="起始权重（--start_checkpoint，可空）",
            value="D:\\Scene_0325\\chkpnt30000_from_tile0001_raw.pth",
        )

        with gr.Row():
            iterations = gr.Number(label="总训练步数（--iterations）", value=40000, precision=0)
            start_ins_feat_iter = gr.Number(label="实例特征起始步（--start_ins_feat_iter）", value=30000, precision=0)
            sam_level = gr.Number(label="SAM层级（--sam_level）", value=0, precision=0)

        with gr.Row():
            save_iterations_text = gr.Textbox(
                label="保存点云步数（--save_iterations，逗号/空格分隔）",
                value="40000",
                placeholder="例如: 35000 40000",
            )
            checkpoint_iterations_text = gr.Textbox(
                label="保存checkpoint步数（--checkpoint_iterations，逗号/空格分隔）",
                value="40000",
                placeholder="例如: 36000,40000",
            )

        with gr.Row():
            eval_mode = gr.Checkbox(label="启用评估集（--eval）", value=False)
            save_memory = gr.Checkbox(label="省显存模式（--save_memory）", value=False)
            checkpoint_max_points = gr.Number(
                label="起始ckpt点数上限（下采样，0=关闭）",
                value=0,
                precision=0,
            )

        gr.Markdown("### 流程选择")
        with gr.Row():
            run_sam3 = gr.Checkbox(label="跑 SAM3（生成语义mask）", value=False)
            run_training = gr.Checkbox(label="跑训练（train.py）", value=True)
        with gr.Row():
            sam3_export_to_language_features = gr.Checkbox(
                label="自动转为 OpenGaussian 掩码（language_features/*_s.npy）",
                value=True,
            )
        with gr.Row():
            sam3_input_dir = gr.Textbox(
                label="SAM3输入图片目录（--input-dir）",
                value="D:\\Scene_0325\\images",
            )
            sam3_output_dir = gr.Textbox(
                label="SAM3输出目录（--output-dir，可空）",
                value="",
                placeholder="为空则默认 <scene_path>/sam3_masks",
            )
        with gr.Row():
            sam3_checkpoint = gr.Textbox(
                label="SAM3权重路径（--checkpoint-path）",
                value=str(SAM3_DEFAULT_CKPT),
            )
            sam3_device = gr.Dropdown(
                label="SAM3设备（--device）",
                choices=["cuda", "cpu"],
                value="cuda",
            )

        gr.Markdown("### 后处理（聚类 + 语义聚合 + 实例分割）")
        with gr.Row():
            run_postprocess = gr.Checkbox(label="跑后处理流水线", value=False)
            run_post_mask2inst = gr.Checkbox(label="语义mask转实例mask（连通域）", value=True)
            run_post_cluster = gr.Checkbox(label="聚类（KMeans）", value=True)
            run_post_semantic = gr.Checkbox(label="语义聚合", value=True)
            run_post_instance = gr.Checkbox(label="实例分割（proj2d）", value=True)
        with gr.Row():
            post_mask2inst_output_dir = gr.Textbox(
                label="连通域实例mask输出目录（可空）",
                value="",
                placeholder="为空则默认 <input_ply_dir>/language_features_instance",
            )
            post_mask2inst_ignore_ids = gr.Textbox(label="连通域忽略语义ID", value="0")
            post_mask2inst_min_area = gr.Number(label="连通域最小面积", value=20, precision=0)
        with gr.Row():
            post_iteration = gr.Number(label="后处理使用迭代步", value=40000, precision=0)
            post_input_ply = gr.Textbox(
                label="后处理输入点云（可空）",
                value="",
                placeholder="为空则默认 <model_path>/point_cloud/iteration_xxx/point_cloud.ply",
            )
            post_n_clusters = gr.Number(label="聚类数量（K）", value=8, precision=0)
            post_vote_ignore_ids = gr.Textbox(label="语义投票忽略ID", value="0,7")
        with gr.Row():
            post_cluster_output_dir = gr.Textbox(
                label="聚类输出目录（可空）",
                value="",
                placeholder="为空则默认 <input_ply_dir>/kmeans_8",
            )
            post_sem_output_dir = gr.Textbox(
                label="语义聚合输出目录（可空）",
                value="",
                placeholder="为空则默认 <input_ply_dir>/sem_stage",
            )
            post_instance_output_dir = gr.Textbox(
                label="实例分割输出目录（可空）",
                value="",
                placeholder="为空则默认 <input_ply_dir>/sem_ins_proj2d",
            )
        with gr.Row():
            post_min_votes = gr.Number(label="语义最小票数", value=800, precision=0)
            post_min_top1_ratio = gr.Number(label="语义Top1比例阈值", value=0.4, precision=2)
            post_auto_sync_id2label = gr.Checkbox(label="从 language_features 自动同步 id2label", value=True)
        post_id2label = gr.Textbox(
            label="id2label 映射",
            value="0:background,1:vehicle,2:person,3:bicycle,4:vegetation,5:road,6:traffic_facility,7:other",
        )
        with gr.Row():
            post_sem_stuff_ids = gr.Textbox(label="语义聚合中的stuff_ids", value="0,1,2,3,4,5,6,7,255")
            post_semantic_only_ids = gr.Textbox(label="实例阶段仅处理语义ID", value="1")
            post_instance_single_label = gr.Textbox(label="实例单标签（id/label，可空）", value="")
            post_instance_stuff_ids = gr.Textbox(label="实例阶段stuff_ids", value="0,4,5,6,7,255")
        with gr.Row():
            post_instance_mask_dir = gr.Textbox(label="2D实例掩码目录", value="D:\\Scene_0325\\language_features_instance")
            post_instance_mask_mode = gr.Dropdown(label="实例掩码匹配模式", choices=["name", "index"], value="name")
            post_instance_mask_suffix = gr.Textbox(label="实例掩码后缀", value="_inst.npy")
        with gr.Row():
            post_instance_mask_level = gr.Number(label="实例掩码层级", value=0, precision=0)
            post_instance_mask_ignore_ids = gr.Textbox(label="实例掩码忽略ID", value="0")
            post_instance_vote_stride = gr.Number(label="实例投票视角步长", value=1, precision=0)
        with gr.Row():
            post_instance_min_mask_points = gr.Number(label="实例最小mask点数", value=30, precision=0)
            post_instance_match_iou = gr.Number(label="实例匹配IoU", value=0.2, precision=2)
            post_instance_min_point_votes = gr.Number(label="实例最小点票数", value=2, precision=0)
            post_min_instance_points = gr.Number(label="实例最小点数", value=180, precision=0)
            post_save_instance_parts = gr.Checkbox(label="保存每个实例子PLY", value=True)

        inputs = [
            python_exec,
            path_mode,
            source_path,
            model_path,
            start_checkpoint,
            iterations,
            start_ins_feat_iter,
            sam_level,
            save_iterations_text,
            checkpoint_iterations_text,
            resolution,
            checkpoint_max_points,
            data_device,
            eval_mode,
            save_memory,
            run_sam3,
            run_training,
            sam3_input_dir,
            sam3_output_dir,
            sam3_checkpoint,
            sam3_device,
            sam3_export_to_language_features,
            post_auto_sync_id2label,
            run_postprocess,
            run_post_mask2inst,
            post_mask2inst_output_dir,
            post_mask2inst_min_area,
            post_mask2inst_ignore_ids,
            run_post_cluster,
            run_post_semantic,
            run_post_instance,
            post_iteration,
            post_input_ply,
            post_n_clusters,
            post_cluster_output_dir,
            post_sem_output_dir,
            post_instance_output_dir,
            post_vote_ignore_ids,
            post_min_votes,
            post_min_top1_ratio,
            post_id2label,
            post_sem_stuff_ids,
            post_semantic_only_ids,
            post_instance_single_label,
            post_instance_stuff_ids,
            post_instance_mask_dir,
            post_instance_mask_mode,
            post_instance_mask_suffix,
            post_instance_mask_level,
            post_instance_mask_ignore_ids,
            post_instance_vote_stride,
            post_instance_min_mask_points,
            post_instance_match_iou,
            post_instance_min_point_votes,
            post_min_instance_points,
            post_save_instance_parts,
        ]

        with gr.Row():
            btn_preview = gr.Button("生成命令", variant="secondary")
            btn_start = gr.Button("开始执行", variant="primary")
            btn_stop = gr.Button("停止训练", variant="stop")
            btn_refresh = gr.Button("刷新日志", variant="secondary")

        command_preview = gr.Textbox(label="训练命令预览", lines=3)
        status_box = gr.Textbox(label="状态", value=STATE.status_text())
        progress_bar = gr.Slider(label="训练进度(%)", minimum=0, maximum=100, step=0.1, value=0.0, interactive=False)
        current_iter_box = gr.Number(label="当前迭代", value=0, precision=0, interactive=False)
        monitor_box = gr.Textbox(label="资源监控（每3秒刷新）", lines=3, interactive=False)
        log_box = gr.Textbox(label="日志（最近 500 行）", lines=20)
        with gr.Row():
            stage1_gt_preview = gr.Image(
                label="GT 预览（train_process/gt）",
                interactive=False,
                height=360,
            )
            stage1_ins_feat_preview = gr.Image(
                label="Stage1 特征图预览（train_process/stage1/ins_feat）",
                interactive=False,
                height=360,
            )
        stage1_ins_feat_status = gr.Textbox(
            label="对比图状态",
            value="点击“刷新特征图”加载最新 GT/INS_FEAT 对比",
            lines=3,
            interactive=False,
        )
        with gr.Row():
            btn_refresh_ins_feat = gr.Button("刷新特征图", variant="secondary")

        btn_preview.click(fn=preview_command, inputs=inputs, outputs=[command_preview])
        start_evt = btn_start.click(
            fn=start_training,
            inputs=inputs,
            outputs=[status_box, log_box, command_preview, progress_bar, current_iter_box, monitor_box],
        )
        btn_stop.click(fn=stop_training, outputs=[status_box, log_box, progress_bar, current_iter_box, monitor_box])
        refresh_evt = btn_refresh.click(
            fn=refresh_logs,
            outputs=[status_box, log_box, progress_bar, current_iter_box, monitor_box],
        )
        btn_refresh_ins_feat.click(
            fn=refresh_stage1_ins_feat_preview,
            inputs=[path_mode, model_path],
            outputs=[stage1_gt_preview, stage1_ins_feat_preview, stage1_ins_feat_status],
        )

        run_training.change(
            fn=toggle_training_controls,
            inputs=[run_training],
            outputs=[
                model_path,
                start_checkpoint,
                iterations,
                start_ins_feat_iter,
                sam_level,
                save_iterations_text,
                checkpoint_iterations_text,
                data_device,
                resolution,
                checkpoint_max_points,
                eval_mode,
                save_memory,
            ],
        )
        run_sam3.change(
            fn=toggle_sam3_controls,
            inputs=[run_sam3],
            outputs=[
                sam3_export_to_language_features,
                sam3_input_dir,
                sam3_output_dir,
                sam3_checkpoint,
                sam3_device,
            ],
        )
        run_training.change(
            fn=toggle_start_button,
            inputs=[run_sam3, run_training, run_postprocess],
            outputs=[btn_start],
        )
        run_sam3.change(
            fn=toggle_start_button,
            inputs=[run_sam3, run_training, run_postprocess],
            outputs=[btn_start],
        )
        run_postprocess.change(
            fn=toggle_start_button,
            inputs=[run_sam3, run_training, run_postprocess],
            outputs=[btn_start],
        )
        run_postprocess.change(
            fn=toggle_post_controls,
            inputs=[run_postprocess],
            outputs=[
                run_post_mask2inst,
                post_mask2inst_output_dir,
                post_mask2inst_ignore_ids,
                post_mask2inst_min_area,
                run_post_cluster,
                run_post_semantic,
                run_post_instance,
                post_iteration,
                post_input_ply,
                post_n_clusters,
                post_cluster_output_dir,
                post_sem_output_dir,
                post_instance_output_dir,
                post_vote_ignore_ids,
                post_min_votes,
                post_min_top1_ratio,
                post_auto_sync_id2label,
                post_id2label,
                post_sem_stuff_ids,
                post_semantic_only_ids,
                post_instance_single_label,
                post_instance_stuff_ids,
                post_instance_mask_dir,
                post_instance_mask_mode,
                post_instance_mask_suffix,
                post_instance_mask_level,
                post_instance_mask_ignore_ids,
                post_instance_vote_stride,
                post_instance_min_mask_points,
                post_instance_match_iou,
                post_instance_min_point_votes,
                post_min_instance_points,
                post_save_instance_parts,
            ],
        )
        source_path.change(
            fn=sync_id2label_for_ui,
            inputs=[path_mode, source_path, post_id2label, post_auto_sync_id2label],
            outputs=[post_id2label],
        )
        path_mode.change(
            fn=sync_id2label_for_ui,
            inputs=[path_mode, source_path, post_id2label, post_auto_sync_id2label],
            outputs=[post_id2label],
        )
        post_auto_sync_id2label.change(
            fn=sync_id2label_for_ui,
            inputs=[path_mode, source_path, post_id2label, post_auto_sync_id2label],
            outputs=[post_id2label],
        )

        demo.load(fn=refresh_logs, outputs=[status_box, log_box, progress_bar, current_iter_box, monitor_box])
        demo.load(
            fn=toggle_training_controls,
            inputs=[run_training],
            outputs=[
                model_path,
                start_checkpoint,
                iterations,
                start_ins_feat_iter,
                sam_level,
                save_iterations_text,
                checkpoint_iterations_text,
                data_device,
                resolution,
                checkpoint_max_points,
                eval_mode,
                save_memory,
            ],
        )
        demo.load(
            fn=toggle_sam3_controls,
            inputs=[run_sam3],
            outputs=[
                sam3_export_to_language_features,
                sam3_input_dir,
                sam3_output_dir,
                sam3_checkpoint,
                sam3_device,
            ],
        )
        demo.load(
            fn=toggle_post_controls,
            inputs=[run_postprocess],
            outputs=[
                run_post_mask2inst,
                post_mask2inst_output_dir,
                post_mask2inst_ignore_ids,
                post_mask2inst_min_area,
                run_post_cluster,
                run_post_semantic,
                run_post_instance,
                post_iteration,
                post_input_ply,
                post_n_clusters,
                post_cluster_output_dir,
                post_sem_output_dir,
                post_instance_output_dir,
                post_vote_ignore_ids,
                post_min_votes,
                post_min_top1_ratio,
                post_auto_sync_id2label,
                post_id2label,
                post_sem_stuff_ids,
                post_semantic_only_ids,
                post_instance_single_label,
                post_instance_stuff_ids,
                post_instance_mask_dir,
                post_instance_mask_mode,
                post_instance_mask_suffix,
                post_instance_mask_level,
                post_instance_mask_ignore_ids,
                post_instance_vote_stride,
                post_instance_min_mask_points,
                post_instance_match_iou,
                post_instance_min_point_votes,
                post_min_instance_points,
                post_save_instance_parts,
            ],
        )
        demo.load(
            fn=sync_id2label_for_ui,
            inputs=[path_mode, source_path, post_id2label, post_auto_sync_id2label],
            outputs=[post_id2label],
        )
        demo.load(fn=toggle_start_button, inputs=[run_sam3, run_training, run_postprocess], outputs=[btn_start])
        timer = gr.Timer(3.0)
        timer.tick(fn=refresh_logs, outputs=[status_box, log_box, progress_bar, current_iter_box, monitor_box])

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.queue()
    launch_kwargs = dict(server_name="0.0.0.0", server_port=7860, share=False)

    if WSL_POWERSHELL.exists():
        threading.Timer(1.0, lambda: open_browser_windows_from_wsl("http://localhost:7860")).start()
        launch_kwargs["inbrowser"] = False
    else:
        launch_kwargs["inbrowser"] = True

    try:
        app.launch(**launch_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "localhost is not accessible" in msg or "shareable link must be created" in msg:
            print("[train_ui] localhost 检测失败，自动重试 share=True（并设置 NO_PROXY）。")
            os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost,0.0.0.0")
            os.environ.setdefault("no_proxy", "127.0.0.1,localhost,0.0.0.0")
            launch_kwargs["share"] = True
            app.launch(**launch_kwargs)
        else:
            raise
