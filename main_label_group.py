import math
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import random
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import savemat
from torch.utils.data import DataLoader, TensorDataset
from ODNN_functions import (
    generate_complex_weights,
    generate_fields_ts,
)
from odnn_generate_label import (
    compute_label_centers,
)
from odnn_io import load_complex_modes_from_mat
from odnn_training_eval import spot_energy_and_snr
# MultiWL model
from odnn_multiwl_model import D2NNModelMultiWL

from odnn_training_io import train_multiwl_staged, print_stage_summary, save_staged_training_info

from odnn_training_visualization import visualize_phase_masks

from odnn_training_visualization import (
    capture_eigenmode_propagation_multiwl,
    visualize_model_slices_multiwl,
    capture_eigenmode_propagation,
    export_superposition_slices,
    plot_amplitude_comparison_grid,
    plot_reconstruction_vs_input,
    plot_sys_vs_label_strict,
    save_superposition_triptych,
    save_mode_triptych,
    visualize_model_slices,
    save_mode_triptych_multiwl,
)
from odnn_training_eval import (
    spot_energy_and_snr,
    _make_circle_mask,
)
from odnn_multiwl_metrices import (
    evaluate_multiwl_comprehensive_metrics,
    plot_and_save_multiwl_metrics,
)
from odnn_processing import prepare_sample, pad_field_to_layer  # ★ 添加 pad_field_to_layer

# ============================================================
# Reproducibility / device
# ============================================================
SEED = 424242
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
print("Using Device:", device)


# ============================================================
# Parameters
# ============================================================

USE_SINGLEPOINT_LABEL = True
USE_SINGLEPOINT_LABEL = False
USE_MODEGROUP_LABEL = True              # ★ 新增：启用 mode group 标签

# Mode 分组定义（0-indexed，对应 num_modes=6 的情况）
# 含义：第 1 组只含 mode 0；第 2 组含 mode 1,2；第 3 组含 mode 3,4,5
MODE_GROUPS = [
    [0],            # group 1: mode 1
    [1, 2],         # group 2: mode 2, 3
    [3, 4, 5],      # group 3: mode 4, 5, 6
]

# 每个波长在 group 目标点处用的 ROI 半径
MODEGROUP_RADIUS_PER_WL = [10, 20]      
SINGLEPOINT_RADIUS_PER_WL = [10, 20]    # 654 → 小圆, 852 → 大圆
LABEL_RADIUS_PER_WL = [10, 20]
field_size = 141
num_modes = 6
circle_focus_radius = 15
circle_detectsize = 25
focus_radius = circle_focus_radius 
detectsize = circle_detectsize
layer_size = 300

# SLM
z_layers = 44.774e-3
pixel_size = 12.5e-6
z_prop = 130e-3
z_input_to_first = 0
out_size = 500
padding_ratio_out = 0.5

num_layer_option = [1, 2]

# ============================================================
# Wavelengths (MultiWL) — 起始波长 + 间隔 + 数量
# ============================================================
wl_start_nm = 654       # 起始波长 (nm)
wl_spacing_nm = 198       # 波长间隔 (nm)
wl_count = 2                # 波长数量
base_wavelength_idx = 0     # 基准波长索引

wavelengths = (wl_start_nm + np.arange(wl_count) * wl_spacing_nm).astype(np.float32) * 1e-9
L = int(len(wavelengths))

if base_wavelength_idx >= L:
    base_wavelength_idx = 0

print(f"★ Wavelengths ({L}): {wavelengths*1e9} nm | start={wl_start_nm} nm | "
      f"spacing={wl_spacing_nm} nm | base_idx={base_wavelength_idx}")

print(f"★ Wavelengths: {wavelengths*1e9} nm, spacing={wl_spacing_nm} nm, count={L}")

# data options
phase_option = 4
label_pattern_mode = "circle"
show_detection_overlap_debug = True

# train hyperparams
epochs = 1000
lr = 1.99
padding_ratio = 0.5
evaluation_mode = "eigenmode"
training_dataset_mode = "eigenmode"

num_superposition_eval_samples = 1000
num_superposition_train_samples = 2
superposition_eval_seed = 20240116
superposition_train_seed = 20240115
batch_size = 16

# ============================================================
# Output root 添加参数
# ============================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT = Path(
    f"results/"
    f"{num_modes}modes/"
    f"{L}wl_{wl_start_nm:.1f}nm_sp{wl_spacing_nm:.1f}nm_"
    f"base{base_wavelength_idx}_"
    f"ls{layer_size}_out_{out_size}_zp{z_prop*1e3:.0f}mm_z{z_layers*1e3:.1f}mm_zin{z_input_to_first*1e3:.1f}mm_"
    f"pr{padding_ratio}_c{circle_focus_radius}_"
    f"{timestamp}"
)
RUN_ROOT.mkdir(parents=True, exist_ok=True)

# prediction viz samples
num_pred_diag_samples = 3
num_superposition_visual_samples = 2

# ============================================================
# 多波长标签生成函数
# ============================================================
def generate_detector_patterns_multiwl(
    H: int,
    W: int,
    num_modes: int,
    num_wavelengths: int,
    radius,                          # ★ int 或 list[int]（长度=num_wavelengths）
    pattern_mode: str = "circle",
    show_debug: bool = False,
    margin_ratio: float = 0.15,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[int]]:
    """
    多波长标签生成（行=mode, 列=wavelength），支持每个波长不同 ROI 半径。

    Parameters
    ----------
    radius : int | list[int]
        - int           : 所有波长用同一个半径（向下兼容旧用法）
        - list[int]/tuple: 长度必须 = num_wavelengths，按波长顺序对应
                          例如 [10, 20] 表示 wl0(654nm)=r10, wl1(852nm)=r20

    Returns
    -------
    patterns           : (H, W, num_modes*num_wavelengths) float32, 二值标签
    evaluation_regions : list[(x0, x1, y0, y1)]  按 idx = mode*num_wl + wl 顺序
    per_label_radii    : list[int]               每个 label 实际使用的半径
    """
    # ----- 解析 radius -----
    if np.isscalar(radius):
        radius_per_wl = [int(radius)] * num_wavelengths
    else:
        radius_per_wl = [int(r) for r in radius]
        assert len(radius_per_wl) == num_wavelengths, (
            f"radius 列表长度 {len(radius_per_wl)} != num_wavelengths {num_wavelengths}"
        )

    total_labels = num_modes * num_wavelengths
    num_rows = num_modes          # ★ 固定：行 = mode
    num_cols = num_wavelengths    # ★ 固定：列 = wavelength

    # ----- 等距布局，留 margin -----
    max_r = max(radius_per_wl)
    margin_x = max(int(W * margin_ratio), max_r + 5)
    margin_y = max(int(H * margin_ratio), max_r + 5)

    if num_cols > 1:
        xs = np.linspace(margin_x, W - 1 - margin_x, num_cols)
    else:
        xs = np.array([W / 2.0])
    if num_rows > 1:
        ys = np.linspace(margin_y, H - 1 - margin_y, num_rows)
    else:
        ys = np.array([H / 2.0])
    
    xs = xs[::-1]   # inverse x-axis for better visualizationsd

    # ----- 中心 + 每个 label 的半径 -----
    centers: list[tuple[int, int]] = []
    per_label_radii: list[int] = []
    for mode_idx in range(num_rows):
        for wl_idx in range(num_cols):
            cy = int(round(ys[mode_idx]))
            cx = int(round(xs[wl_idx]))
            centers.append((cy, cx))
            per_label_radii.append(radius_per_wl[wl_idx])   # ★ 关键：列对应 wl 半径

    # ----- 检查 ROI 之间不会重叠 -----
    if num_rows > 1:
        row_gap = float(ys[1] - ys[0])
        if row_gap < 2 * max_r:
            print(f"[WARN] 行间距 {row_gap:.1f} < 2·max_r {2*max_r}, ROI 会重叠！")
    if num_cols > 1:
        col_gap = float(xs[1] - xs[0])
        if col_gap < (radius_per_wl[0] + radius_per_wl[-1]):
            print(f"[WARN] 列间距 {col_gap:.1f} 太小, ROI 可能重叠！")

    # ----- 生成 patterns -----
    if pattern_mode == "circle":
        patterns = np.zeros((H, W, total_labels), dtype=np.float32)
        yy, xx = np.ogrid[:H, :W]
        for idx, ((cy, cx), r) in enumerate(zip(centers, per_label_radii)):
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
            patterns[:, :, idx] = mask.astype(np.float32)
    else:
        raise NotImplementedError(f"Unsupported pattern_mode: {pattern_mode}")

    # ----- 生成 evaluation_regions -----
    evaluation_regions: list[tuple[int, int, int, int]] = []
    for (cy, cx), r in zip(centers, per_label_radii):
        x0 = max(0, cx - r)
        x1 = min(W, cx + r)
        y0 = max(0, cy - r)
        y1 = min(H, cy + r)
        evaluation_regions.append((x0, x1, y0, y1))

    # ----- debug 可视化 -----
    if show_debug:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(patterns.sum(axis=2), cmap='gray')
        for idx, ((cy, cx), r) in enumerate(zip(centers, per_label_radii)):
            mode_idx = idx // num_wavelengths
            wl_idx = idx % num_wavelengths
            ax.text(cx, cy, f"M{mode_idx}\nW{wl_idx}\nr={r}",
                    ha='center', va='center', color='red', fontsize=7)
        for x in xs:
            ax.axvline(x, color='cyan', linestyle=':', linewidth=0.5, alpha=0.4)
        for y in ys:
            ax.axhline(y, color='cyan', linestyle=':', linewidth=0.5, alpha=0.4)
        ax.set_title(
            f"MultiWL Labels: {num_modes} modes × {num_wavelengths} wl\n"
            f"radius per wl = {radius_per_wl}"
        )
        plt.savefig("debug_multiwl_labels.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✔ Debug label layout saved -> debug_multiwl_labels.png")

    return patterns, evaluation_regions, per_label_radii

def generate_detector_patterns_multiwl_singlepoint(
    H: int,
    W: int,
    num_modes: int,
    num_wavelengths: int,
    radius,                       # int 或 list[int] (长度=num_wavelengths)
    pattern_mode: str = "circle",
    show_debug: bool = False,
    margin_ratio: float = 0.2,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    单点标签版本：所有 mode 目标都是同一个点，只有 wavelength 区分位置。

    结果：
      - 画布上只有 num_wavelengths 个不同位置
      - 但返回的 patterns shape 仍然是 (H, W, num_modes*num_wavelengths)
        以兼容现有的数据集 / 损失 / metrics 流程
      - 同一 wavelength 下，所有 mode 的 label 完全相同
    """
    # ----- 解析 radius -----
    if np.isscalar(radius):
        radius_per_wl = [int(radius)] * num_wavelengths
    else:
        radius_per_wl = [int(r) for r in radius]
        assert len(radius_per_wl) == num_wavelengths, (
            f"radius 列表长度 {len(radius_per_wl)} != num_wavelengths {num_wavelengths}"
        )

    total_labels = num_modes * num_wavelengths

    # ----- 只在水平方向放 num_wavelengths 个点，居中 -----
    max_r = max(radius_per_wl)
    margin_x = max(int(W * margin_ratio), max_r + 5)
    if num_wavelengths > 1:
        xs = np.linspace(margin_x, W - 1 - margin_x, num_wavelengths)
    else:
        xs = np.array([W / 2.0])
    cy_single = H // 2   # 所有点都在画布垂直中心

    # ----- 每个 wavelength 一个固定中心 -----
    centers_per_wl = []
    for wl_idx in range(num_wavelengths):
        cx = int(round(xs[wl_idx]))
        centers_per_wl.append((cy_single, cx))

    # ----- 生成 patterns: 所有 mode 共享同一个 wl 的圆 -----
    if pattern_mode == "circle":
        # 先生成 num_wavelengths 个圆形 mask
        wl_masks = []
        yy, xx = np.ogrid[:H, :W]
        for (cy, cx), r in zip(centers_per_wl, radius_per_wl):
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
            wl_masks.append(mask.astype(np.float32))

        # 复制：mode_idx*num_wl + wl_idx 全部用同一个 wl_mask
        patterns = np.zeros((H, W, total_labels), dtype=np.float32)
        for mode_idx in range(num_modes):
            for wl_idx in range(num_wavelengths):
                idx = mode_idx * num_wavelengths + wl_idx
                patterns[:, :, idx] = wl_masks[wl_idx]
    else:
        raise NotImplementedError(f"Unsupported pattern_mode: {pattern_mode}")

    # ----- evaluation_regions: 每个 label 一个 region (虽然内容重复) -----
    evaluation_regions = []
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wavelengths):
            cy, cx = centers_per_wl[wl_idx]
            r = radius_per_wl[wl_idx]
            x0 = max(0, cx - r)
            x1 = min(W, cx + r)
            y0 = max(0, cy - r)
            y1 = min(H, cy + r)
            evaluation_regions.append((x0, x1, y0, y1))

    # ----- debug 可视化 -----
    if show_debug:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(patterns[:, :, :num_wavelengths].sum(axis=2), cmap='gray')
        for wl_idx, ((cy, cx), r) in enumerate(zip(centers_per_wl, radius_per_wl)):
            ax.text(cx, cy, f"λ{wl_idx}\nr={r}",
                    ha='center', va='center', color='red', fontsize=14, fontweight='bold')
            circle_overlay = plt.Circle((cx, cy), r, fill=False, color='lime', linewidth=2)
            ax.add_patch(circle_overlay)
        ax.axvline(margin_x, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(W - margin_x, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
        plt.title(
            f"SinglePoint Labels: {num_wavelengths} 波长 × 同 1 点\n"
            f"radius per wl = {radius_per_wl}, all {num_modes} modes 共享同一目标位置"
        )
        plt.savefig(RUN_ROOT / "debug_singlepoint_labels.png", dpi=150)
        plt.close()
        print(f"✔ SinglePoint label layout -> {RUN_ROOT / 'debug_singlepoint_labels.png'}")

    return patterns, evaluation_regions

def generate_detector_patterns_multiwl_modegroup(
    H: int,
    W: int,
    num_modes: int,
    num_wavelengths: int,
    mode_groups: list[list[int]],
    radius,                       # int 或 list[int]
    pattern_mode: str = "circle",
    show_debug: bool = False,
    margin_ratio: float = 0.2,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    Mode-group 标签生成：
      - 同 group 内的所有 mode 共享同一目标 ROI 位置
      - 不同 group 在垂直方向分开（行 = group）
      - 不同 wavelength 在水平方向分开（列 = wavelength），可有不同 ROI 半径
      - patterns shape 仍为 (H, W, num_modes*num_wavelengths)，与下游完全兼容
    """
    # ----- 校验 mode_groups -----
    all_mode_idxs = [m for g in mode_groups for m in g]
    assert sorted(all_mode_idxs) == list(range(num_modes)), (
        f"mode_groups 必须覆盖 0..{num_modes-1} 且每个 mode 只能出现一次, "
        f"got groups={mode_groups}, flatten_sorted={sorted(all_mode_idxs)}"
    )
    num_groups = len(mode_groups)

    # ----- radius -----
    if np.isscalar(radius):
        radius_per_wl = [int(radius)] * num_wavelengths
    else:
        radius_per_wl = [int(r) for r in radius]
        assert len(radius_per_wl) == num_wavelengths

    total_labels = num_modes * num_wavelengths
    max_r = max(radius_per_wl)
    margin_x = max(int(W * margin_ratio), max_r + 5)
    margin_y = max(int(H * margin_ratio), max_r + 5)

    if num_wavelengths > 1:
        xs = np.linspace(margin_x, W - 1 - margin_x, num_wavelengths)
    else:
        xs = np.array([W / 2.0])
    if num_groups > 1:
        ys = np.linspace(margin_y, H - 1 - margin_y, num_groups)
    else:
        ys = np.array([H / 2.0])

    xs = xs[::-1]   # 与现有 multiwl 版保持一致（左右翻转）

    # ----- 中心 -----
    group_wl_centers: dict[tuple[int, int], tuple[int, int]] = {}
    for g_idx in range(num_groups):
        for wl_idx in range(num_wavelengths):
            group_wl_centers[(g_idx, wl_idx)] = (int(round(ys[g_idx])), int(round(xs[wl_idx])))

    # ----- 间距警告 -----
    if num_groups > 1 and (ys[1] - ys[0]) < 2 * max_r:
        print(f"[WARN] group 行间距 {ys[1]-ys[0]:.1f} < 2·max_r {2*max_r}, ROI 会重叠！")
    if num_wavelengths > 1 and abs(xs[1] - xs[0]) < (radius_per_wl[0] + radius_per_wl[-1]):
        print(f"[WARN] λ 列间距 {abs(xs[1]-xs[0]):.1f} 太小, ROI 可能重叠！")

    # ----- mode -> group 映射 -----
    mode_to_group: dict[int, int] = {}
    for g_idx, modes_in_group in enumerate(mode_groups):
        for m in modes_in_group:
            mode_to_group[m] = g_idx

    # ----- 生成 (group, wl) 圆 -----
    if pattern_mode != "circle":
        raise NotImplementedError(f"Unsupported pattern_mode: {pattern_mode}")

    yy, xx = np.ogrid[:H, :W]
    group_wl_masks: dict[tuple[int, int], np.ndarray] = {}
    for (g_idx, wl_idx), (cy, cx) in group_wl_centers.items():
        r = radius_per_wl[wl_idx]
        m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        group_wl_masks[(g_idx, wl_idx)] = m.astype(np.float32)

    # ----- 复制到 patterns：同 group 同 wl 的所有 mode 共享 mask -----
    patterns = np.zeros((H, W, total_labels), dtype=np.float32)
    for mode_idx in range(num_modes):
        g_idx = mode_to_group[mode_idx]
        for wl_idx in range(num_wavelengths):
            idx = mode_idx * num_wavelengths + wl_idx
            patterns[:, :, idx] = group_wl_masks[(g_idx, wl_idx)]

    # ----- evaluation_regions -----
    evaluation_regions: list[tuple[int, int, int, int]] = []
    for mode_idx in range(num_modes):
        g_idx = mode_to_group[mode_idx]
        for wl_idx in range(num_wavelengths):
            cy, cx = group_wl_centers[(g_idx, wl_idx)]
            r = radius_per_wl[wl_idx]
            evaluation_regions.append((
                max(0, cx - r), min(W, cx + r),
                max(0, cy - r), min(H, cy + r),
            ))

    # ----- debug 可视化 -----
    if show_debug:
        fig, ax = plt.subplots(figsize=(9, 8))
        composite = np.zeros((H, W), dtype=np.float32)
        for m in group_wl_masks.values():
            composite = np.maximum(composite, m)
        ax.imshow(composite, cmap='gray')

        cmap_groups = plt.cm.tab10
        for g_idx, modes_in_group in enumerate(mode_groups):
            color = cmap_groups(g_idx % 10)
            for wl_idx in range(num_wavelengths):
                cy, cx = group_wl_centers[(g_idx, wl_idx)]
                r = radius_per_wl[wl_idx]
                ax.add_patch(plt.Circle((cx, cy), r, fill=False, color=color, linewidth=2))
                modes_str = ",".join(str(m + 1) for m in modes_in_group)
                ax.text(cx, cy, f"G{g_idx+1}\nM:{modes_str}\nλ{wl_idx}\nr={r}",
                        ha='center', va='center', color='white', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.75))
        for x in xs:
            ax.axvline(x, color='cyan', linestyle=':', linewidth=0.5, alpha=0.4)
        for y in ys:
            ax.axhline(y, color='cyan', linestyle=':', linewidth=0.5, alpha=0.4)
        ax.set_title(
            f"ModeGroup Labels: {num_groups} groups × {num_wavelengths} wl\n"
            f"groups = {mode_groups} | radius per wl = {radius_per_wl}"
        )
        out_p = RUN_ROOT / "debug_modegroup_labels.png"
        plt.savefig(out_p, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✔ ModeGroup label layout -> {out_p}")

    return patterns, evaluation_regions

# ============================================================
# Helpers
# ============================================================
def _safe_norm_np(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = v.sum(axis=-1, keepdims=True)
    return v / (s + eps)


def _per_sample_corrcoef(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (np.sqrt((a0 * a0).sum() + eps) * np.sqrt((b0 * b0).sum() + eps))
    return float((a0 * b0).sum() / denom)


def save_checkpoint_multiwl(model, out_path: Path, meta: dict):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {"state_dict": model.state_dict(), "meta": meta}
    torch.save(ckpt, str(out_path))


def extract_phase_masks_multiwl(model: D2NNModelMultiWL) -> list[np.ndarray]:
    masks = []
    for layer in getattr(model, "layers", []):
        if hasattr(layer, "phase"):
            ph = layer.phase.detach().cpu().numpy()
            masks.append(np.remainder(ph, 2 * np.pi))
    return masks


def save_training_curves(
    *,
    losses: list[float],
    epoch_durations: list[float],
    out_dir: Path,
    tag: str,
    num_layers: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    epochs_arr = np.arange(1, len(losses) + 1, dtype=np.int32)
    cum_times = np.cumsum(np.asarray(epoch_durations, dtype=np.float64))
    total_time = float(cum_times[-1]) if len(cum_times) else 0.0

    fig, ax = plt.subplots()
    ax.plot(epochs_arr, np.asarray(losses, dtype=np.float64), label="Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"MultiWL Training Loss ({num_layers} layers)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    loss_plot_path = out_dir / f"loss_curve_layers{num_layers}_{tag}.png"
    fig.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(epochs_arr, cum_times, label="Cumulative Time")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Time (seconds)")
    ax.set_title(f"Cumulative Training Time ({num_layers} layers)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    time_plot_path = out_dir / f"epoch_time_layers{num_layers}_{tag}.png"
    fig.savefig(time_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    mat_path = out_dir / f"training_curves_layers{num_layers}_{tag}.mat"
    savemat(
        str(mat_path),
        {
            "epochs": epochs_arr.astype(np.float64),
            "losses": np.asarray(losses, dtype=np.float64),
            "epoch_durations": np.asarray(epoch_durations, dtype=np.float64),
            "cumulative_epoch_times": cum_times,
            "total_training_time": np.asarray([total_time], dtype=np.float64),
            "num_layers": np.asarray([num_layers], dtype=np.int32),
        },
    )
    return {"loss_plot": loss_plot_path, "time_plot": time_plot_path, "mat": mat_path, "total_time": total_time}


def _make_circle_mask_local(h: int, w: int, r: float, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)
    return mask.to(torch.float32)


@torch.no_grad()
def region_energy_fractions(
    I_bhw: torch.Tensor,
    evaluation_regions: list[tuple[int, int, int, int]],
    detect_radius: int,
) -> torch.Tensor:
    B, H, W = I_bhw.shape
    M = len(evaluation_regions)
    out = torch.zeros((B, M), device=I_bhw.device, dtype=torch.float32)

    for mi, (x0, x1, y0, y1) in enumerate(evaluation_regions):
        patch = I_bhw[:, y0:y1, x0:x1]
        hh, ww = patch.shape[-2], patch.shape[-1]
        cmask = _make_circle_mask_local(hh, ww, float(detect_radius), device=I_bhw.device)
        out[:, mi] = (patch * cmask.unsqueeze(0)).sum(dim=(-1, -2))

    out = out / (out.sum(dim=1, keepdim=True) + 1e-12)
    return out

@torch.no_grad()
def evaluate_group_isolation_db(
    model: D2NNModelMultiWL,
    mode_groups: list[list[int]],
    evaluation_regions: list[tuple[int, int, int, int]],
    detect_radius: int,
    *,
    num_modes: int,
    num_wavelengths: int,
    wavelengths_m: np.ndarray,
    mmf_modes: torch.Tensor,        # (num_modes, H_mode, W_mode) complex
    layer_size: int,
    device: torch.device,
) -> dict:
    """
    Group-level isolation：
      - 把每个本征模 m 单独送入模型 → 得 I_pred(λ)
      - 每个 (group, λ) 一个 ROI（同 group 的所有 mode 共享）
      - 把 mode m 的能量按 ROI 聚合：E[m, λ, g] = ∫_ROI(g,λ) I_pred(λ)
      - target group = mode_to_group[m]
      - group_isolation(m, λ) = 10·log10( E_target / Σ_{g'≠target} E[m,λ,g'] )

    返回：
      - per_mode_per_wl_db (M, L)
      - mean_db
      - group_crosstalk_matrix (L, G, G)   ← 行=源 mode 所在 group, 列=ROI group
    """
    model.eval()
    M = num_modes
    L = num_wavelengths
    G = len(mode_groups)

    mode_to_group = {m: g for g, modes in enumerate(mode_groups) for m in modes}

    # 每个 group 对应的 ROI（每个 λ 取该 group 的代表 region；同 group 内所有 mode 的 region 相同）
    # 取 group 内第一个 mode 的 region 作为代表
    rep_mode_per_group = [g[0] for g in mode_groups]
    group_regions_per_wl: list[list[tuple[int,int,int,int]]] = []
    for wl_idx in range(L):
        regs = []
        for g_idx in range(G):
            m_rep = rep_mode_per_group[g_idx]
            regs.append(evaluation_regions[m_rep * L + wl_idx])
        group_regions_per_wl.append(regs)

    # 输入每个 eigenmode
    per_mode_per_wl_db = np.zeros((M, L), dtype=np.float64)
    energy_mlg = np.zeros((M, L, G), dtype=np.float64)

    for m_idx in range(M):
        mode_field = mmf_modes[m_idx].to(device=device, dtype=torch.complex64)
        padded = pad_field_to_layer(mode_field, layer_size)
        x = padded[None, None, ...].repeat(1, L, 1, 1).contiguous()
        I_pred = model(x)                            # (1, L, H, W) intensity
        I_pred = I_pred[0].to(torch.float32)         # (L, H, W)

        for wl_idx in range(L):
            I_hw = I_pred[wl_idx]
            for g_idx in range(G):
                x0, x1, y0, y1 = group_regions_per_wl[wl_idx][g_idx]
                patch = I_hw[y0:y1, x0:x1]
                hh, ww = patch.shape
                cmask = _make_circle_mask_local(hh, ww, float(detect_radius), device=I_hw.device)
                energy_mlg[m_idx, wl_idx, g_idx] = float((patch * cmask).sum().item())

            target_g = mode_to_group[m_idx]
            E_t = energy_mlg[m_idx, wl_idx, target_g]
            E_o = energy_mlg[m_idx, wl_idx, :].sum() - E_t
            per_mode_per_wl_db[m_idx, wl_idx] = 10.0 * np.log10(max(E_t, 1e-12) / max(E_o, 1e-12))

    # group crosstalk matrix per wl: 把同 group 内的所有 source mode 求和归一
    group_crosstalk = np.zeros((L, G, G), dtype=np.float64)
    for wl_idx in range(L):
        for src_g in range(G):
            src_modes = mode_groups[src_g]
            row = energy_mlg[src_modes, wl_idx, :].sum(axis=0)   # (G,)
            group_crosstalk[wl_idx, src_g, :] = row / (row.sum() + 1e-12)

    return {
        "per_mode_per_wl_db": per_mode_per_wl_db,
        "mean_db": float(np.mean(per_mode_per_wl_db)),
        "energy_mlg": energy_mlg,
        "group_crosstalk_matrix": group_crosstalk,
        "mode_groups": mode_groups,
    }

@torch.no_grad()
def save_prediction_diagnostics_multiwl(
    model: D2NNModelMultiWL,
    dataset: TensorDataset,
    *,
    wavelengths: np.ndarray,
    evaluation_regions: list,
    detect_radius: int,
    sample_indices: list[int],
    out_dir: Path,
    device: torch.device,
    tag: str,
    num_modes: int,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    Lloc = int(len(wavelengths))

    saved = []
    for si in sample_indices:
        x, label_img, amp = dataset[si]
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(device=device, dtype=torch.complex64).unsqueeze(0)
        label_img = label_img.to(device=device, dtype=torch.float32).unsqueeze(0)
        amp = amp.to(device=device, dtype=torch.float32).unsqueeze(0)

        xin = x.repeat(1, Lloc, 1, 1).contiguous()
        I_pred = model(xin)

        I_in = (torch.abs(x[0, 0]) ** 2).detach().cpu().numpy()

        amp2 = (amp[0] ** 2).detach().cpu().numpy()
        true_energy_frac = _safe_norm_np(amp2)

        labels_per_wl = []
        for wl_idx in range(Lloc):
            label_indices = [k * Lloc + wl_idx for k in range(num_modes)]
            wl_label_patterns = MMF_Label_data[:, :, label_indices].numpy()
            energy = true_energy_frac.reshape(1, -1)
            label_wl = np.einsum("nm,hwm->hw", energy, wl_label_patterns)
            labels_per_wl.append(label_wl)

        fig = plt.figure(figsize=(4 * (1 + 2 * Lloc), 10))
        gs = fig.add_gridspec(
            3, 1 + 2 * Lloc,
            height_ratios=[1.0, 1.0, 0.55],
            hspace=0.35, wspace=0.25,
        )

        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(I_in, cmap="inferno")
        ax0.set_title("Input |E|^2", fontsize=10, fontweight="bold")
        ax0.axis("off")

        for li in range(Lloc):
            wl_regions = [evaluation_regions[k * Lloc + li] for k in range(num_modes)]

            ax_label = fig.add_subplot(gs[0, 1 + 2 * li])
            ax_label.imshow(labels_per_wl[li], cmap="inferno")
            ax_label.set_title(f"Label λ={wavelengths[li] * 1e9:.0f}nm", fontsize=10, fontweight="bold")
            ax_label.axis("off")

            axI = fig.add_subplot(gs[0, 2 + 2 * li])
            I_li = I_pred[0, li].detach().cpu().numpy()
            axI.imshow(I_li, cmap="inferno")
            axI.set_title(f"Pred λ={wavelengths[li] * 1e9:.0f}nm", fontsize=10, fontweight="bold")
            axI.axis("off")

            for (x0, x1, y0, y1) in wl_regions:
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                axI.add_patch(Circle((cx, cy), radius=detect_radius, linewidth=0.8,
                                     edgecolor="cyan", facecolor="none", linestyle=":", alpha=0.9))

            I_bhw = I_pred[:, li].to(torch.float32)
            pred_energy_frac = region_energy_fractions(
                I_bhw, wl_regions, detect_radius=detect_radius
            )[0].detach().cpu().numpy()

            axb = fig.add_subplot(gs[1, 1 + 2 * li: 3 + 2 * li])
            idx = np.arange(num_modes)
            width = 0.35
            axb.bar(idx - width / 2, true_energy_frac, width, label="True", alpha=0.8)
            axb.bar(idx + width / 2, pred_energy_frac, width, label="Pred", alpha=0.8)
            axb.set_ylim(0, 1.0)
            axb.set_xticks(idx)
            axb.set_xticklabels([f"M{i}" for i in idx])
            axb.grid(True, alpha=0.3, axis="y")
            axb.set_title(f"Energy Ratio (λ={wavelengths[li] * 1e9:.0f}nm)", fontsize=10)
            axb.set_ylabel("Energy Fraction")
            if li == 0:
                axb.legend(loc="upper right")

        fig.add_subplot(gs[1, 0]).axis("off")

        ax_wl = fig.add_subplot(gs[2, :])
        R = np.zeros((Lloc, Lloc), dtype=np.float64)
        for s in range(Lloc):
            src_img = I_pred[0, s].detach().cpu().numpy()
            E_st = np.zeros(Lloc, dtype=np.float64)
            for t in range(Lloc):
                t_regions = [evaluation_regions[m * Lloc + t] for m in range(num_modes)]
                total = 0.0
                for (x0, x1, y0, y1) in t_regions:
                    patch = src_img[y0:y1, x0:x1]
                    h, w = patch.shape
                    yy, xx = np.ogrid[:h, :w]
                    cy = (h - 1) / 2.0
                    cx = (w - 1) / 2.0
                    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= float(detect_radius) ** 2
                    total += float(patch[mask].sum())
                E_st[t] = total
            R[s] = E_st / (E_st.sum() + 1e-12)

        x_axis = np.arange(Lloc)
        group_width = 0.85
        bar_w = group_width / Lloc
        for t in range(Lloc):
            offset = (t - (Lloc - 1) / 2.0) * bar_w
            ax_wl.bar(x_axis + offset, R[:, t], width=bar_w * 0.95, alpha=0.85,
                      label=f"ROI@{wavelengths[t] * 1e9:.0f}nm")

        ax_wl.set_ylim(0.0, 1.0)
        ax_wl.set_ylabel("Pred Energy Ratio\n(over ROI wavelength sets)")
        ax_wl.set_xticks(x_axis)
        ax_wl.set_xticklabels([f"{w * 1e9:.0f}" for w in wavelengths])
        ax_wl.set_xlabel("Source wavelength (nm)")
        ax_wl.grid(True, axis="y", alpha=0.25)
        ax_wl.set_title("Predicted energy distribution of each source λ across wavelength-ROI sets", fontsize=11)
        ax_wl.legend(ncol=min(Lloc, 6), fontsize=9, loc="upper right")

        fig.suptitle(f"MultiWL Prediction Analysis - Sample {si}", fontsize=14, fontweight="bold", y=0.98)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        out_path = out_dir / f"{tag}_sample{si:04d}.png"
        fig.savefig(out_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


# ============================================================
# Mode context
# ============================================================
def build_mode_context(base_modes: np.ndarray, num_modes: int) -> dict:
    if base_modes.shape[2] < num_modes:
        raise ValueError("Requested modes exceed file modes.")
    mmf_data = base_modes[:, :, :num_modes].transpose(2, 0, 1)

    mmf_data_amp_norm = (np.abs(mmf_data) - np.min(np.abs(mmf_data))) / (
        (np.max(np.abs(mmf_data)) - np.min(np.abs(mmf_data))) + 1e-12
    )
    mmf_data = mmf_data_amp_norm * np.exp(1j * np.angle(mmf_data))

    if phase_option in [1, 2, 3, 5]:
        base_amplitudes_local, base_phases_local = generate_complex_weights(1000, num_modes, phase_option)
    elif phase_option == 4:
        base_amplitudes_local = np.eye(num_modes, dtype=np.float32)
        base_phases_local = np.eye(num_modes, dtype=np.float32)
    else:
        raise ValueError("Unsupported phase_option")

    return {
        "mmf_data_np": mmf_data,
        "mmf_data_ts": torch.from_numpy(mmf_data),
        "base_amplitudes": base_amplitudes_local,
        "base_phases": base_phases_local,
    }


# ============================================================
# Load eigenmodes
# ============================================================
eigenmodes_OM4 = load_complex_modes_from_mat("data/mmf_6modes_GRIN_141_PD1.20.mat", key="modes_field")
print("Loaded modes shape:", eigenmodes_OM4.shape, "dtype:", eigenmodes_OM4.dtype)

mode_context = build_mode_context(eigenmodes_OM4, num_modes)
MMF_data = mode_context["mmf_data_np"]
MMF_data_ts = mode_context["mmf_data_ts"]
base_amplitudes = mode_context["base_amplitudes"]
base_phases = mode_context["base_phases"]


# ============================================================
# 生成多波长标签模板
# ============================================================
print(f"\n{'=' * 60}")
print(f"Generating MultiWL Labels: {num_modes} modes × {L} wavelengths")
print(f"USE_MODEGROUP_LABEL  = {USE_MODEGROUP_LABEL}")
print(f"USE_SINGLEPOINT_LABEL = {USE_SINGLEPOINT_LABEL}")
print(f"{'=' * 60}")

if USE_MODEGROUP_LABEL:
    print(f"★ Using MODE-GROUP labels: groups = {MODE_GROUPS}")
    mmf_label_patterns, evaluation_regions = generate_detector_patterns_multiwl_modegroup(
        H=out_size, W=out_size,
        num_modes=num_modes, num_wavelengths=L,
        mode_groups=MODE_GROUPS,
        radius=MODEGROUP_RADIUS_PER_WL,
        pattern_mode=label_pattern_mode,
        show_debug=show_detection_overlap_debug,
        margin_ratio=0.2,
    )
elif USE_SINGLEPOINT_LABEL:
    print("★ Using SINGLE-POINT labels")
    mmf_label_patterns, evaluation_regions = generate_detector_patterns_multiwl_singlepoint(
        H=out_size, W=out_size,
        num_modes=num_modes, num_wavelengths=L,
        radius=SINGLEPOINT_RADIUS_PER_WL,
        pattern_mode=label_pattern_mode,
        show_debug=show_detection_overlap_debug,
        margin_ratio=0.2,
    )
else:
    print("★ Using DEFAULT multi-wl grid labels")
    mmf_label_patterns, evaluation_regions, _ = generate_detector_patterns_multiwl(
        H=out_size, W=out_size,
        num_modes=num_modes, num_wavelengths=L,
        radius=circle_focus_radius,
        pattern_mode=label_pattern_mode,
        show_debug=show_detection_overlap_debug,
        margin_ratio=0.2,
    )

MMF_Label_data = torch.from_numpy(mmf_label_patterns).to(torch.float32)
print(f"✔ Generated {len(evaluation_regions)} evaluation regions")


# Overlap debug
if show_detection_overlap_debug:
    detection_debug_dir = RUN_ROOT / "detection_region_debug"
    detection_debug_dir.mkdir(parents=True, exist_ok=True)
    overlap_map = np.zeros((out_size, out_size), dtype=np.float32)
    for (x0, x1, y0, y1) in evaluation_regions:
        overlap_map[y0:y1, x0:x1] += 1.0
    overlap_pixels = int(np.count_nonzero(overlap_map > 1.0 + 1e-6))
    max_overlap = float(overlap_map.max()) if overlap_map.size else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.zeros((out_size, out_size), dtype=np.float32), cmap="Greys")
    axes[0].set_title("MultiWL Detector Layout")
    axes[0].set_axis_off()

    detect_radius_eval = int(detectsize // 2)
    for idx_region, (x0, x1, y0, y1) in enumerate(evaluation_regions):
        mode_idx = idx_region // L
        wl_idx = idx_region % L
        color = plt.cm.tab10(wl_idx % 10)
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.0,
                         edgecolor=color, facecolor="none")
        axes[0].add_patch(rect)
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        axes[0].add_patch(Circle((cx, cy), radius=detect_radius_eval,
                                 linewidth=1.0, edgecolor=color, linestyle="--", fill=False))
        axes[0].text(cx, cy, f"M{mode_idx}W{wl_idx}", ha='center', va='center',
                     color='white', fontsize=7,
                     bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))

    im1 = axes[1].imshow(overlap_map, cmap="viridis")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Detector Coverage (overlap map)")
    axes[1].set_axis_off()

    overlap_plot_path = detection_debug_dir / f"multiwl_overlap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    fig.tight_layout()
    fig.savefig(overlap_plot_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if overlap_pixels > 0:
        print(f"⚠ Detection regions overlap: {overlap_pixels} pixels (max {max_overlap:.1f})")
    else:
        print("✔ No overlap between evaluation regions")
    print(f"✔ Overlap debug plot -> {overlap_plot_path}")


# ============================================================
# Dataset builders (多波长版本)
# ============================================================
def build_eigenmode_dataset_multiwl() -> tuple[list[TensorDataset], dict]:
    datasets_per_wl = []

    if phase_option == 4:
        num_samples = num_modes
        amplitudes = base_amplitudes[:num_samples]
        phases = base_phases[:num_samples]
    else:
        amplitudes = base_amplitudes
        phases = base_phases
        num_samples = amplitudes.shape[0]

    complex_weights = amplitudes * np.exp(1j * phases)
    complex_weights_ts = torch.from_numpy(complex_weights.astype(np.complex64))
    image_data = generate_fields_ts(
        complex_weights_ts, MMF_data_ts, num_samples, num_modes, field_size
    ).to(torch.complex64)

    # ★ 修复：只 pad 输入场到 layer_size，不涉及 label
    images_prepared = []
    for i in range(num_samples):
        img_padded = pad_field_to_layer(image_data[i], layer_size)
        if img_padded.ndim == 2:
            img_padded = img_padded.unsqueeze(0)
        images_prepared.append(img_padded.to(torch.complex64))
    image_tensor = torch.stack(images_prepared, dim=0)

    for wl_idx in range(L):
        label_indices = [k * L + wl_idx for k in range(num_modes)]
        wl_label_patterns = MMF_Label_data[:, :, label_indices]

        amp = torch.from_numpy(amplitudes.astype(np.float32))
        energy = amp ** 2
        label_img = torch.einsum('nm,hwm->nhw', energy, wl_label_patterns)
        label_img = label_img.unsqueeze(1).contiguous()

        amp_tensor = torch.from_numpy(np.asarray(amplitudes, dtype=np.float32))
        ds = TensorDataset(image_tensor, label_img, amp_tensor)
        datasets_per_wl.append(ds)

    meta = {"amplitudes": amplitudes, "phases": phases}
    return datasets_per_wl, meta

def build_superposition_dataset_multiwl(num_samples: int, rng_seed: int) -> tuple[list[TensorDataset], dict]:
    rng = np.random.RandomState(rng_seed)
    amplitudes = rng.uniform(0.0, 1.0, size=(num_samples, num_modes)).astype(np.float32)
    amplitudes = amplitudes / (np.linalg.norm(amplitudes, axis=1, keepdims=True) + 1e-12)

    if phase_option == 4:
        phases = np.zeros_like(amplitudes)
    else:
        phases = rng.uniform(0.0, 2 * np.pi, size=(num_samples, num_modes)).astype(np.float32)

    complex_weights = amplitudes * np.exp(1j * phases)
    complex_weights_ts = torch.from_numpy(complex_weights.astype(np.complex64))
    image_data = generate_fields_ts(
        complex_weights_ts, MMF_data_ts, num_samples, num_modes, field_size
    ).to(torch.complex64)

    # ★ 修复：只 pad 输入场到 layer_size，不涉及 label
    images_prepared = []
    for i in range(num_samples):
        img_padded = pad_field_to_layer(image_data[i], layer_size)
        if img_padded.ndim == 2:
            img_padded = img_padded.unsqueeze(0)
        images_prepared.append(img_padded.to(torch.complex64))
    image_tensor = torch.stack(images_prepared, dim=0)

    datasets_per_wl = []
    for wl_idx in range(L):
        label_indices = [k * L + wl_idx for k in range(num_modes)]
        wl_label_patterns = MMF_Label_data[:, :, label_indices]

        amp = torch.from_numpy(amplitudes.astype(np.float32))
        energy = amp ** 2
        label_img = torch.einsum('nm,hwm->nhw', energy, wl_label_patterns)
        label_img = label_img.unsqueeze(1).contiguous()

        amp_tensor = torch.from_numpy(np.asarray(amplitudes, dtype=np.float32))
        ds = TensorDataset(image_tensor, label_img, amp_tensor)
        datasets_per_wl.append(ds)

    meta = {"amplitudes": amplitudes, "phases": phases}
    return datasets_per_wl, meta

# ============================================================
# Train/Eval loop
# ============================================================
all_losses: list[list[float]] = []

comprehensive_metrics_per_layer: dict[int, dict] = {}

detect_radius_eval = int(detectsize // 2)

for num_layer in num_layer_option:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'=' * 70}\nTraining D2NNModelMultiWL with {num_layer} layers\n{'=' * 70}")

    # 构建数据集
    if training_dataset_mode == "eigenmode":
        train_datasets_per_wl, train_meta = build_eigenmode_dataset_multiwl()
    elif training_dataset_mode == "superposition":
        train_datasets_per_wl, train_meta = build_superposition_dataset_multiwl(
            num_superposition_train_samples, superposition_train_seed
        )
    else:
        raise ValueError("Unknown training_dataset_mode")

    if evaluation_mode == "eigenmode":
        test_datasets_per_wl, test_meta = build_eigenmode_dataset_multiwl()
    elif evaluation_mode == "superposition":
        test_datasets_per_wl, test_meta = build_superposition_dataset_multiwl(
            num_superposition_eval_samples, superposition_eval_seed
        )
    else:
        raise ValueError("Unknown evaluation_mode")

    # 模型
    model = D2NNModelMultiWL(
        num_layers=num_layer,
        layer_size=layer_size,
        z_layers=z_layers,
        z_prop=z_prop,
        pixel_size=pixel_size,
        wavelengths=wavelengths,
        device=device,
        padding_ratio=padding_ratio,
        z_input_to_first=float(z_input_to_first),
        base_wavelength_idx=base_wavelength_idx,
        out_size=out_size,
        padding_ratio_out=padding_ratio_out,
    ).to(device)

    losses: list[float] = []
    epoch_durations: list[float] = []
    t0 = time.time()

    g = torch.Generator()
    g.manual_seed(SEED)

    training_result = train_multiwl_staged(
        model=model,
        train_datasets_per_wl=train_datasets_per_wl,
        wavelengths=wavelengths,
        base_wavelength_idx=base_wavelength_idx,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        device=device,
        seed=SEED,
        scheduler_gamma=0.99,
        stage_ratios=[0.25, 0.25, 0.25, 0.25],
        verbose=True,
    )

    losses = training_result['losses']
    epoch_durations = training_result['epoch_durations']
    stage_info = training_result['stage_info']
    total_time = training_result['total_time']
    all_losses.append(losses)

    print_stage_summary(stage_info)

    training_output_dir = RUN_ROOT / "training_analysis"

    train_logs = save_training_curves(
        losses=losses,
        epoch_durations=epoch_durations,
        out_dir=training_output_dir,
        tag=f"multiwl_m{num_modes}_{L}wl_sp{wl_spacing_nm:.1f}nm_ls{layer_size}_{num_layer}L_{run_tag}",
        num_layers=num_layer,
    )
    print(f"✔ Training curves saved -> {train_logs['loss_plot']}")

    save_staged_training_info(
        stage_info,
        training_output_dir / f"stage_info_layers{num_layer}_{run_tag}.txt"
    )

    # checkpoint
    ckpt_dir = RUN_ROOT / "checkpoints"
    ckpt_path = ckpt_dir / f"multiwl_{num_layer}L_m{num_modes}_{L}wl_{wl_start_nm:.0f}nm_sp{wl_spacing_nm:.1f}nm.pth"

    save_checkpoint_multiwl(
        model,
        ckpt_path,
        meta={
            "num_layers": int(num_layer),
            "layer_size": int(layer_size),
            "num_modes": int(num_modes),
            "num_wavelengths": int(L),
            "wavelengths": wavelengths.astype(np.float32),
            "out_size": int(out_size),
            "padding_ratio_out": float(padding_ratio_out),
            "wl_start_nm": float(wl_start_nm),
            "wl_spacing_nm": float(wl_spacing_nm),
            "total_training_time_sec": float(total_time),
        },
    )
    print("✔ Checkpoint saved ->", ckpt_path)

    # phase_masks
    phase_masks = extract_phase_masks_multiwl(model)
    if phase_masks:
        # ★ 完整参数命名规则
        # 包含：模式数、层数、波长信息、画布尺寸、传播距离、像素尺寸、ROI 参数、时间戳
        param_str = (
            f"m{num_modes}"                        # 模式数
            f"_{num_layer}L"                        # 衍射层数
            f"_{L}wl{wl_start_nm:.0f}nm"            # 波长数 + 起始波长
            f"sp{wl_spacing_nm:.0f}nm"              # 波长间隔
            f"_ls{layer_size}"                      # 层画布尺寸
            f"_out{out_size}"                       # 输出画布尺寸
            f"_pr{padding_ratio}"                   # padding 比例
            f"_zin{z_input_to_first*1e3:.1f}mm"     # 输入到第一层
            f"_zL{z_layers*1e3:.1f}mm"              # 层间距
            f"_zp{z_prop*1e3:.0f}mm"                # 最后一段传播
            f"_dx{pixel_size*1e6:.1f}um"            # 像素尺寸
            f"_r{circle_focus_radius}"              # ROI 半径
            f"_d{circle_detectsize}"                # 探测器尺寸
            f"_{run_tag}"                           # 时间戳
        )

        pm_dir = RUN_ROOT / "phase_masks" / f"{num_layer}L_run{run_tag}"
        pm_dir.mkdir(parents=True, exist_ok=True)

        # ★ .mat 文件名（带完整参数）
        pm_mat = pm_dir / f"phase_masks_{param_str}.mat"
        # ★ PNG 基础名（visualize_phase_masks 会自动加 _layer{i}.png 后缀）
        base_name = f"mask_{param_str}"

        # ★ 保存 .mat 时一并存入参数 dict，方便后续读取还原
        mat_payload = {
            "phase_masks": np.stack(phase_masks, axis=0).astype(np.float32),
            # 元数据（标量都包成 1D array 以便 MATLAB 读取）
            "num_modes": np.array([num_modes], dtype=np.int32),
            "num_layers": np.array([num_layer], dtype=np.int32),
            "num_wavelengths": np.array([L], dtype=np.int32),
            "wavelengths_nm": (wavelengths * 1e9).astype(np.float64),
            "wl_start_nm": np.array([wl_start_nm], dtype=np.float64),
            "wl_spacing_nm": np.array([wl_spacing_nm], dtype=np.float64),
            "base_wavelength_idx": np.array([base_wavelength_idx], dtype=np.int32),
            "layer_size": np.array([layer_size], dtype=np.int32),
            "out_size": np.array([out_size], dtype=np.int32),
            "pixel_size_m": np.array([pixel_size], dtype=np.float64),
            "padding_ratio": np.array([padding_ratio], dtype=np.float64),
            "padding_ratio_out": np.array([padding_ratio_out], dtype=np.float64),
            "z_input_to_first_m": np.array([z_input_to_first], dtype=np.float64),
            "z_layers_m": np.array([z_layers], dtype=np.float64),
            "z_prop_m": np.array([z_prop], dtype=np.float64),
            "circle_focus_radius_px": np.array([circle_focus_radius], dtype=np.int32),
            "circle_detectsize_px": np.array([circle_detectsize], dtype=np.int32),
            "epochs": np.array([epochs], dtype=np.int32),
            "lr": np.array([lr], dtype=np.float64),
            "batch_size": np.array([batch_size], dtype=np.int32),
            "seed": np.array([SEED], dtype=np.int32),
            "timestamp": np.array([run_tag]),
        }

        savemat(str(pm_mat), mat_payload)
        print(f"✔ Phase masks saved -> {pm_mat}")

        # ★ PNG 可视化
        png_paths = visualize_phase_masks(
            phase_masks,
            out_dir=pm_dir,
            base_name=base_name,
            save_degree=False,
            dpi=300,
            cmap="twilight",
            show_stats=True,
        )
        print(f"✔ Generated {len(png_paths)} phase mask PNGs -> {pm_dir}")


    # 预测可视化
    diag_dir = RUN_ROOT / "prediction_viz" / f"{num_layer}L_{run_tag}"
    n_vis = min(num_pred_diag_samples, len(test_datasets_per_wl[0]))
    diag_paths = save_prediction_diagnostics_multiwl(
        model,
        test_datasets_per_wl[0],
        wavelengths=wavelengths,
        evaluation_regions=evaluation_regions,
        detect_radius=detect_radius_eval,
        sample_indices=list(range(n_vis)),
        out_dir=diag_dir,
        device=device,
        tag=f"pred_L{num_layer}",
        num_modes=num_modes,
    )
    if diag_paths:
        print(f"✔ Prediction diagnostics ({len(diag_paths)}) -> {diag_paths[0].parent}")

    # ============================================================
    # Per-layer propagation slices (multi-wavelength)
    # ============================================================
    slice_dir = RUN_ROOT / "propagation_slices" / f"L{num_layer}_{run_tag}"
    slice_dir.mkdir(parents=True, exist_ok=True)

    eigen_ds = test_datasets_per_wl[0]
    n_modes_to_dump = min(num_modes, len(eigen_ds))

    for mode_idx in range(n_modes_to_dump):
        img_complex, _label_img, _amp = eigen_ds[mode_idx]
        if img_complex.ndim == 3:
            eigen_field = img_complex.squeeze(0)
        else:
            eigen_field = img_complex

        info = capture_eigenmode_propagation_multiwl(
            model,
            eigen_field,
            mode_index=mode_idx,
            layer_size=layer_size,
            z_input_to_first=float(z_input_to_first),
            z_layers=float(z_layers),
            z_prop=float(z_prop),
            pixel_size=float(pixel_size),
            wavelengths=wavelengths,
            output_dir=slice_dir,
            tag=f"L{num_layer}_mode{mode_idx+1}",
            base_wavelength_idx=base_wavelength_idx,
            fractions_between_layers=tuple(
                (1/4, 1/2, 3/4) for _ in range(num_layer)
            ),
            output_fractions=(0.2, 0.4, 0.6, 0.8),
            out_size=out_size,
            padding_ratio_out=padding_ratio_out,
        )
        print(f"  ✔ mode {mode_idx+1} slices -> {info['fig_path']}")
        print(f"  ✔ mode {mode_idx+1} .mat   -> {info['mat_path']}")

    # ============================================================
    # Metrics: SNR / Isolation / Crosstalk / Insertion Loss
    # ============================================================
    comp_metrics = evaluate_multiwl_comprehensive_metrics(
        model,
        evaluation_regions,
        detect_radius=detect_radius_eval,
        device=device,
        num_modes=num_modes,
        num_wavelengths=L,
        wavelengths_m=wavelengths,
        mmf_modes=MMF_data_ts,
        layer_size=layer_size,
    )
    comprehensive_metrics_per_layer[int(num_layer)] = comp_metrics
    print(
        f"\n[{num_layer} layers] "
        f"SNR={comp_metrics['snr_db_mean']:.2f} dB, "
        f"Mode Iso={comp_metrics['mode_isolation_db_mean']:.2f} dB, "
        f"WL Iso={comp_metrics['wavelength_isolation_db_mean']:.2f} dB, "
        f"IL={comp_metrics['insertion_loss_db_mean']:.2f} dB"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    # ============================================================
    # Group Isolation (only meaningful when USE_MODEGROUP_LABEL=True)
    # ============================================================
    if USE_MODEGROUP_LABEL:
        group_metrics = evaluate_group_isolation_db(
            model=model,
            mode_groups=MODE_GROUPS,
            evaluation_regions=evaluation_regions,
            detect_radius=detect_radius_eval,
            num_modes=num_modes,
            num_wavelengths=L,
            wavelengths_m=wavelengths,
            mmf_modes=MMF_data_ts,
            layer_size=layer_size,
            device=device,
        )
        comprehensive_metrics_per_layer[int(num_layer)]["group_isolation"] = group_metrics

        print(f"[{num_layer} layers] Group Iso = {group_metrics['mean_db']:.2f} dB "
              f"({len(MODE_GROUPS)} groups)")

        # 可视化：每层一张 group crosstalk 热力图
        gi_dir = RUN_ROOT / "metrics_analysis" / f"L{num_layer}" / "group_isolation"
        gi_dir.mkdir(parents=True, exist_ok=True)
        for wl_idx in range(L):
            cm = group_metrics["group_crosstalk_matrix"][wl_idx]
            cm_db = 10.0 * np.log10(np.clip(cm, 1e-6, None))
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            im = ax.imshow(cm_db, cmap="magma", vmin=-30, vmax=0)
            ax.set_title(f"Group Crosstalk (dB) — L={num_layer}, "
                         f"λ={wavelengths[wl_idx]*1e9:.0f} nm")
            ax.set_xlabel("ROI group"); ax.set_ylabel("Source group")
            ax.set_xticks(range(len(MODE_GROUPS)))
            ax.set_yticks(range(len(MODE_GROUPS)))
            tick_labels = [f"G{g+1}\n[{','.join(str(m+1) for m in MODE_GROUPS[g])}]"
                           for g in range(len(MODE_GROUPS))]
            ax.set_xticklabels(tick_labels, fontsize=8)
            ax.set_yticklabels(tick_labels, fontsize=8)
            for r in range(len(MODE_GROUPS)):
                for c in range(len(MODE_GROUPS)):
                    ax.text(c, r, f"{cm_db[r,c]:.1f}", ha="center", va="center",
                            color="white" if cm_db[r,c] < -15 else "black", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            fig.savefig(gi_dir / f"group_crosstalk_wl{wl_idx}_{wavelengths[wl_idx]*1e9:.0f}nm.png",
                        dpi=250, bbox_inches="tight")
            plt.close(fig)

        # per-mode bar
        fig, ax = plt.subplots(figsize=(8, 4))
        x_axis = np.arange(num_modes)
        width = 0.8 / L
        for wl_idx in range(L):
            ax.bar(x_axis + (wl_idx - (L-1)/2)*width,
                   group_metrics["per_mode_per_wl_db"][:, wl_idx],
                   width=width*0.95,
                   label=f"λ={wavelengths[wl_idx]*1e9:.0f} nm")
        ax.set_xticks(x_axis)
        ax.set_xticklabels([f"M{m+1}\n→G{MODE_GROUPS_idx_of := list({m: g for g,ms in enumerate(MODE_GROUPS) for m in ms}.get(m_i)+1 for m_i in range(num_modes))[m]}"
                            if False else f"M{m+1}" for m in range(num_modes)])
        ax.set_ylabel("Group Isolation (dB)")
        ax.set_title(f"Per-mode Group Isolation — {num_layer} layers "
                     f"(mean = {group_metrics['mean_db']:.2f} dB)")
        ax.axhline(group_metrics['mean_db'], color='red', linestyle='--', alpha=0.6,
                   label=f"mean = {group_metrics['mean_db']:.2f} dB")
        ax.grid(True, axis='y', alpha=0.3); ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(gi_dir / "per_mode_group_isolation.png", dpi=250, bbox_inches="tight")
        plt.close(fig)

print("\n" + "=" * 70)
print("All training completed!")
print("=" * 70)


# ============================================================
# 保存指标分析
# ============================================================
metrics_dir = RUN_ROOT / "metrics_analysis"
metrics_dir.mkdir(parents=True, exist_ok=True)
tag = datetime.now().strftime("%Y%m%d_%H%M%S")

sorted_layers = sorted(comprehensive_metrics_per_layer.keys())

if len(sorted_layers) == 0:
    print("⚠ No metrics were computed! Check that training loop completed.")
else:
    layer_counts = np.asarray(sorted_layers, dtype=np.int32)

    # ---- 汇总核心指标 ----
    snr_mean_arr = np.array([comprehensive_metrics_per_layer[nl]["snr_db_mean"] for nl in sorted_layers])
    mode_iso_arr = np.array([comprehensive_metrics_per_layer[nl]["mode_isolation_db_mean"] for nl in sorted_layers])
    wl_iso_arr = np.array([comprehensive_metrics_per_layer[nl]["wavelength_isolation_db_mean"] for nl in sorted_layers])
    target_all_roi_db_arr = np.array([
        float(np.mean(comprehensive_metrics_per_layer[nl]["target_all_roi_db"]))
        for nl in sorted_layers
    ])
    il_arr = np.array([comprehensive_metrics_per_layer[nl]["insertion_loss_db_mean"] for nl in sorted_layers])
    has_group_iso = USE_MODEGROUP_LABEL and all(
        "group_isolation" in comprehensive_metrics_per_layer[nl]
        for nl in sorted_layers
    )
    if has_group_iso:
        group_iso_arr = np.array([
            comprehensive_metrics_per_layer[nl]["group_isolation"]["mean_db"]
            for nl in sorted_layers
        ])
        # 单独折线图
        fig, ax = plt.subplots(figsize=(7.5, 4.5))
        ax.plot(layer_counts, group_iso_arr, marker="D", color="tab:brown",
                linewidth=2, markersize=8)
        ax.set_xlabel("Number of Layers", fontsize=12)
        ax.set_ylabel("Group Isolation (dB)", fontsize=12)
        ax.set_title(f"Mode-Group Isolation vs. Layer Count\n"
                     f"groups = {MODE_GROUPS}", fontsize=11)
        ax.set_xticks(layer_counts); ax.grid(True, alpha=0.3, linestyle="--")
        for x, y in zip(layer_counts, group_iso_arr):
            if np.isfinite(y):
                ax.annotate(f"{y:.2f} dB", (x, y), textcoords="offset points",
                            xytext=(0, 10), ha="center", fontsize=10, color="tab:brown")
        fig.tight_layout()
        fig.savefig(metrics_dir / f"metric_group_isolation_db_{tag}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✔ Group Isolation line chart saved")

    # ---- 5-panel 汇总图 ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) SNR
    axes[0, 0].plot(layer_counts, snr_mean_arr, marker="o", color="tab:blue", linewidth=2)
    axes[0, 0].set_title("SNR (dB)")
    axes[0, 0].set_xlabel("Layers"); axes[0, 0].set_ylabel("dB")
    axes[0, 0].grid(True, alpha=0.3); axes[0, 0].set_xticks(layer_counts)
    for x, y in zip(layer_counts, snr_mean_arr):
        if np.isfinite(y):
            axes[0, 0].annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=9)

    # (0,1) 三维 Isolation 合并对比
    axes[0, 1].plot(layer_counts, mode_iso_arr, marker="o", color="tab:green",
                    linewidth=2, label="① Same-λ Mode Iso")
    axes[0, 1].plot(layer_counts, wl_iso_arr, marker="s", linestyle="--", color="tab:orange",
                    linewidth=2, label="② Same-Mode WL Iso")
    axes[0, 1].plot(layer_counts, target_all_roi_db_arr, marker="^", linestyle=":",
                    color="tab:purple", linewidth=2, label="③ Target/All ROI")
    axes[0, 1].set_title("Three Isolation Dimensions (dB)")
    axes[0, 1].set_xlabel("Layers"); axes[0, 1].set_ylabel("dB")
    axes[0, 1].legend(fontsize=9); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xticks(layer_counts)

    # (0,2) Insertion Loss
    axes[0, 2].plot(layer_counts, il_arr, marker="o", color="tab:red", linewidth=2)
    axes[0, 2].set_title("Insertion Loss (dB)")
    axes[0, 2].set_xlabel("Layers"); axes[0, 2].set_ylabel("dB")
    axes[0, 2].grid(True, alpha=0.3); axes[0, 2].set_xticks(layer_counts)
    for x, y in zip(layer_counts, il_arr):
        if np.isfinite(y):
            axes[0, 2].annotate(f"{y:.2f}", (x, y), textcoords="offset points",
                                xytext=(0, 8), ha="center", fontsize=9)

    # (1,0) Crosstalk heatmap (mode, last layer)
    best_nl = sorted_layers[-1]
    ct_mat = comprehensive_metrics_per_layer[best_nl]["crosstalk_matrix_per_wl"]  # (L, M, M)
    ct_mean = ct_mat.mean(axis=0)
    ct_db = 10.0 * np.log10(np.clip(ct_mean, 1e-6, None))
    im = axes[1, 0].imshow(ct_db, cmap="magma", vmin=-30, vmax=0)
    axes[1, 0].set_title(f"Mode Crosstalk (dB) — {best_nl} layers")
    axes[1, 0].set_xlabel("ROI mode"); axes[1, 0].set_ylabel("Input mode")
    axes[1, 0].set_xticks(range(num_modes)); axes[1, 0].set_yticks(range(num_modes))
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    for r in range(num_modes):
        for c in range(num_modes):
            axes[1, 0].text(c, r, f"{ct_db[r, c]:.0f}", ha="center", va="center",
                            color="white" if ct_db[r, c] < -15 else "black", fontsize=9)

    # (1,1) Wavelength crosstalk heatmap (last layer, mode 0)
    ct_wl = comprehensive_metrics_per_layer[best_nl]["crosstalk_matrix_wl"]  # (M, L, L)
    ct_wl_mean = ct_wl.mean(axis=0)  # 跨 mode 平均 (L, L)
    ct_wl_db = 10.0 * np.log10(np.clip(ct_wl_mean, 1e-6, None))
    wl_nm = (wavelengths * 1e9).astype(np.float64)
    im2 = axes[1, 1].imshow(ct_wl_db, cmap="magma", vmin=-30, vmax=0)
    axes[1, 1].set_title(f"WL Crosstalk (dB) — {best_nl} layers")
    axes[1, 1].set_xlabel("Target λ"); axes[1, 1].set_ylabel("Source λ")
    axes[1, 1].set_xticks(range(L))
    axes[1, 1].set_xticklabels([f"{w:.0f}" for w in wl_nm], fontsize=8)
    axes[1, 1].set_yticks(range(L))
    axes[1, 1].set_yticklabels([f"{w:.0f}" for w in wl_nm], fontsize=8)
    fig.colorbar(im2, ax=axes[1, 1], fraction=0.046, pad=0.04)
    for r in range(L):
        for c in range(L):
            axes[1, 1].text(c, r, f"{ct_wl_db[r, c]:.0f}", ha="center", va="center",
                            color="white" if ct_wl_db[r, c] < -15 else "black", fontsize=9)

    # (1,2) Target/All ROI per mode (bar chart, last layer)
    tar_ratio = comprehensive_metrics_per_layer[best_nl]["target_all_roi_ratio"]  # (M,)
    axes[1, 2].bar(range(num_modes), tar_ratio, color="tab:purple", alpha=0.8)
    axes[1, 2].set_xticks(range(num_modes))
    axes[1, 2].set_xticklabels([f"M{m+1}" for m in range(num_modes)])
    axes[1, 2].set_ylabel("Target / All ROI")
    axes[1, 2].set_title(f"Target/All ROI — {best_nl} layers")
    axes[1, 2].set_ylim(0, 1.05)
    axes[1, 2].axhline(float(np.mean(tar_ratio)), color="red", linestyle="--",
                        label=f"mean={float(np.mean(tar_ratio)):.4f}")
    axes[1, 2].legend(fontsize=9)
    axes[1, 2].grid(axis="y", alpha=0.3)

    fig.suptitle("Core Metrics vs. Number of Layers", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    summary_path = metrics_dir / f"core_metrics_summary_{tag}.png"
    fig.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Core metrics summary -> {summary_path}")

    # ============================================================
    # 单独折线图（每个指标一张大图）
    # ============================================================

    # 1. SNR
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(layer_counts, snr_mean_arr, marker="o", color="tab:red", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Layers", fontsize=12)
    ax.set_ylabel("SNR (dB)", fontsize=12)
    ax.set_title("Signal-to-Noise Ratio vs. Layer Count", fontsize=13)
    ax.set_xticks(layer_counts); ax.grid(True, alpha=0.3, linestyle="--")
    for x, y in zip(layer_counts, snr_mean_arr):
        if np.isfinite(y):
            ax.annotate(f"{y:.2f} dB", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10, color="tab:red")
    fig.tight_layout()
    fig.savefig(metrics_dir / f"metric_snr_db_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ SNR line chart saved")

    # 2. Same-λ Mode Isolation (同波长内的 isolation)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(layer_counts, mode_iso_arr, marker="o", color="tab:green", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Layers", fontsize=12)
    ax.set_ylabel("Mode Isolation (dB)", fontsize=12)
    ax.set_title("Same-Wavelength Mode Isolation vs. Layer Count\n"
                 r"$10\log_{10}\left(\frac{E[m,l,m]}{\sum_{j\neq m} E[m,l,j]}\right)$",
                 fontsize=12)
    ax.set_xticks(layer_counts); ax.grid(True, alpha=0.3, linestyle="--")
    for x, y in zip(layer_counts, mode_iso_arr):
        if np.isfinite(y):
            ax.annotate(f"{y:.2f} dB", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10, color="tab:green")
    fig.tight_layout()
    fig.savefig(metrics_dir / f"metric_mode_isolation_db_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Same-λ Mode Isolation line chart saved")

    # 3. Same-Mode Wavelength Isolation (同 mode 内的 isolation)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(layer_counts, wl_iso_arr, marker="s", color="tab:orange", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Layers", fontsize=12)
    ax.set_ylabel("Wavelength Isolation (dB)", fontsize=12)
    ax.set_title("Same-Mode Wavelength Isolation vs. Layer Count\n"
                 r"$10\log_{10}\left(\frac{E[m,l,m]}{\sum_{l'\neq l} E[m,l',m]}\right)$",
                 fontsize=12)
    ax.set_xticks(layer_counts); ax.grid(True, alpha=0.3, linestyle="--")
    for x, y in zip(layer_counts, wl_iso_arr):
        if np.isfinite(y):
            ax.annotate(f"{y:.2f} dB", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10, color="tab:orange")
    fig.tight_layout()
    fig.savefig(metrics_dir / f"metric_wl_isolation_db_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Same-Mode WL Isolation line chart saved")

    # 4. Target / All ROI Isolation
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(layer_counts, target_all_roi_db_arr, marker="^", color="tab:purple",
            linewidth=2, markersize=8)
    ax.set_xlabel("Number of Layers", fontsize=12)
    ax.set_ylabel("Target/All ROI (dB)", fontsize=12)
    ax.set_title("Target vs. All ROI Isolation vs. Layer Count\n"
                 r"$10\log_{10}\left(\frac{E_{target}}{E_{all} - E_{target}}\right)$",
                 fontsize=12)
    ax.set_xticks(layer_counts); ax.grid(True, alpha=0.3, linestyle="--")
    for x, y in zip(layer_counts, target_all_roi_db_arr):
        if np.isfinite(y):
            ax.annotate(f"{y:.2f} dB", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10, color="tab:purple")
    fig.tight_layout()
    fig.savefig(metrics_dir / f"metric_target_all_roi_db_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Target/All ROI Isolation line chart saved")

    # 5. Insertion Loss
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(layer_counts, il_arr, marker="o", color="tab:cyan", linewidth=2, markersize=8)
    ax.set_xlabel("Number of Layers", fontsize=12)
    ax.set_ylabel("Insertion Loss (dB)", fontsize=12)
    ax.set_title("Insertion Loss vs. Layer Count\n"
                 r"$\mathrm{IL} = -10\log_{10}(E_{out}/E_{in})$", fontsize=12)
    ax.set_xticks(layer_counts); ax.grid(True, alpha=0.3, linestyle="--")
    for x, y in zip(layer_counts, il_arr):
        if np.isfinite(y):
            ax.annotate(f"{y:.2f} dB", (x, y), textcoords="offset points",
                        xytext=(0, 10), ha="center", fontsize=10, color="tab:cyan")
    fig.tight_layout()
    fig.savefig(metrics_dir / f"metric_insertion_loss_db_{tag}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ Insertion Loss line chart saved")

    # ---- 每个 num_layer 单独保存详细可视化和 .mat ----
    for nl in sorted_layers:
        m = comprehensive_metrics_per_layer[nl]
        plot_and_save_multiwl_metrics(
            m,
            output_dir=metrics_dir / f"L{nl}",
            tag=f"L{nl}_{tag}",
            num_modes=num_modes,
            num_wavelengths=L,
        )

    # ---- 汇总 .mat ----
    # ---- 汇总 .mat ----
    savemat_dict = {
        "layers": layer_counts.astype(np.float64),
        "wavelengths_m": wavelengths.astype(np.float64),
        "snr_db_mean": snr_mean_arr,
        # 三维 isolation
        "mode_isolation_db_mean": mode_iso_arr,
        "wavelength_isolation_db_mean": wl_iso_arr,
        "target_all_roi_db_mean": target_all_roi_db_arr,
        # insertion loss
        "insertion_loss_db_mean": il_arr,
    }

    # ---- Mode-group isolation（仅在启用 mode group 标签时附加） ----
    has_group_iso = USE_MODEGROUP_LABEL and all(
        "group_isolation" in comprehensive_metrics_per_layer[nl]
        for nl in sorted_layers
    )
    if has_group_iso:
        group_iso_arr = np.array([
            comprehensive_metrics_per_layer[nl]["group_isolation"]["mean_db"]
            for nl in sorted_layers
        ], dtype=np.float64)

        savemat_dict["group_isolation_db_mean"] = group_iso_arr
        savemat_dict["mode_groups_flat"] = np.array(
            [m for g in MODE_GROUPS for m in g], dtype=np.int32
        )
        savemat_dict["mode_groups_sizes"] = np.array(
            [len(g) for g in MODE_GROUPS], dtype=np.int32
        )
        savemat_dict["num_mode_groups"] = np.array([len(MODE_GROUPS)], dtype=np.int32)

        # 可选：把每层的 group crosstalk matrix 也一起存（shape: (num_layers, L, G, G)）
        G = len(MODE_GROUPS)
        ct_stack = np.stack([
            comprehensive_metrics_per_layer[nl]["group_isolation"]["group_crosstalk_matrix"]
            for nl in sorted_layers
        ], axis=0)  # (num_layers, L, G, G)
        savemat_dict["group_crosstalk_matrix_per_layer"] = ct_stack

        print(f"  ✔ Group isolation data appended to .mat "
              f"(mean over layers: {group_iso_arr.mean():.2f} dB)")

    savemat(str(metrics_dir / f"metrics_summary_{tag}.mat"), savemat_dict)
    print(f"✔ Metrics .mat saved -> {metrics_dir / f'metrics_summary_{tag}.mat'}")

print("\n✅ All outputs saved successfully!")
