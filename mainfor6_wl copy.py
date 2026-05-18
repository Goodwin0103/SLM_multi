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
from odnn_processing import prepare_sample
from odnn_training_eval import spot_energy_and_snr
# MultiWL model
from odnn_multiwl_model import D2NNModelMultiWL

from odnn_training_io import train_multiwl_staged, print_stage_summary, save_staged_training_info

from odnn_training_visualization import visualize_phase_masks

from odnn_training_visualization import (
    capture_eigenmode_propagation,
    export_superposition_slices,
    plot_amplitude_comparison_grid,
    plot_reconstruction_vs_input,
    plot_sys_vs_label_strict,
    save_superposition_triptych,
    save_mode_triptych,
    visualize_model_slices,
)

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

device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
print("Using Device:", device)


# ============================================================
# Parameters
# ============================================================
field_size = 176
layer_size = 300
num_modes = 10

circle_focus_radius = 10
circle_detectsize = 20
focus_radius = circle_focus_radius
detectsize = circle_detectsize

batch_size = 16

evaluation_mode = "eigenmode"      # "eigenmode" or "superposition"
training_dataset_mode = "eigenmode"    # "eigenmode" or "superposition"

num_superposition_eval_samples = 1000
num_superposition_train_samples = 100
superposition_eval_seed = 20240116
superposition_train_seed = 20240115

num_layer_option = [1, 2, 3, 4, 5]
# num_layer_option = [7 ]

# SLM
z_layers = 49.465e-3
pixel_size = 12.5e-6
z_prop = 20e-2
z_input_to_first = 0

# wavelengths (MultiWL)
wavelengths = np.array([532e-9, 650e-9], dtype=np.float32)
# wavelengths = np.array([1530e-9, 1540e-9, 1550e-9, 1560e-9], dtype=np.float32)
# wavelengths = np.array([1530e-9, 1535e-9, 1540e-9], dtype=np.float32)
# wavelengths = np.array([1530e-9, 1535e-9, 1540e-9, 1545e-9, 1550e-9, 1555e-9, 1560e-9, 1565e-9], dtype=np.float32)
base_wavelength_idx = 1
L = int(len(wavelengths))

# data options
phase_option = 4
label_pattern_mode = "circle"
show_detection_overlap_debug = True

# train hyperparams
epochs = 1000
lr = 1.99
padding_ratio = 0.5

# output root
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# 格式: 20260226_143052

RUN_ROOT = Path(f"results/10modes/eigenmode/1530_1565_base_{base_wavelength_idx}_weighted_{timestamp}")
RUN_ROOT.mkdir(parents=True, exist_ok=True)

# prediction viz samples
num_pred_diag_samples = 3
num_superposition_visual_samples = 2


# ============================================================
# 新增：多波长标签生成函数
# ============================================================
def generate_detector_patterns_multiwl(
    H: int,
    W: int,
    num_modes: int,
    num_wavelengths: int,
    radius: int,
    pattern_mode: str = "circle",
    show_debug: bool = False,
    margin_ratio: float = 0.2,  # 🆕 新增参数：边距比例（默认25%）
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    生成多波长标签图案（更集中的布局）
    
    Args:
        H, W: 图像尺寸
        num_modes: 模式数
        num_wavelengths: 波长数
        radius: 标签半径
        pattern_mode: 图案模式
        show_debug: 是否显示调试图
        margin_ratio: 边距比例（0.0-0.5），值越大标签越集中
    
    Returns:
        patterns: (H, W, num_modes * num_wavelengths)
        evaluation_regions: [(x0, x1, y0, y1), ...]
    """
    total_labels = num_modes * num_wavelengths
    
    # ===== 新布局：每个 mode 一行，每个波长一列 =====
    num_rows = num_modes
    num_cols = num_wavelengths

    # 🔑 关键修改：使用比例边距，使标签更集中
    margin_x = int(W * margin_ratio)  # 水平边距
    margin_y = int(H * margin_ratio)  # 垂直边距
    
    # 确保边距至少能容纳标签
    min_margin = radius + 5
    margin_x = max(margin_x, min_margin)
    margin_y = max(margin_y, min_margin)
    
    # 在缩小的区域内均匀分布
    xs = np.linspace(margin_x, W - 1 - margin_x, num_cols)
    ys = np.linspace(margin_y, H - 1 - margin_y, num_rows)

    # centers 顺序：idx = mode * L + wl（mode-major）
    centers = []
    for mode_idx in range(num_rows):
        for wl_idx in range(num_cols):
            cx = int(round(xs[wl_idx]))
            cy = int(round(ys[mode_idx]))
            centers.append((cy, cx))
    
    # 生成图案
    if pattern_mode == "circle":
        patterns = np.zeros((H, W, total_labels), dtype=np.float32)
        for idx, (cy, cx) in enumerate(centers):
            yy, xx = np.ogrid[:H, :W]
            mask = (yy - cy)**2 + (xx - cx)**2 <= radius**2
            patterns[:, :, idx] = mask.astype(np.float32)
    else:
        raise NotImplementedError(f"Unsupported pattern_mode: {pattern_mode}")
    
    # 生成评估区域
    evaluation_regions = []
    for cy, cx in centers:
        x0 = max(0, int(cx - radius))
        x1 = min(W, int(cx + radius))
        y0 = max(0, int(cy - radius))
        y1 = min(H, int(cy + radius))
        evaluation_regions.append((x0, x1, y0, y1))
    
    if show_debug:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(patterns.sum(axis=2), cmap='gray')
        for idx, (cy, cx) in enumerate(centers):
            mode_idx = idx // num_wavelengths
            wl_idx = idx % num_wavelengths
            ax.text(cx, cy, f"M{mode_idx}W{wl_idx}", 
                   ha='center', va='center', color='red', fontsize=8)
        
        # 🆕 显示边距信息
        ax.axvline(margin_x, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(W - margin_x, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(margin_y, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(H - margin_y, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
        
        plt.title(f"MultiWL Labels: {num_modes} modes × {num_wavelengths} wavelengths\n"
                 f"Margin: {margin_ratio*100:.0f}% ({margin_x}×{margin_y} pixels)")
        plt.savefig(RUN_ROOT / "debug_multiwl_labels.png", dpi=150)
        plt.close()
        print(f"✔ Debug label layout saved -> {RUN_ROOT / 'debug_multiwl_labels.png'}")
    
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


def _make_circle_mask(h: int, w: int, r: float, device: torch.device) -> torch.Tensor:
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
    """
    计算每个区域的能量比例 (B, M)
    """
    B, H, W = I_bhw.shape
    M = len(evaluation_regions)
    out = torch.zeros((B, M), device=I_bhw.device, dtype=torch.float32)

    for mi, (x0, x1, y0, y1) in enumerate(evaluation_regions):
        patch = I_bhw[:, y0:y1, x0:x1]
        hh, ww = patch.shape[-2], patch.shape[-1]
        cmask = _make_circle_mask(hh, ww, float(detect_radius), device=I_bhw.device)
        out[:, mi] = (patch * cmask.unsqueeze(0)).sum(dim=(-1, -2))

    out = out / (out.sum(dim=1, keepdim=True) + 1e-12)
    return out

@torch.no_grad()
def evaluate_target_wl_over_all_wl_roi_ratio(
    model: D2NNModelMultiWL,
    loader: DataLoader,
    *,
    device: torch.device,
    evaluation_regions: list[tuple[int,int,int,int]],
    detect_radius: int,
    L: int,
    num_modes: int,
) -> dict:
    """
    修正版：对每个源波长s，计算它在"自己的ROI集合"中的能量占比
    ratio_s = E(s在s的ROI) / E(s在所有ROI)
    """
    model.eval()

    ratio_list = []

    for images, label_img, amp in loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)
        if images.ndim == 3:
            images = images.unsqueeze(1)

        x = images.repeat(1, L, 1, 1).contiguous()
        I_blhw = model(x).to(torch.float32)  # (B,L,H,W)
        B = I_blhw.shape[0]

        ratios = torch.zeros((B, L), device=device, dtype=torch.float32)

        for s in range(L):
            src = I_blhw[:, s]  # (B,H,W) - 源波长s的输出

            # 🔑 关键修改：计算在"每个波长ROI集合"中的能量
            E_in_each_wl_roi = torch.zeros((B, L), device=device, dtype=torch.float32)
            
            for t in range(L):
                # 波长t的ROI集合（3个mode）
                t_regions = [evaluation_regions[m * L + t] for m in range(num_modes)]
                
                total = torch.zeros((B,), device=device, dtype=torch.float32)
                for (x0, x1, y0, y1) in t_regions:
                    patch = src[:, y0:y1, x0:x1]
                    hh, ww = patch.shape[-2], patch.shape[-1]
                    cmask = _make_circle_mask(hh, ww, float(detect_radius), device=device)
                    total += (patch * cmask.unsqueeze(0)).sum(dim=(-1, -2))
                
                E_in_each_wl_roi[:, t] = total

            # 🔑 ratio = E(s在s的ROI) / E(s在所有ROI)
            denom = E_in_each_wl_roi.sum(dim=1) + 1e-12
            ratios[:, s] = E_in_each_wl_roi[:, s] / denom

        ratio_list.append(ratios.detach().cpu())

    ratio_all = torch.cat(ratio_list, dim=0).numpy()  # (N,L)

    return {
        "ratio_mean": float(ratio_all.mean()),
        "ratio_per_wl": ratio_all.mean(axis=0),  # (L,)
    }

@torch.no_grad()
def evaluate_spot_metrics_multiwl(
    model: D2NNModelMultiWL,
    loader: DataLoader,
    *,
    device: torch.device,
    evaluation_regions: list,
    detect_radius: int,
    wl_idx: int,
    L: int,
    num_modes: int,
) -> dict:
    """
    评估指定波长的重建指标
    """
    model.eval()
    pred_amp_list, true_amp_list = [], []

    for images, label_img, amp in loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)
        amp = amp.to(device, dtype=torch.float32, non_blocking=True)

        if images.ndim == 3:
            images = images.unsqueeze(1)

        amp2 = amp ** 2
        true_energy_frac = amp2 / (amp2.sum(dim=1, keepdim=True) + 1e-12)
        true_amp_frac = torch.sqrt(true_energy_frac + 1e-12)
        true_amp_list.append(true_amp_frac.detach().cpu())

        x = images.repeat(1, L, 1, 1).contiguous()
        I_blhw = model(x)
        I_bhw = I_blhw[:, wl_idx].to(torch.float32)

        # 提取该波长对应的区域（每个模式在该波长的标签位置）
        # 标签索引: mode_k * L + wl_idx
        wl_regions = [evaluation_regions[k * L + wl_idx] for k in range(num_modes)]
        
        pred_energy_frac = region_energy_fractions(
            I_bhw,
            evaluation_regions=wl_regions,
            detect_radius=detect_radius,
        )
        pred_amp_frac = torch.sqrt(pred_energy_frac + 1e-12)
        pred_amp_list.append(pred_amp_frac.detach().cpu())

    pred = torch.cat(pred_amp_list, dim=0).numpy()
    true = torch.cat(true_amp_list, dim=0).numpy()

    diff = pred - true
    abs_diff = np.abs(diff)
    rel = abs_diff / (np.abs(true) + 1e-12)
    cc = np.asarray([_per_sample_corrcoef(pred[i], true[i]) for i in range(pred.shape[0])], dtype=np.float64)

    return {
        "avg_amplitudes_diff": float(abs_diff.mean()),
        "avg_relative_amp_err": float(rel.mean()),
        "cc_recon_amp": cc,
        "amplitudes_diff": diff,
    }

# ============================================================
# Multi-wavelength: SNR / Isolation / Crosstalk
# ============================================================
@torch.no_grad()
def evaluate_snr_isolation_crosstalk_multiwl(
    model: D2NNModelMultiWL,
    loader: DataLoader,
    *,
    device: torch.device,
    evaluation_regions: list,
    detect_radius: int,
    wl_idx: int,
    L: int,
    num_modes: int,
    eps: float = 1e-12,
) -> dict:
    """
    评估 wl_idx 这个波长的 SNR / Isolation / Crosstalk

    新增:
      isolation_db_mean_allroi : 分母 = 全部 M·L - 1 个 ROI 的能量之和
      isolation_db_wc_allroi   : 分母 = 全部 M·L - 1 个 ROI 中能量最大的那个
      crosstalk_matrix_full    : (num_modes, num_modes * L)
                                 行 = 输入 mode (target),
                                 列 = 全部 detector 的能量占比
                                 列序: idx = k * L + w（与 evaluation_regions 一致）
    """
    model.eval()

    snr_ratio_list:        list[float] = []
    iso_db_list:           list[float] = []   # 同 λ
    iso_db_wc_list:        list[float] = []   # 同 λ 最坏
    iso_db_allroi_list:    list[float] = []   # 🆕 全部 ROI
    iso_db_wc_allroi_list: list[float] = []   # 🆕 全部 ROI 最坏

    cm_sum   = np.zeros((num_modes, num_modes),       dtype=np.float64)
    cm_count = np.zeros(num_modes,                    dtype=np.int64)
    # 🆕 全 ROI 串扰矩阵: (num_modes, num_modes * L)
    cm_full_sum   = np.zeros((num_modes, num_modes * L), dtype=np.float64)
    cm_full_count = np.zeros(num_modes,                  dtype=np.int64)

    # 同波长 ROI 在 evaluation_regions 里的索引
    same_wl_indices = [k * L + wl_idx for k in range(num_modes)]
    # 同波长 ROI 在"全 ROI 列表"里对应 target 的位置（用于排除自己）
    # target_t 对应的全 ROI 列索引 = t * L + wl_idx

    for images, label_img, amp in loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)
        amp    = amp.to(device, dtype=torch.float32, non_blocking=True)
        if images.ndim == 3:
            images = images.unsqueeze(1)

        x = images.repeat(1, L, 1, 1).contiguous()
        I_blhw = model(x).to(torch.float32)        # (B, L, H, W)
        I_bhw  = I_blhw[:, wl_idx]                 # (B, H, W) - 该波长输出
        B, H, W = I_bhw.shape

        # ---- 🆕 计算「全部 M·L 个 ROI」的能量 (B, M*L) ----
        E_full = torch.zeros((B, num_modes * L), device=device, dtype=torch.float32)
        for mi, (x0, x1, y0, y1) in enumerate(evaluation_regions):
            patch = I_bhw[:, y0:y1, x0:x1]
            hh, ww = patch.shape[-2], patch.shape[-1]
            cmask = _make_circle_mask(hh, ww, float(detect_radius), device=device)
            E_full[:, mi] = (patch * cmask.unsqueeze(0)).sum(dim=(-1, -2))

        E_full_np = E_full.detach().cpu().numpy().astype(np.float64)  # (B, M*L)
        # 同波长子集 (B, num_modes)
        E_np = E_full_np[:, same_wl_indices]

        # ---- SNR_full = 同波长 ROI 总能量 / 整图能量 ----
        total_full = I_bhw.sum(dim=(-1, -2)).detach().cpu().numpy().astype(np.float64)
        roi_sum    = E_np.sum(axis=1)
        ratio = roi_sum / (total_full + eps)
        snr_ratio_list.extend(ratio.tolist())

        # ---- Isolation + Crosstalk ----
        target_idx = torch.argmax(amp, dim=1).detach().cpu().numpy()
        for b in range(B):
            t = int(target_idx[b])
            # target ROI 在「同波长子集」里就是 t；在「全 ROI」里是 t * L + wl_idx
            t_full = t * L + wl_idx
            Et = E_full_np[b, t_full]   # 与 E_np[b, t] 等价

            # === 原指标: 只算同波长 others (M-1 个) ===
            mask_others = np.ones(num_modes, dtype=bool); mask_others[t] = False
            E_others = E_np[b, mask_others]
            E_sum_others = float(E_others.sum())
            E_max_others = float(E_others.max()) if E_others.size > 0 else 0.0
            iso_db_list.append   (10.0 * np.log10((Et + eps) / (E_sum_others + eps)))
            iso_db_wc_list.append(10.0 * np.log10((Et + eps) / (E_max_others + eps)))

            # === 🆕 全 ROI others (M·L - 1 个) ===
            mask_full = np.ones(num_modes * L, dtype=bool); mask_full[t_full] = False
            E_full_others     = E_full_np[b, mask_full]
            E_full_sum_others = float(E_full_others.sum())
            E_full_max_others = float(E_full_others.max()) if E_full_others.size > 0 else 0.0
            iso_db_allroi_list.append   (10.0 * np.log10((Et + eps) / (E_full_sum_others + eps)))
            iso_db_wc_allroi_list.append(10.0 * np.log10((Et + eps) / (E_full_max_others + eps)))

            # 同波长串扰矩阵 (原有)
            row = E_np[b] / (E_np[b].sum() + eps)
            cm_sum[t]   += row
            cm_count[t] += 1
            # 🆕 全 ROI 串扰矩阵: 行 = 输入 mode, 列 = 全部 ROI
            row_full = E_full_np[b] / (E_full_np[b].sum() + eps)
            cm_full_sum[t]   += row_full
            cm_full_count[t] += 1

        del I_blhw, I_bhw, E_full

    # 平均同波长串扰
    crosstalk_matrix = np.full((num_modes, num_modes), np.nan, dtype=np.float64)
    for k in range(num_modes):
        if cm_count[k] > 0:
            crosstalk_matrix[k] = cm_sum[k] / cm_count[k]

    # 🆕 平均全 ROI 串扰
    crosstalk_matrix_full = np.full((num_modes, num_modes * L), np.nan, dtype=np.float64)
    for k in range(num_modes):
        if cm_full_count[k] > 0:
            crosstalk_matrix_full[k] = cm_full_sum[k] / cm_full_count[k]

    # 标量 SNR
    ratio_mean = float(np.mean(snr_ratio_list)) if snr_ratio_list else float("nan")
    if np.isfinite(ratio_mean):
        rc = float(np.clip(ratio_mean, eps, 1.0 - eps))
        snr_db_full = 10.0 * np.log10(rc / (1.0 - rc))
    else:
        snr_db_full = float("nan")

    return {
        "snr_ratio_full":               ratio_mean,
        "snr_db_full":                  float(snr_db_full),
        # 原有（同波长 isolation）
        "isolation_db_mean":            float(np.nanmean(iso_db_list))         if iso_db_list           else float("nan"),
        "isolation_db_wc":              float(np.nanmean(iso_db_wc_list))      if iso_db_wc_list        else float("nan"),
        "crosstalk_matrix":             crosstalk_matrix,
        # 🆕 跨波长 isolation
        "isolation_db_mean_allroi":     float(np.nanmean(iso_db_allroi_list))  if iso_db_allroi_list    else float("nan"),
        "isolation_db_wc_allroi":       float(np.nanmean(iso_db_wc_allroi_list)) if iso_db_wc_allroi_list else float("nan"),
        "crosstalk_matrix_full":        crosstalk_matrix_full,  # (M, M*L)
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
    """
    保存预测诊断图（简洁版：Label 不再画框/文字标注，与单波长版风格一致）。
    - Label 面板：纯底图，无 overlay
    - Pred 面板：仅一圈 cyan 虚线圆（同单波长版）
    - 中间一行：每个波长下的 mode energy ratio 柱状图
    - 最底部：每个 source λ 的预测能量在不同 ROI 集合的占比
    """
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
        I_pred = model(xin)                                     # (1,L,H,W)

        I_in = (torch.abs(x[0, 0]) ** 2).detach().cpu().numpy()

        amp2 = (amp[0] ** 2).detach().cpu().numpy()
        true_energy_frac = _safe_norm_np(amp2)

        # 为每个波长生成独立的标签图
        labels_per_wl = []
        for wl_idx in range(Lloc):
            label_indices = [k * Lloc + wl_idx for k in range(num_modes)]
            wl_label_patterns = MMF_Label_data[:, :, label_indices].numpy()  # (H, W, M)
            energy = true_energy_frac.reshape(1, -1)
            label_wl = np.einsum("nm,hwm->hw", energy, wl_label_patterns)
            labels_per_wl.append(label_wl)

        # 创建图形
        fig = plt.figure(figsize=(4 * (1 + 2 * Lloc), 10))
        gs = fig.add_gridspec(
            3, 1 + 2 * Lloc,
            height_ratios=[1.0, 1.0, 0.55],
            hspace=0.35, wspace=0.25,
        )

        # 第一列：输入
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(I_in, cmap="inferno")
        ax0.set_title("Input |E|^2", fontsize=10, fontweight="bold")
        ax0.axis("off")

        for li in range(Lloc):
            wl_regions = [evaluation_regions[k * Lloc + li] for k in range(num_modes)]

            # ---- Label 面板（极简：不画任何 overlay，跟单波长版一致）----
            ax_label = fig.add_subplot(gs[0, 1 + 2 * li])
            ax_label.imshow(labels_per_wl[li], cmap="inferno")
            ax_label.set_title(
                f"Label λ={wavelengths[li] * 1e9:.0f}nm",
                fontsize=10, fontweight="bold",
            )
            ax_label.axis("off")
            # ❌ 这里不再画 Rectangle / Circle / 文字标注

            # ---- Pred 面板（只画 cyan 虚线圆，与单波长版一致）----
            axI = fig.add_subplot(gs[0, 2 + 2 * li])
            I_li = I_pred[0, li].detach().cpu().numpy()
            axI.imshow(I_li, cmap="inferno")
            axI.set_title(
                f"Pred λ={wavelengths[li] * 1e9:.0f}nm",
                fontsize=10, fontweight="bold",
            )
            axI.axis("off")

            for (x0, x1, y0, y1) in wl_regions:
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                axI.add_patch(
                    Circle(
                        (cx, cy),
                        radius=detect_radius,
                        linewidth=0.8,
                        edgecolor="cyan",
                        facecolor="none",
                        linestyle=":",
                        alpha=0.9,
                    )
                )

            # ---- 中间一行：能量柱状图（不变）----
            I_bhw = I_pred[:, li].to(torch.float32)
            pred_energy_frac = region_energy_fractions(
                I_bhw, wl_regions, detect_radius=detect_radius
            )[0].detach().cpu().numpy()

            axb = fig.add_subplot(gs[1, 1 + 2 * li : 3 + 2 * li])
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

        # 左下角空白
        fig.add_subplot(gs[1, 0]).axis("off")

        # ---- 最底部：source λ 的能量在不同 ROI 集合的占比（不变）----
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
            ax_wl.bar(
                x_axis + offset, R[:, t],
                width=bar_w * 0.95, alpha=0.85,
                label=f"ROI@{wavelengths[t]*1e9:.0f}nm",
            )

        ax_wl.set_ylim(0.0, 1.0)
        ax_wl.set_ylabel("Pred Energy Ratio\n(over ROI wavelength sets)")
        ax_wl.set_xticks(x_axis)
        ax_wl.set_xticklabels([f"{w * 1e9:.0f}" for w in wavelengths])
        ax_wl.set_xlabel("Source wavelength (nm)")
        ax_wl.grid(True, axis="y", alpha=0.25)
        ax_wl.set_title(
            "Predicted energy distribution of each source λ across wavelength-ROI sets",
            fontsize=11,
        )
        ax_wl.legend(ncol=min(Lloc, 6), fontsize=9, loc="upper right")

        fig.suptitle(
            f"MultiWL Prediction Analysis - Sample {si}",
            fontsize=14, fontweight="bold", y=0.98,
        )
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
eigenmodes_OM4 = load_complex_modes_from_mat("mmf_10modes_GRIN_176_PD1.2.mat", key="modes_field")
print("Loaded modes shape:", eigenmodes_OM4.shape, "dtype:", eigenmodes_OM4.dtype)

mode_context = build_mode_context(eigenmodes_OM4, num_modes)
MMF_data = mode_context["mmf_data_np"]
MMF_data_ts = mode_context["mmf_data_ts"]
base_amplitudes = mode_context["base_amplitudes"]
base_phases = mode_context["base_phases"]


# ============================================================
# 生成多波长标签模板
# ============================================================
print(f"\n{'='*60}")
print(f"Generating MultiWL Labels: {num_modes} modes × {L} wavelengths = {num_modes * L} labels")
print(f"{'='*60}")

mmf_label_patterns, evaluation_regions = generate_detector_patterns_multiwl(
    H=layer_size,
    W=layer_size,
    num_modes=num_modes,
    num_wavelengths=L,
    radius=circle_focus_radius,
    pattern_mode=label_pattern_mode,
    show_debug=show_detection_overlap_debug,
    margin_ratio=0.2,  # 10% margin to avoid edge effects
)

MMF_Label_data = torch.from_numpy(mmf_label_patterns).to(torch.float32)  # (H, W, M*L)

print(f"✔ Generated {len(evaluation_regions)} evaluation regions")

# Overlap debug
if show_detection_overlap_debug:
    detection_debug_dir = RUN_ROOT / "detection_region_debug"
    detection_debug_dir.mkdir(parents=True, exist_ok=True)
    overlap_map = np.zeros((layer_size, layer_size), dtype=np.float32)
    for (x0, x1, y0, y1) in evaluation_regions:
        overlap_map[y0:y1, x0:x1] += 1.0
    overlap_pixels = int(np.count_nonzero(overlap_map > 1.0 + 1e-6))
    max_overlap = float(overlap_map.max()) if overlap_map.size else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(np.zeros((layer_size, layer_size), dtype=np.float32), cmap="Greys")
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
                    color='white', fontsize=7, bbox=dict(boxstyle='round', 
                    facecolor=color, alpha=0.7))

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
    """
    为每个波长构建独立的本征模式数据集
    """
    datasets_per_wl = []
    
    if phase_option == 4:
        num_samples = num_modes
        amplitudes = base_amplitudes[:num_samples]
        phases = base_phases[:num_samples]
    else:
        amplitudes = base_amplitudes
        phases = base_phases
        num_samples = amplitudes.shape[0]

    # 生成输入场
    complex_weights = amplitudes * np.exp(1j * phases)
    complex_weights_ts = torch.from_numpy(complex_weights.astype(np.complex64))
    image_data = generate_fields_ts(
        complex_weights_ts, MMF_data_ts, num_samples, num_modes, field_size
    ).to(torch.complex64)

    dummy_label = torch.zeros([1, layer_size, layer_size], dtype=torch.float32)
    images_prepared = []
    for i in range(num_samples):
        img_i, _ = prepare_sample(image_data[i], dummy_label, layer_size)
        images_prepared.append(img_i)
    image_tensor = torch.stack(images_prepared, dim=0)

    # 为每个波长只提取对应的标签通道
    for wl_idx in range(L):
        label_indices = [k * L + wl_idx for k in range(num_modes)]
        wl_label_patterns = MMF_Label_data[:, :, label_indices]  # (H, W, M)
        
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
    """
    为每个波长构建独立的叠加态数据集
    """
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

    dummy_label = torch.zeros([1, layer_size, layer_size], dtype=torch.float32)
    images_prepared = []
    for i in range(num_samples):
        img_i, _ = prepare_sample(image_data[i], dummy_label, layer_size)
        images_prepared.append(img_i)
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
# Train/Eval loop（修正版）
# ============================================================
all_losses: list[list[float]] = []
metrics_by_wl: dict[int, list[dict]] = {int(li): [] for li in range(L)}
snr_db_per_layer:     dict[int, np.ndarray] = {}   # key=num_layer -> (L,)
snr_db_ring_per_layer:   dict[int, np.ndarray] = {}   # ring-bg union (L,) 
snr_each_ring_per_layer: dict[int, np.ndarray] = {}   # ring-bg each  (L, M)  
iso_mean_per_layer:   dict[int, np.ndarray] = {}
iso_wc_per_layer:     dict[int, np.ndarray] = {}
crosstalk_per_layer:  dict[int, np.ndarray] = {}   # key=num_layer -> (L, M, M)

target_ratio_per_layer: dict[int, np.ndarray] = {}  # key=num_layers -> (L,)

detect_radius_eval = int(detectsize // 2)

for num_layer in num_layer_option:
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'='*70}\nTraining D2NNModelMultiWL with {num_layer} layers\n{'='*70}")

    # 构建数据集
    if training_dataset_mode == "eigenmode":
        train_datasets_per_wl, train_meta = build_eigenmode_dataset_multiwl()
    elif training_dataset_mode == "superposition":
        train_datasets_per_wl, train_meta = build_superposition_dataset_multiwl(
            num_superposition_train_samples, superposition_train_seed
        )
    else:
        raise ValueError("Unknown training_dataset_mode")
    mode_triptych_records: list[dict[str]] = []
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
    ).to(device)

    # 训练
    losses: list[float] = []
    epoch_durations: list[float] = []
    t0 = time.time()

    g = torch.Generator()
    g.manual_seed(SEED)

    # 🔑 使用分阶段训练
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
        stage_ratios=[0.25, 0.25, 0.25, 0.25],  # 可自定义
        verbose=True,
    )
    
    # 提取结果
    losses = training_result['losses']
    epoch_durations = training_result['epoch_durations']
    stage_info = training_result['stage_info']
    total_time = training_result['total_time']
    all_losses.append(losses)
    # 打印摘要
    print_stage_summary(stage_info)
    
    # 在 save_staged_training_info 之后添加
    training_output_dir = RUN_ROOT / "training_analysis"

    # 🔑 保存训练曲线
    train_logs = save_training_curves(
        losses=losses,
        epoch_durations=epoch_durations,
        out_dir=training_output_dir,
        tag=f"staged_multiwl_m{num_modes}_L{L}_ls{layer_size}_nlayer{num_layer}_{run_tag}",
        num_layers=num_layer,
    )
    print(f"✔ Training curves saved -> {train_logs['loss_plot']}")

    # 保存阶段信息
    save_staged_training_info(
        stage_info, 
        training_output_dir / f"stage_info_layers{num_layer}_{run_tag}.txt"  # 🔑 添加 run_tag
    )

    # 保存checkpoint
    ckpt_dir = RUN_ROOT / "checkpoints"
    ckpt_path = ckpt_dir / f"multiwl_{num_layer}layers_m{num_modes}_L{L}.pth"
    save_checkpoint_multiwl(
        model,
        ckpt_path,
        meta={
            "num_layers": int(num_layer),
            "layer_size": int(layer_size),
            "num_modes": int(num_modes),
            "num_wavelengths": int(L),
            "wavelengths": wavelengths.astype(np.float32),
            "total_training_time_sec": float(total_time),
        },
    )
    print("✔ Checkpoint saved ->", ckpt_path)

    # 保存相位掩模
    phase_masks = extract_phase_masks_multiwl(model)
    if phase_masks:
        pm_dir = RUN_ROOT / "phase_masks" / f"L{num_layer}_{run_tag}"
        pm_dir.mkdir(parents=True, exist_ok=True)
        pm_mat = pm_dir / "phase_masks.mat"
        savemat(str(pm_mat), {"phase_masks": np.stack(phase_masks, axis=0).astype(np.float32)})
        print(f"✔ Phase masks saved -> {pm_mat}")
    # 🆕 生成 PNG 可视化（每层单独一张图）
    png_paths = visualize_phase_masks(
        phase_masks,
        out_dir=pm_dir,
        base_name=f"phase_mask_L{num_layer}",
        save_degree=False,  # 保存弧度（或改为 True 显示角度）
        dpi=300,
        cmap="twilight",  # 适合相位的色图
        show_stats=True,
    )
    print(f"✔ Generated {len(png_paths)} phase mask PNGs -> {pm_dir}")
    # 预测可视化
    diag_dir = RUN_ROOT / "prediction_viz" / f"L{num_layer}_{run_tag}"
    n_vis = min(num_pred_diag_samples, len(test_datasets_per_wl[0]))
    diag_paths = save_prediction_diagnostics_multiwl(
        model,
        test_datasets_per_wl[0],  # 使用第一个波长的测试集
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

        # ---- 评估每个波长（amp_err / cc_amp + SNR / iso / crosstalk）----
        snr_db_arr   = np.full(L, np.nan, dtype=np.float64)
        iso_mean_arr = np.full(L, np.nan, dtype=np.float64)
        iso_wc_arr   = np.full(L, np.nan, dtype=np.float64)
        cm_stack     = np.full((L, num_modes, num_modes), np.nan, dtype=np.float64)
        snr_db_ring_arr   = np.full(L, np.nan, dtype=np.float64)               # NEW
        snr_each_ring_arr = np.full((L, num_modes), np.nan, dtype=np.float64)  # NEW
        # ---- 在 num_layer 循环外，初始化容器（新增两个）----
        iso_mean_allroi_per_layer: dict[int, np.ndarray] = {}
        iso_wc_allroi_per_layer:   dict[int, np.ndarray] = {}
        crosstalk_full_per_layer:  dict[int, np.ndarray] = {}   # (L, M, M*L)

        # ---- 在波长循环里，新增 array ----
        iso_mean_allroi_arr = np.full(L, np.nan, dtype=np.float64)
        iso_wc_allroi_arr   = np.full(L, np.nan, dtype=np.float64)
        ct_full_stack       = np.full((L, num_modes, num_modes * L), np.nan, dtype=np.float64)
        for li in range(L):
            test_loader_wl = DataLoader(test_datasets_per_wl[li], batch_size=batch_size, shuffle=False)

            # —— 已有：amp_err / rel / cc ——
            metrics = evaluate_spot_metrics_multiwl(
                model,
                test_loader_wl,
                device=device,
                evaluation_regions=evaluation_regions,
                detect_radius=detect_radius_eval,
                wl_idx=li,
                L=L,
                num_modes=num_modes,
            )
            cc_mean = float(np.nanmean(metrics["cc_recon_amp"]))
            cc_std  = float(np.nanstd(metrics["cc_recon_amp"]))

            # —— 新增：SNR / isolation / crosstalk ——
            m_extra = evaluate_snr_isolation_crosstalk_multiwl(
                model,
                test_loader_wl,
                device=device,
                evaluation_regions=evaluation_regions,
                detect_radius=detect_radius_eval,
                wl_idx=li,
                L=L,
                num_modes=num_modes,
            )
            snr_db_arr[li]   = m_extra["snr_db_full"]
            iso_mean_arr[li] = m_extra["isolation_db_mean"]
            iso_wc_arr[li]   = m_extra["isolation_db_wc"]
            cm_stack[li]     = m_extra["crosstalk_matrix"]
            snr_db_ring_arr[li]    = m_extra["snr_db_ring_union"]
            snr_each_ring_arr[li]  = m_extra["snr_db_ring_each"]
            iso_mean_allroi_per_layer[int(num_layer)] = iso_mean_allroi_arr
            iso_wc_allroi_per_layer[int(num_layer)]   = iso_wc_allroi_arr
            crosstalk_full_per_layer[int(num_layer)]  = ct_full_stack

            print(
                f"[Metrics | {num_layer} layers | λ_idx={li} | λ={wavelengths[li]*1e9:.1f} nm] "
                f"amp_err={metrics['avg_amplitudes_diff']:.4f}, "
                f"rel={metrics['avg_relative_amp_err']:.4f}, "
                f"cc={cc_mean:.4f}±{cc_std:.4f}, "
                f"SNR_cont={snr_db_arr[li]:.2f}dB, "
                f"SNR_ring={snr_db_ring_arr[li]:.2f}dB, "
                f"iso(mean/wc)={iso_mean_arr[li]:.2f}/{iso_wc_arr[li]:.2f}dB"
            )

            metrics_by_wl[int(li)].append({
                "num_layers": int(num_layer),
                **metrics,
                "snr_db_full":       float(snr_db_arr[li]),
                "snr_db_ring_union": float(snr_db_ring_arr[li]),
                "snr_db_ring_each":  np.asarray(snr_each_ring_arr[li], dtype=np.float64).copy(),
                "isolation_db_mean": float(iso_mean_arr[li]),
                "isolation_db_wc":   float(iso_wc_arr[li]),
            })

        snr_db_per_layer[int(num_layer)]    = snr_db_arr
        iso_mean_per_layer[int(num_layer)]  = iso_mean_arr
        iso_wc_per_layer[int(num_layer)]    = iso_wc_arr
        crosstalk_per_layer[int(num_layer)] = cm_stack
        snr_db_ring_per_layer[int(num_layer)]   = snr_db_ring_arr     # NEW
        snr_each_ring_per_layer[int(num_layer)] = snr_each_ring_arr   # NEW



        # ===== 新增：目标波长ROI能量 / 所有波长ROI能量（一次性算全波长）=====
    test_loader_any = DataLoader(test_datasets_per_wl[0], batch_size=batch_size, shuffle=False)
    wl_ratio = evaluate_target_wl_over_all_wl_roi_ratio(
        model,
        test_loader_any,
        device=device,
        evaluation_regions=evaluation_regions,
        detect_radius=detect_radius_eval,
        L=L,
        num_modes=num_modes,
    )
    target_ratio_per_layer[int(num_layer)] = wl_ratio["ratio_per_wl"]

    print(
        f"[TargetWL/AllWL ROI | {num_layer} layers] "
        f"mean={wl_ratio['ratio_mean']:.6f}, per_wl={wl_ratio['ratio_per_wl']}"
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n" + "="*70)
print("All training completed!")
print("="*70)


# ============================================================
# 保存指标分析: 每个 metric 一张图，每张图里 L 条曲线（一个波长一条）
# ============================================================
metrics_dir = RUN_ROOT / "metrics_analysis"
metrics_dir.mkdir(parents=True, exist_ok=True)
tag = datetime.now().strftime("%Y%m%d_%H%M%S")

# ---- 收集成 (NL, L) 矩阵 ----
sorted_layers = sorted(snr_db_per_layer.keys())
layer_counts  = np.asarray(sorted_layers, dtype=np.int32)
NL = len(layer_counts)

def _collect_per_wl(field_extractor) -> np.ndarray:
    """从 metrics_by_wl 收集每个波长的 (NL,) 数组，组装成 (NL, L)。"""
    M = np.full((NL, L), np.nan, dtype=np.float64)
    for li in range(L):
        mlist = metrics_by_wl.get(int(li), [])
        # 按 num_layers 排序对齐 layer_counts
        mlist_sorted = sorted(mlist, key=lambda d: d["num_layers"])
        for i, m in enumerate(mlist_sorted):
            if i < NL:
                M[i, li] = field_extractor(m)
    return M

M_amp_err   = _collect_per_wl(lambda m: float(m["avg_amplitudes_diff"]))
M_rel_err   = _collect_per_wl(lambda m: float(m["avg_relative_amp_err"]))
M_cc_mean   = _collect_per_wl(lambda m: float(np.nanmean(m["cc_recon_amp"])))
M_cc_std    = _collect_per_wl(lambda m: float(np.nanstd (m["cc_recon_amp"])))
M_snr_db    = np.vstack([snr_db_per_layer[nl]   for nl in sorted_layers])  # (NL, L)
M_snr_db_ring   = np.vstack([snr_db_ring_per_layer[nl]   for nl in sorted_layers])  # (NL, L) ring-bg union
M_snr_each_ring = np.stack ([snr_each_ring_per_layer[nl] for nl in sorted_layers])  # (NL, L, M)
M_iso_mean  = np.vstack([iso_mean_per_layer[nl] for nl in sorted_layers])
M_iso_wc    = np.vstack([iso_wc_per_layer[nl]   for nl in sorted_layers])
M_target_wl = np.vstack([target_ratio_per_layer[nl] for nl in sorted_layers])  # 已有
CT_4d       = np.stack([crosstalk_per_layer[nl] for nl in sorted_layers], axis=0)  # (NL, L, M, M)

wl_nm     = (wavelengths * 1e9).astype(np.float64)
wl_labels = [f"{w:.0f} nm" for w in wl_nm]
cmap_wl   = plt.cm.viridis(np.linspace(0.15, 0.85, L))

def _save_single_metric(fname_stem: str, ylabel: str, title: str, plot_fn) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    plot_fn(ax)
    ax.set_xlabel("Number of layers")
    ax.set_ylabel(ylabel)
    ax.set_xticks(layer_counts)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out = metrics_dir / f"{fname_stem}_{tag}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ {ylabel:<28s} -> {out}")
    return out

# 1) avg_amp_error
_save_single_metric(
    "metric_avg_amp_error", "avg_amp_error",
    "Average amplitude error vs. layers",
    lambda ax: [ax.plot(layer_counts, M_amp_err[:, li], marker="o",
                        color=cmap_wl[li], label=wl_labels[li]) for li in range(L)],
)

# 2) avg_relative_amp_error
_save_single_metric(
    "metric_avg_relative_amp_error", "avg_relative_amp_error",
    "Average relative amplitude error vs. layers",
    lambda ax: [ax.plot(layer_counts, M_rel_err[:, li], marker="o",
                        color=cmap_wl[li], label=wl_labels[li]) for li in range(L)],
)

# 3) cc_amp mean ± std
_save_single_metric(
    "metric_cc_amp", "cc_amp mean ± std",
    "Reconstruction amplitude correlation vs. layers",
    lambda ax: [ax.errorbar(layer_counts, M_cc_mean[:, li], yerr=M_cc_std[:, li],
                            marker="o", capsize=3, color=cmap_wl[li],
                            label=wl_labels[li]) for li in range(L)],
)

# 4) SNR (dB)
_save_single_metric(
    "metric_snr_full_db", "SNR_full (dB)",
    "Signal containment (dB) vs. layers",
    lambda ax: [ax.plot(layer_counts, M_snr_db[:, li], marker="o",
                        color=cmap_wl[li], label=wl_labels[li]) for li in range(L)],
)

# 5) Isolation mean (dB)
_save_single_metric(
    "metric_isolation_db_mean", "Isolation mean (dB)",
    "Mode isolation (mean) vs. layers",
    lambda ax: [ax.plot(layer_counts, M_iso_mean[:, li], marker="o",
                        color=cmap_wl[li], label=wl_labels[li]) for li in range(L)],
)

# 6) Isolation worst-case (dB)
_save_single_metric(
    "metric_isolation_db_worst", "Isolation worst-case (dB)",
    "Mode isolation (worst-case) vs. layers",
    lambda ax: [ax.plot(layer_counts, M_iso_wc[:, li], marker="s", linestyle="--",
                        color=cmap_wl[li], label=wl_labels[li]) for li in range(L)],
)

# 7) target_wl_over_all_wl_roi （MultiWL 专属，保留之前的）
_save_single_metric(
    "metric_target_wl_ratio", "TargetWL / AllWL (ROI)",
    "Wavelength-demux ratio vs. layers",
    lambda ax: [ax.plot(layer_counts, M_target_wl[:, li], marker="o",
                        color=cmap_wl[li], label=wl_labels[li]) for li in range(L)],
)

# ---- 一份汇总 .mat ----
metrics_mat_path = metrics_dir / f"metrics_vs_layers_{tag}.mat"
savemat(
    str(metrics_dir / f"metrics_vs_layers_{tag}.mat"),
    {
        "layers":          layer_counts.astype(np.float64),
        "wavelengths_m":   wavelengths.astype(np.float64),
        "wavelengths_nm":  (wavelengths * 1e9).astype(np.float64),

        "avg_amp_error":          M_amp_err,
        "avg_relative_amp_error": M_rel_err,
        "cc_amp_mean":            M_cc_mean,
        "cc_amp_std":             M_cc_std,

        # ===== SNR：两套都保存 =====
        "snr_db_full":        M_snr_db,         # (NL, L) containment dB（旧公式）
        "snr_db_ring_union":  M_snr_db_ring,    # (NL, L) ring-bg union dB（新）       ← NEW
        "snr_db_ring_each":   M_snr_each_ring,  # (NL, L, M) per-ROI ring-bg dB        ← NEW

        "isolation_db_mean":   M_iso_mean,
        "isolation_db_worst":  M_iso_wc,
    },
)
print(f"  ✔ Metrics .mat -> {metrics_mat_path}")

# ---- Crosstalk 热力图: 每 (num_layer, wavelength) 一张 (linear + dB) ----
ct_dir = metrics_dir / "crosstalk_heatmaps"
ct_dir.mkdir(parents=True, exist_ok=True)
for li_idx, n_layer in enumerate(layer_counts):
    for wl_idx in range(L):
        mat = CT_4d[li_idx, wl_idx]
        mat_db = 10.0 * np.log10(np.clip(mat, 1e-6, None))

        # linear
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"Crosstalk (linear) — {n_layer} layers, λ={wl_nm[wl_idx]:.0f}nm")
        ax.set_xlabel("Detector index"); ax.set_ylabel("Input mode index")
        ax.set_xticks(range(num_modes)); ax.set_yticks(range(num_modes))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="energy fraction")
        for r in range(num_modes):
            for c in range(num_modes):
                v = mat[r, c]
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.2f}", ha="center", va="center",
                            color=("white" if v < 0.5 else "black"), fontsize=8)
        fig.tight_layout()
        fig.savefig(ct_dir / f"crosstalk_linear_L{n_layer}_wl{wl_idx:02d}_{tag}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

        # dB
        fig, ax = plt.subplots(figsize=(5.5, 5))
        im = ax.imshow(mat_db, cmap="magma", vmin=-30, vmax=0)
        ax.set_title(f"Crosstalk (dB) — {n_layer} layers, λ={wl_nm[wl_idx]:.0f}nm")
        ax.set_xlabel("Detector index"); ax.set_ylabel("Input mode index")
        ax.set_xticks(range(num_modes)); ax.set_yticks(range(num_modes))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="dB")
        for r in range(num_modes):
            for c in range(num_modes):
                v = mat_db[r, c]
                if np.isfinite(v):
                    ax.text(c, r, f"{v:.0f}", ha="center", va="center",
                            color=("white" if v < -15 else "black"), fontsize=8)
        fig.tight_layout()
        fig.savefig(ct_dir / f"crosstalk_db_L{n_layer}_wl{wl_idx:02d}_{tag}.png",
                    dpi=300, bbox_inches="tight")
        plt.close(fig)

print(f"  ✔ Crosstalk heatmaps -> {ct_dir}")
print("\n✅ All outputs saved successfully!")

