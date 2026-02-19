#%%
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
import torch.optim as optim
import torch.nn.functional as F
from scipy.io import savemat
from torch.optim.lr_scheduler import ExponentialLR
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

# MultiWL model
from odnn_multiwl_model import D2NNModelMultiWL

# superposition sampler
from odnn_training_eval import build_superposition_eval_context


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

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Using Device:", device)


# ============================================================
# Parameters
# ============================================================
field_size = 25
layer_size = 110
num_modes = 3

circle_focus_radius = 5
circle_detectsize = 10
focus_radius = circle_focus_radius
detectsize = circle_detectsize

batch_size = 16

evaluation_mode = "superposition"      # "eigenmode" or "superposition"
training_dataset_mode = "eigenmode"    # "eigenmode" or "superposition"

num_superposition_eval_samples = 1000
num_superposition_train_samples = 100
superposition_eval_seed = 20240116
superposition_train_seed = 20240115

num_layer_option = [2, 3, 4, 5, 6]

# geometry / propagation params
z_layers = 40e-6
pixel_size = 1e-6
z_prop = 120e-6
z_input_to_first = 40e-6

# wavelengths (MultiWL)
# wavelengths = np.array([650e-9, 1568e-9], dtype=np.float32)
wavelengths = np.array([1550e-9], dtype=np.float32)
# wavelengths = np.array([1550e-9, 1568e-9, 1650e-9], dtype=np.float32)
base_wavelength_idx = 0
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
RUN_ROOT = Path(f"results/1550nm_base_{base_wavelength_idx}")
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
    show_debug: bool = False
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """
    生成多波长标签图案
    
    Returns:
        patterns: (H, W, num_modes * num_wavelengths)
        evaluation_regions: [(x0, x1, y0, y1), ...]
    """
    total_labels = num_modes * num_wavelengths
    
    # 计算布局
    num_rows = int(np.floor(np.sqrt(total_labels)))
    num_cols = int(np.ceil(total_labels / num_rows))
    
    # 计算中心坐标
    centers, row_spacing, col_spacing = compute_label_centers(H, W, total_labels, radius)
    
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
        plt.title(f"MultiWL Labels: {num_modes} modes × {num_wavelengths} wavelengths")
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


@torch.no_grad()
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
    保存预测诊断图（为每个波长显示独立的标签）
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
        I_pred = model(xin)

        I_in = (torch.abs(x[0, 0]) ** 2).detach().cpu().numpy()

        amp2 = (amp[0] ** 2).detach().cpu().numpy()
        true_energy_frac = _safe_norm_np(amp2)

        # 🔧 关键修改：为每个波长生成独立的标签图
        labels_per_wl = []
        for wl_idx in range(Lloc):
            label_indices = [k * Lloc + wl_idx for k in range(num_modes)]
            wl_label_patterns = MMF_Label_data[:, :, label_indices].numpy()  # (H, W, M)
            
            # 使用真实能量分布生成该波长的标签
            energy = true_energy_frac.reshape(1, -1)  # (1, M)
            label_wl = np.einsum('nm,hwm->hw', energy, wl_label_patterns)  # (H, W)
            labels_per_wl.append(label_wl)

        # 创建图形：Input + L个Label + L个Pred + L个柱状图
        fig = plt.figure(figsize=(4 * (1 + 2*Lloc), 8))
        gs = fig.add_gridspec(2, 1 + 2*Lloc, height_ratios=[1.0, 1.0])

        # 第一列：输入
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(I_in, cmap="inferno")
        ax0.set_title("Input |E|^2", fontsize=10, fontweight='bold')
        ax0.axis("off")

        # 为每个波长显示：Label + Pred
        for li in range(Lloc):
            # Label 列
            ax_label = fig.add_subplot(gs[0, 1 + 2*li])
            ax_label.imshow(labels_per_wl[li], cmap="inferno")
            ax_label.set_title(f"Label λ={wavelengths[li]*1e9:.0f}nm", 
                              fontsize=10, fontweight='bold')
            ax_label.axis("off")

            # 绘制该波长对应的标签区域
            wl_regions = [evaluation_regions[k * Lloc + li] for k in range(num_modes)]
            for region_idx, (x0, x1, y0, y1) in enumerate(wl_regions):
                color = plt.cm.tab10(li % 10)
                rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.2, 
                               edgecolor=color, facecolor="none", alpha=0.9)
                ax_label.add_patch(rect)
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                ax_label.add_patch(Circle((cx, cy), radius=detect_radius, linewidth=1.0, 
                                    edgecolor=color, linestyle="--", fill=False, alpha=0.9))
                # 标注模式索引
                ax_label.text(cx, cy, f"M{region_idx}", ha='center', va='center', 
                            color='white', fontsize=8, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))

            # Pred 列
            axI = fig.add_subplot(gs[0, 2 + 2*li])
            I_li = I_pred[0, li].detach().cpu().numpy()
            axI.imshow(I_li, cmap="inferno")
            axI.set_title(f"Pred λ={wavelengths[li]*1e9:.0f}nm", 
                         fontsize=10, fontweight='bold')
            axI.axis("off")

            # 绘制预测图上的区域框
            for region_idx, (x0, x1, y0, y1) in enumerate(wl_regions):
                color = plt.cm.tab10(li % 10)
                rect = Rectangle((x0, y0), x1 - x0, y1 - y0, linewidth=1.2, 
                               edgecolor=color, facecolor="none", alpha=0.9)
                axI.add_patch(rect)
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                axI.add_patch(Circle((cx, cy), radius=detect_radius, linewidth=1.0, 
                                    edgecolor=color, linestyle="--", fill=False, alpha=0.9))

            # 能量柱状图
            I_bhw = I_pred[:, li].to(torch.float32)
            pred_energy_frac = region_energy_fractions(
                I_bhw, wl_regions, detect_radius=detect_radius
            )[0].detach().cpu().numpy()

            axb = fig.add_subplot(gs[1, 1 + 2*li:3 + 2*li])  # 跨两列
            idx = np.arange(num_modes)
            width = 0.35
            axb.bar(idx - width/2, true_energy_frac, width, label="True", alpha=0.8)
            axb.bar(idx + width/2, pred_energy_frac, width, label="Pred", alpha=0.8)
            axb.set_ylim(0, 1.0)
            axb.set_xticks(idx)
            axb.set_xticklabels([f"M{i}" for i in idx])
            axb.grid(True, alpha=0.3, axis='y')
            axb.set_title(f"Energy Ratio (λ={wavelengths[li]*1e9:.0f}nm)", fontsize=10)
            axb.set_ylabel("Energy Fraction")
            if li == 0:
                axb.legend(loc='upper right')

        # 左下角空白
        fig.add_subplot(gs[1, 0]).axis("off")

        fig.suptitle(f"MultiWL Prediction Analysis - Sample {si}", 
                    fontsize=14, fontweight='bold', y=0.98)
        fig.tight_layout(rect=[0, 0.0, 1, 0.96])
        
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
eigenmodes_OM4 = load_complex_modes_from_mat("mmf_103modes_25_PD_1.15.mat", key="modes_field")
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

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=0.99)

    # 训练
    losses: list[float] = []
    epoch_durations: list[float] = []
    t0 = time.time()

    g = torch.Generator()
    g.manual_seed(SEED)

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        # 🔧 关键修改：遍历每个波长
        for wl_idx in range(L):
            train_loader_wl = DataLoader(
                train_datasets_per_wl[wl_idx], 
                batch_size=batch_size, 
                shuffle=True, 
                generator=g
            )
            
            for images, label_img, amp in train_loader_wl:
                images = images.to(device, dtype=torch.complex64, non_blocking=True)
                label_img = label_img.to(device, dtype=torch.float32, non_blocking=True)

                if images.ndim == 3:
                    images = images.unsqueeze(1)

                x = images.repeat(1, L, 1, 1).contiguous()

                optimizer.zero_grad(set_to_none=True)
                I_blhw = model(x)
                
                # 只计算当前波长的损失
                loss = F.mse_loss(I_blhw[:, wl_idx], label_img[:, 0])
                
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.item())
                batch_count += 1

        scheduler.step()
        avg_loss = epoch_loss / max(1, batch_count)
        losses.append(avg_loss)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_durations.append(time.time() - epoch_t0)

        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] loss={avg_loss:.10f}")

    total_time = time.time() - t0
    all_losses.append(losses)
    print(f"Training done: {num_layer} layers, time={total_time:.2f}s")

    # 保存训练曲线
    training_output_dir = RUN_ROOT / "training_analysis"
    train_logs = save_training_curves(
        losses=losses,
        epoch_durations=epoch_durations,
        out_dir=training_output_dir,
        tag=f"multiwl_m{num_modes}_L{L}_ls{layer_size}_nlayer{num_layer}_{run_tag}",
        num_layers=num_layer,
    )
    print(f"✔ Training curves saved -> {train_logs['loss_plot']}")

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

    # 评估每个波长
    for li in range(L):
        test_loader_wl = DataLoader(test_datasets_per_wl[li], batch_size=batch_size, shuffle=False)
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
        cc_std = float(np.nanstd(metrics["cc_recon_amp"]))

        print(
            f"[Metrics | {num_layer} layers | λ_idx={li} | λ={wavelengths[li]*1e9:.1f} nm] "
            f"amp_err={metrics['avg_amplitudes_diff']:.6f}, "
            f"rel_err={metrics['avg_relative_amp_err']:.6f}, "
            f"cc={cc_mean:.6f}±{cc_std:.6f}"
        )

        metrics_by_wl[int(li)].append({"num_layers": int(num_layer), **metrics})

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n" + "="*70)
print("All training completed!")
print("="*70)


# ============================================================
# 保存指标分析
# ============================================================
metrics_dir = RUN_ROOT / "metrics_analysis"
metrics_dir.mkdir(parents=True, exist_ok=True)
tag = datetime.now().strftime("%Y%m%d_%H%M%S")

for li in range(L):
    mlist = metrics_by_wl.get(int(li), [])
    if not mlist:
        continue

    layer_counts = np.asarray([m["num_layers"] for m in mlist], dtype=np.int32)
    amp_err = np.asarray([m["avg_amplitudes_diff"] for m in mlist], dtype=np.float64)
    amp_err_rel = np.asarray([m["avg_relative_amp_err"] for m in mlist], dtype=np.float64)
    cc_amp_mean = np.asarray([float(np.nanmean(m["cc_recon_amp"])) for m in mlist], dtype=np.float64)
    cc_amp_std = np.asarray([float(np.nanstd(m["cc_recon_amp"])) for m in mlist], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(7, 9), sharex=True)

    axes[0].plot(layer_counts, amp_err, marker="o")
    axes[0].set_ylabel("Avg Amplitude Error")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(layer_counts, amp_err_rel, marker="o", color="tab:orange")
    axes[1].set_ylabel("Avg Relative Error")
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(layer_counts, cc_amp_mean, yerr=cc_amp_std, marker="o", capsize=4, color="tab:green")
    axes[2].set_ylabel("Correlation Coef")
    axes[2].set_xlabel("Number of Layers")
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(0.0, 1.01)

    fig.suptitle(f"Metrics vs Layers | λ={wavelengths[li]*1e9:.1f} nm")
    fig.tight_layout(rect=[0, 0.0, 1, 0.96])

    fig_path = metrics_dir / f"metrics_wl{li}_{tag}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✔ Metrics plot saved -> {fig_path}")

    mat_path = metrics_dir / f"metrics_wl{li}_{tag}.mat"
    savemat(
        str(mat_path),
        {
            "layer_counts": layer_counts,
            "avg_amp_error": amp_err,
            "avg_relative_amp_error": amp_err_rel,
            "cc_amp_mean": cc_amp_mean,
            "cc_amp_std": cc_amp_std,
            "wavelength_nm": np.array([wavelengths[li] * 1e9], dtype=np.float32),
        },
    )
    print(f"✔ Metrics MAT saved -> {mat_path}")

print("\n✅ All outputs saved successfully!")
