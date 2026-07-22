import argparse
import json
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

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Using Device:", device)


# ============================================================
# Parameters
# ============================================================
field_size = 176
layer_size = 300
num_modes = 2

circle_focus_radius = 5
circle_detectsize = 10
focus_radius = circle_focus_radius
detectsize = circle_detectsize

batch_size = 16

evaluation_mode = "eigenmode"      # "eigenmode" or "superposition"
training_dataset_mode = "eigenmode"    # "eigenmode" or "superposition"

num_superposition_eval_samples = 1000
num_superposition_train_samples = 100
superposition_eval_seed = 20240116
superposition_train_seed = 20240115

num_layer_option = [1, 2, 3, 4]

# SLM
z_layers = 45e-3
pixel_size = 12.5e-6
z_prop = 20e-2
z_input_to_first = 0
out_size = 600
padding_ratio_out = 0.5

# ============================================================
# Wavelengths (MultiWL) — 起始波长 + 间隔 + 数量
# ============================================================
wl_start_nm = 1550       # 起始波长 (nm)
wl_spacing_nm = 0.5         # 波长间隔 (nm)
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
margin_ratio = 0.2

# train hyperparams
epochs = 1000
lr = 1.99
lr_gamma = 0.99
padding_ratio = 0.5

# prediction viz samples
num_pred_diag_samples = 3
num_superposition_visual_samples = 2


# ============================================================
# frontend config override (--config + --mat_file)
# ============================================================
_p = argparse.ArgumentParser(add_help=False)
_p.add_argument("--config",     type=str, default=None)
_p.add_argument("--mat_file",   type=str, default=None)
_p.add_argument("--output_dir", type=str, default=None)
_cli, _ = _p.parse_known_args()

_cfg: dict = {}
if _cli.config:
    with open(_cli.config) as _f:
        _cfg = json.load(_f)

if _cfg:
    field_size            = int(_cfg.get("field_size",            field_size))
    layer_size            = int(_cfg.get("layer_size",            layer_size))
    out_size              = int(_cfg.get("out_size",              out_size))
    padding_ratio_out     = float(_cfg.get("padding_ratio_out",     padding_ratio_out))
    num_modes             = int(_cfg.get("num_modes",             num_modes))
    batch_size            = int(_cfg.get("batch_size",            batch_size))
    epochs                = int(_cfg.get("epochs",                epochs))
    lr                    = float(_cfg.get("lr",                  lr))
    lr_gamma              = float(_cfg.get("lr_gamma",            lr_gamma))
    padding_ratio         = float(_cfg.get("padding_ratio",       padding_ratio))
    z_layers              = float(_cfg.get("z_layers_um",         z_layers * 1e6))   * 1e-6
    z_prop                = float(_cfg.get("z_prop_um",           z_prop * 1e6))     * 1e-6
    z_input_to_first      = float(_cfg.get("z_input_to_first_um", z_input_to_first * 1e6)) * 1e-6
    pixel_size            = float(_cfg.get("pixel_size_um",       pixel_size * 1e6)) * 1e-6
    wl_start_nm   = float(_cfg.get("wl_start_nm", wl_start_nm))
    wl_spacing_nm = float(_cfg.get("wl_spacing_nm", wl_spacing_nm))
    wl_count      = int(_cfg.get("wl_count", wl_count))
    wavelengths   = (wl_start_nm + np.arange(wl_count) * wl_spacing_nm).astype(np.float32) * 1e-9
    L                     = int(len(wavelengths))
    base_wavelength_idx   = int(_cfg.get("base_wavelength_idx",   base_wavelength_idx))
    training_dataset_mode = str(_cfg.get("training_dataset_mode", training_dataset_mode))
    evaluation_mode       = str(_cfg.get("evaluation_mode",       evaluation_mode))
    phase_option          = int(_cfg.get("phase_option",          phase_option))
    label_pattern_mode    = str(_cfg.get("label_pattern_mode",    label_pattern_mode))
    circle_focus_radius   = int(_cfg.get("circle_focus_radius",   circle_focus_radius))
    margin_ratio          = float(_cfg.get("margin_ratio",        margin_ratio))
    circle_detectsize     = int(_cfg.get("circle_detectsize",   circle_detectsize))
    label_config          = _cfg.get("label_config", None)  # New: from Dataset & Label Designer
    num_layers_list_cfg   = _cfg.get("num_layers_list", num_layer_option)
    if isinstance(num_layers_list_cfg, list):
        num_layer_option = [int(x) for x in num_layers_list_cfg]
    elif isinstance(num_layers_list_cfg, (int, float)):
        num_layer_option = [int(num_layers_list_cfg)]
    else:
        num_layer_option = [int(_cfg.get("num_layers", num_layer_option[0]))]
    if _cli.output_dir:
        RUN_ROOT = Path(_cli.output_dir)
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
    elif _cfg.get("output_dir"):
        RUN_ROOT = Path(_cfg["output_dir"])
        RUN_ROOT.mkdir(parents=True, exist_ok=True)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        RUN_ROOT = Path(
            f"results/"
            f"{num_modes}modes/"
            f"{L}wl_{wl_start_nm:.1f}nm_sp{wl_spacing_nm:.1f}nm_"
            f"base{base_wavelength_idx}_"
            f"ls{layer_size}_out_{out_size}_zp{z_prop*1e3:.0f}mm_z{z_layers*1e3:.1f}mm_"
            f"pr{padding_ratio}_c{circle_focus_radius}_"
            f"{timestamp}"
        )
        RUN_ROOT.mkdir(parents=True, exist_ok=True)

# metrics log path for frontend monitoring
if _cli.output_dir or _cfg.get("output_dir"):
    _METRICS_LOG = RUN_ROOT / "logs" / "metrics_wl.jsonl"
    _METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
    _METRICS_LOG.write_text("")
    _train_log = RUN_ROOT / "logs" / "training_wl.log"
else:
    _METRICS_LOG = Path(__file__).resolve().parent / "frontend" / "logs" / "metrics_wl.jsonl"
    _METRICS_LOG.parent.mkdir(parents=True, exist_ok=True)
    _METRICS_LOG.write_text("")
    _train_log = Path(__file__).resolve().parent / "frontend" / "logs" / "training_wl.log"

# ============================================================
# 多波长标签生成函数
# ============================================================
def generate_detector_patterns_multiwl(
    H: int,
    W: int,
    num_modes: int,
    num_wavelengths: int,
    radius: int,
    pattern_mode: str = "circle",
    show_debug: bool = False,
    margin_ratio: float = 0.2,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    total_labels = num_modes * num_wavelengths
    margin_x = int(W * margin_ratio)
    margin_y = int(H * margin_ratio)
    min_margin = radius + 5
    margin_x = max(margin_x, min_margin)
    margin_y = max(margin_y, min_margin)

    # Within each wavelength band, lay out modes in a grid
    inner_margin = radius + 3
    total_per_wl = num_modes
    avail_y = H - 2 * margin_y
    band_h = avail_y / max(num_wavelengths, 1)
    band_w = W - 2 * margin_x
    ncols = max(1, min(total_per_wl, int(np.ceil(np.sqrt(total_per_wl * band_w / max(band_h, 1))))))
    nrows = int(np.ceil(total_per_wl / ncols))

    centers = []
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wavelengths):
            band_y0 = margin_y + wl_idx * band_h
            band_y1 = margin_y + (wl_idx + 1) * band_h
            row = mode_idx // ncols
            col = mode_idx % ncols
            xs_arr = np.linspace(margin_x, W - 1 - margin_x, ncols)
            ys_arr = np.linspace(band_y0 + inner_margin, band_y1 - inner_margin, nrows)
            cx = int(round(xs_arr[col]))
            cy = int(round(ys_arr[row]))
            centers.append((cy, cx))

    if pattern_mode == "circle":
        patterns = np.zeros((H, W, total_labels), dtype=np.float32)
        for idx, (cy, cx) in enumerate(centers):
            yy, xx = np.ogrid[:H, :W]
            mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            patterns[:, :, idx] = mask.astype(np.float32)
    else:
        raise NotImplementedError(f"Unsupported pattern_mode: {pattern_mode}")

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
        ax.axvline(margin_x, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(W - margin_x, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(margin_y, color='cyan', linestyle='--', linewidth=1, alpha=0.5)
        ax.axhline(H - margin_y, color='cyan', linestyle='--', linewidth=1, alpha=0.5)

        plt.title(f"MultiWL Labels: {num_modes} modes × {num_wavelengths} wavelengths\n"
                 f"Margin: {margin_ratio * 100:.0f}% ({margin_x}×{margin_y} pixels)")
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
    mmf_modes: torch.Tensor,
    layer_size: int,
    device: torch.device,
) -> dict:
    """Group-level isolation for mode-group labels.

    Each (group, wavelength) pair has one ROI. Mode energy is aggregated
    per ROI, and group isolation = 10*log10(E_target / sum(E_other_groups)).

    Returns dict with:
      - per_mode_per_wl_db: (M, L) array
      - mean_db: scalar
      - group_crosstalk_matrix: (L, G, G) per-wavelength
      - mode_groups: the input grouping
    """
    model.eval()
    M = num_modes
    L = num_wavelengths
    G = len(mode_groups)

    mode_to_group = {m: g for g, modes in enumerate(mode_groups) for m in modes}

    # representative mode per group (first one)
    rep_mode_per_group = [g[0] for g in mode_groups]
    group_regions_per_wl: list[list[tuple[int, int, int, int]]] = []
    for wl_idx in range(L):
        regs = []
        for g_idx in range(G):
            m_rep = rep_mode_per_group[g_idx]
            regs.append(evaluation_regions[m_rep * L + wl_idx])
        group_regions_per_wl.append(regs)

    per_mode_per_wl_db = np.zeros((M, L), dtype=np.float64)
    energy_mlg = np.zeros((M, L, G), dtype=np.float64)

    for m_idx in range(M):
        mode_field = mmf_modes[m_idx].to(device=device, dtype=torch.complex64)
        padded = pad_field_to_layer(mode_field, layer_size)
        x = padded[None, None, ...].repeat(1, L, 1, 1).contiguous()
        I_pred = model(x)
        I_pred = I_pred[0].to(torch.float32)

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

    # group crosstalk matrix per wavelength
    group_crosstalk = np.zeros((L, G, G), dtype=np.float64)
    for wl_idx in range(L):
        for src_g in range(G):
            src_modes = mode_groups[src_g]
            row = energy_mlg[src_modes, wl_idx, :].sum(axis=0)
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
_mat_path = _cli.mat_file if _cli.mat_file else "mmf_10modes_GRIN_176_PD1.2.mat"
eigenmodes_OM4, _mode_info = load_complex_modes_from_mat(_mat_path, key="modes_field")
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
print(f"Generating MultiWL Labels: {num_modes} modes × {L} wavelengths = {num_modes * L} labels")
if label_config:
    print(f"  label_type:  {label_config.get('label_type', 'modes')}")
    print(f"  label_shape: {label_config.get('label_shape', 'circle')}")
print(f"{'=' * 60}")

if label_config:
    # Use centralized label designer
    from label_designer import generate_labels
    try:
        mmf_label_patterns, evaluation_regions, _per_label_radii = generate_labels(
            out_size=out_size,
            num_modes=num_modes,
            num_wavelengths=L,
            label_config=label_config,
            modes_field=eigenmodes_OM4,
            mode_info=_mode_info,
            show_debug=show_detection_overlap_debug,
            debug_save_path=str(RUN_ROOT / "debug_multiwl_labels.png"),
        )
        print(f"✔ Generated {len(evaluation_regions)} evaluation regions (via label_designer)")
    except Exception as e:
        print(f"[WARN] label_designer failed ({e}), falling back to circle mode")
        label_config = None  # fall through to default

if not label_config:
    # Fallback: use inline function (backward compat). Only "circle" is supported here;
    # other shapes require label_designer via label_config.
    _pmode = label_pattern_mode if label_pattern_mode == "circle" else "circle"
    if label_pattern_mode != "circle":
        print(f"[WARN] label_pattern_mode='{label_pattern_mode}' not supported without label_config, "
              f"falling back to 'circle'. Use Dataset & Label Designer to enable custom shapes.")
    mmf_label_patterns, evaluation_regions = generate_detector_patterns_multiwl(
        H=out_size,
        W=out_size,
        num_modes=num_modes,
        num_wavelengths=L,
        radius=circle_focus_radius,
        pattern_mode=_pmode,
        show_debug=show_detection_overlap_debug,
        margin_ratio=margin_ratio,
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
        scheduler_gamma=lr_gamma,
        stage_ratios=[0.25, 0.25, 0.25, 0.25],
        verbose=True,
        num_layer=num_layer,
        total_layers=len(num_layer_option),
        metrics_path=str(_METRICS_LOG),
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
            "field_size": int(field_size),
            "num_modes": int(num_modes),
            "num_wavelengths": int(L),
            "wavelengths": wavelengths.astype(np.float32),
            "base_wavelength_idx": int(base_wavelength_idx),
            "z_layers": float(z_layers),
            "z_prop": float(z_prop),
            "z_input_to_first": float(z_input_to_first),
            "pixel_size": float(pixel_size),
            "padding_ratio": float(padding_ratio),
            "out_size": int(out_size),
            "padding_ratio_out": float(padding_ratio_out),
            "wl_start_nm": float(wl_start_nm),
            "wl_spacing_nm": float(wl_spacing_nm),
            "circle_focus_radius": int(circle_focus_radius),
            "margin_ratio": float(margin_ratio),
            "phase_option": int(phase_option),
            "circle_detectsize": int(circle_detectsize),
            "total_training_time_sec": float(total_time),
            **({"label_config": label_config} if label_config else {}),
        },
    )
    print("✔ Checkpoint saved ->", ckpt_path)

    # 相位掩模
    phase_masks = extract_phase_masks_multiwl(model)
    if phase_masks:
        pm_dir = RUN_ROOT / "phase_masks" / f"{num_layer}L_{L}wl_{wl_start_nm:.0f}nm_sp{wl_spacing_nm:.1f}nm"
        pm_mat = pm_dir / f"phase_masks_{num_layer}L_{L}wl_{wl_start_nm:.0f}nm.mat"
        base_name=f"mask_{num_layer}L_{L}wl_{wl_start_nm:.0f}nm_sp{wl_spacing_nm:.1f}nm"
        pm_dir.mkdir(parents=True, exist_ok=True)  
        savemat(str(pm_mat), {"phase_masks": np.stack(phase_masks, axis=0).astype(np.float32)})
        print(f"✔ Phase masks saved -> {pm_mat}")

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
    diag_dir = RUN_ROOT / "prediction_viz" / f"L{num_layer}_{run_tag}"
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

    # ============================================================
    # Group Isolation (mode_groups label)
    # ============================================================
    if label_config and label_config.get("label_type") == "mode_groups":
        from label_designer import _groups_from_mode_info as _gfi
        _mg = label_config.get("mode_groups")
        if _mg is None and _mode_info is not None:
            _mg = _gfi(_mode_info, num_modes)
        if _mg:
            print(f"\n--- Group Isolation ({len(_mg)} groups) ---")
            group_metrics = evaluate_group_isolation_db(
                model=model,
                mode_groups=_mg,
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
            print(f"  Mean Group Isolation: {group_metrics['mean_db']:.2f} dB")
            for wl_idx in range(L):
                print(f"  λ={wavelengths[wl_idx]*1e9:.1f} nm: "
                      f"{10*np.log10(np.clip(np.diag(group_metrics['group_crosstalk_matrix'][wl_idx]), 1e-6, None))}")

    # 每层追加写一行JSON，供 batch_train_wl.py 收集
    try:
        _summary_log = RUN_ROOT / "logs" / "summary_metrics_wl.jsonl"
        _summary_log.parent.mkdir(parents=True, exist_ok=True)
        with open(_summary_log, "a") as _sf:
            _sf.write(json.dumps({
                "num_modes": int(num_modes),
                "num_layers": int(num_layer),
                "snr_db_mean": float(comp_metrics["snr_db_mean"]),
                "mode_isolation_db_mean": float(comp_metrics["mode_isolation_db_mean"]),
                "target_all_roi_ratio_mean": float(comp_metrics["target_all_roi_ratio_mean"]),
                "throughput_mean": float(np.mean(comp_metrics["throughput_per_mode"])),
            }) + "\n")
    except Exception as _e:
        print(f"WARN: write summary_metrics_wl.jsonl failed: {_e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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
    savemat(str(metrics_dir / f"metrics_summary_{tag}.mat"), {
        "layers": layer_counts.astype(np.float64),
        "wavelengths_m": wavelengths.astype(np.float64),
        "snr_db_mean": snr_mean_arr,
        # 三维 isolation
        "mode_isolation_db_mean": mode_iso_arr,
        "wavelength_isolation_db_mean": wl_iso_arr,
        "target_all_roi_db_mean": target_all_roi_db_arr,
        # insertion loss
        "insertion_loss_db_mean": il_arr,
    })
    print(f"✔ Metrics .mat saved")

print("\n✅ All outputs saved successfully!")
