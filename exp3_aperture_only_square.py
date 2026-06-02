"""
exp3_aperture_only_square.py
============================
实验 3: 孔径裁切实验 (Aperture Cutoff Experiment) - 方形版本

包含 3 种孔径分析:
  A. 方形 cutoff   - 只保留 max(|x|,|y|) <= a (从中央扩张)
  B. 反向方形      - 只保留 max(|x|,|y|) >= a (从外围收缩)
  C. 方形环带扫描 - 只保留 a-w/2 <= max(|x|,|y|) <= a+w/2 的方形环带

每个 a 都生成一张诊断图: 所有层裁剪后的 mask + 波长分离效率 + 模式分离效率.

直接修改下面的 USER CONFIG 段即可运行:
  python exp3_aperture_only_square.py
"""

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader, TensorDataset

from ODNN_functions import generate_complex_weights, generate_fields_ts
from odnn_io import load_complex_modes_from_mat
from odnn_processing import prepare_sample
from odnn_multiwl_model import D2NNModelMultiWL


# =============================================================================
# ============================ USER CONFIG ====================================
# =============================================================================
CKPT_PATH = "results/2modes/2wl_1.5350000239777728e-06(0.2)_base_0_ls_600_zp_0.2z_0.045_pr0.5_c5_20260529_185848/checkpoints/multiwl_10layers_m2_L2.pth"

MODE_FILE = "mmf_10modes_GRIN_176_PD1.2.mat"
DEVICE_STR = "cuda:0"      # "cuda:0" / "cuda:3" / "cpu"
SEED = 424242

# === 物理参数 (会被 ckpt meta 自动覆盖) ===
PARAMS = dict(
    field_size=176,
    layer_size=600,
    num_modes=2,
    circle_focus_radius=5,
    circle_detectsize=10,
    z_layers=45e-3,
    pixel_size=12.5e-6,
    z_prop=20e-2,
    z_input_to_first=0.0,
    padding_ratio=0.5,
    base_wavelength_idx=0,
    phase_option=4,
    batch_size=16,
)

# === 默认波长 (会从 ckpt meta 自动覆盖) ===
DEFAULT_WAVELENGTHS = np.array([1535e-9, 1536e-9], dtype=np.float32)

# === 三个子扫描的采样点配置 (a 表示半边长 in mm) ===
N_DISK = 20            # 方形 cutoff 扫描点数
N_RING = 12            # 方形环带扫描点数
RING_WIDTH_MM = 0.30   # 方形环带宽度 (mm)
B_R_INNERS_MM = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.25, 3.5]  # 反向 cutoff 列表
# =============================================================================


# =============================================================================
# 工具
# =============================================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _make_circle_mask(h: int, w: int, r: float, device: torch.device) -> torch.Tensor:
    """ROI 内部仍用圆形检测器(与训练评估一致), 这个不变"""
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)).to(torch.float32)


def generate_detector_patterns_multiwl(H, W, num_modes, num_wavelengths, radius,
                                       margin_ratio=0.2):
    num_rows = num_modes
    num_cols = num_wavelengths
    margin_x = max(int(W * margin_ratio), radius + 5)
    margin_y = max(int(H * margin_ratio), radius + 5)
    xs = np.linspace(margin_x, W - 1 - margin_x, num_cols)
    ys = np.linspace(margin_y, H - 1 - margin_y, num_rows)

    centers = []
    for mi in range(num_rows):
        for wi in range(num_cols):
            centers.append((int(round(ys[mi])), int(round(xs[wi]))))

    eval_regions = []
    for cy, cx in centers:
        x0 = max(0, int(cx - radius)); x1 = min(W, int(cx + radius))
        y0 = max(0, int(cy - radius)); y1 = min(H, int(cy + radius))
        eval_regions.append((x0, x1, y0, y1))
    return eval_regions


def build_eigenmode_dataset(mode_file: str, params: dict) -> TensorDataset:
    base = load_complex_modes_from_mat(mode_file, key="modes_field")
    nm = params["num_modes"]
    fs = params["field_size"]
    ls = params["layer_size"]

    mmf_data = base[:, :, :nm].transpose(2, 0, 1)
    mn, mx = np.min(np.abs(mmf_data)), np.max(np.abs(mmf_data))
    mmf_data = (np.abs(mmf_data) - mn) / (mx - mn + 1e-12) * np.exp(1j * np.angle(mmf_data))
    mmf_data_ts = torch.from_numpy(mmf_data)

    if params["phase_option"] == 4:
        amplitudes = np.eye(nm, dtype=np.float32)
        phases = np.eye(nm, dtype=np.float32)
        ns = nm
    else:
        amplitudes, phases = generate_complex_weights(1000, nm, params["phase_option"])
        ns = amplitudes.shape[0]

    cw = amplitudes * np.exp(1j * phases)
    cw_ts = torch.from_numpy(cw.astype(np.complex64))
    image_data = generate_fields_ts(cw_ts, mmf_data_ts, ns, nm, fs).to(torch.complex64)

    dummy_label = torch.zeros([1, ls, ls], dtype=torch.float32)
    images_prepared = []
    for i in range(ns):
        img_i, _ = prepare_sample(image_data[i], dummy_label, ls)
        images_prepared.append(img_i)
    image_tensor = torch.stack(images_prepared, dim=0)

    amp_tensor = torch.from_numpy(amplitudes.astype(np.float32))
    dummy_lb = torch.zeros((ns, 1, ls, ls), dtype=torch.float32)
    return TensorDataset(image_tensor, dummy_lb, amp_tensor)


# =============================================================================
# 评估函数
# =============================================================================
@torch.no_grad()
def evaluate_demux_ratio(model, loader, *, device, evaluation_regions,
                         detect_radius, L, num_modes):
    """波长分离效率: target-WL / all-WL ROI ratio"""
    model.eval()
    ratio_list = []
    for images, _label, _amp in loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)
        if images.ndim == 3:
            images = images.unsqueeze(1)
        x = images.repeat(1, L, 1, 1).contiguous()
        I_blhw = model(x).to(torch.float32)
        B = I_blhw.shape[0]
        ratios = torch.zeros((B, L), device=device)

        for s in range(L):
            src = I_blhw[:, s]
            E_per_wl = torch.zeros((B, L), device=device)
            for t in range(L):
                t_regions = [evaluation_regions[m * L + t] for m in range(num_modes)]
                total = torch.zeros((B,), device=device)
                for (x0, x1, y0, y1) in t_regions:
                    patch = src[:, y0:y1, x0:x1]
                    hh, ww = patch.shape[-2], patch.shape[-1]
                    cmask = _make_circle_mask(hh, ww, float(detect_radius), device=device)
                    total += (patch * cmask.unsqueeze(0)).sum(dim=(-1, -2))
                E_per_wl[:, t] = total
            ratios[:, s] = E_per_wl[:, s] / (E_per_wl.sum(dim=1) + 1e-12)
        ratio_list.append(ratios.detach().cpu())
    ratio_all = torch.cat(ratio_list, dim=0).numpy()
    return {
        "ratio_mean": float(ratio_all.mean()),
        "ratio_per_wl": ratio_all.mean(axis=0).tolist(),
    }


@torch.no_grad()
def evaluate_mode_separation_eff(
    model, dataset, *, device, evaluation_regions,
    detect_radius, L, num_modes, batch_size,
) -> np.ndarray:
    """模式分离效率 (L, num_modes)"""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    mode_eff = np.zeros((L, num_modes), dtype=np.float64)
    counts   = np.zeros(num_modes, dtype=np.int64)
    model.eval()

    for images, _label, amp in loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)
        amp    = amp.to(device, dtype=torch.float32,   non_blocking=True)
        if images.ndim == 3:
            images = images.unsqueeze(1)
        x = images.repeat(1, L, 1, 1).contiguous()
        I_blhw = model(x).to(torch.float32)
        B = I_blhw.shape[0]

        target_idx = torch.argmax(amp, dim=1).detach().cpu().numpy()
        for b in range(B):
            t = int(target_idx[b])
            for li in range(L):
                src = I_blhw[b, li]
                E_per_mode = np.zeros(num_modes, dtype=np.float64)
                for m in range(num_modes):
                    x0, x1, y0, y1 = evaluation_regions[m * L + li]
                    patch = src[y0:y1, x0:x1]
                    hh, ww = patch.shape
                    cmask = _make_circle_mask(hh, ww, float(detect_radius), device=device)
                    E_per_mode[m] = float((patch * cmask).sum().item())
                tot = E_per_mode.sum() + 1e-12
                mode_eff[li, t] += E_per_mode[t] / tot
            counts[t] += 1

    for m in range(num_modes):
        if counts[m] > 0:
            mode_eff[:, m] /= counts[m]
    return mode_eff


# =============================================================================
# 单 r_cut 诊断图
# =============================================================================
def save_aperture_diagnostic_figure(
    *, cropped_phases: list, aperture_2d: np.ndarray,
    wl_ratio_per_wl: list, mode_eff: np.ndarray,
    a_mm: float, scan_label: str,
    wavelengths: np.ndarray, out_path: Path,
):
    """单个 a 的综合诊断图: 顶部 mask + 左下 WL eff + 右下 Mode eff"""
    n_layers = len(cropped_phases)
    L = int(len(wavelengths))
    num_modes = int(mode_eff.shape[1])

    fig = plt.figure(figsize=(max(14, 2.6 * n_layers), 9))
    gs = fig.add_gridspec(2, max(n_layers, 2),
                          height_ratios=[1.0, 0.85], hspace=0.35, wspace=0.20)

    # 顶部: 每层裁剪后的 mask (圆外/方外透明)
    for i in range(n_layers):
        ax = fig.add_subplot(gs[0, i])
        ph_wrapped = np.remainder(cropped_phases[i], 2 * np.pi) / (2 * np.pi)
        rgba = plt.cm.twilight(ph_wrapped)
        rgba[..., 3] = aperture_2d.astype(np.float32)
        ax.imshow(rgba)
        ax.set_title(f"L{i+1}\n(cropped)", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("0.6")

    # 左下: 波长分离
    half = max(n_layers // 2, 1)
    ax_wl = fig.add_subplot(gs[1, :half])
    wl_idx = np.arange(L)
    colors_wl = plt.cm.viridis(np.linspace(0.2, 0.8, L))
    bars = ax_wl.bar(wl_idx, wl_ratio_per_wl, color=colors_wl, edgecolor="0.3")
    ax_wl.set_ylim(0, 1.05)
    ax_wl.set_xticks(wl_idx)
    ax_wl.set_xticklabels([f"{w*1e9:.1f} nm" for w in wavelengths])
    ax_wl.axhline(1.0 / L, color='r', linestyle=':', alpha=0.5,
                  label=f"Random (1/{L})")
    ax_wl.set_ylabel("Wavelength demux ratio\n(target-λ / all-λ ROI)")
    ax_wl.set_title("Wavelength separation efficiency")
    ax_wl.grid(True, axis='y', alpha=0.3)
    ax_wl.legend(fontsize=9, loc='lower right')
    for b, v in zip(bars, wl_ratio_per_wl):
        ax_wl.text(b.get_x() + b.get_width()/2, float(v) + 0.02,
                   f"{float(v):.3f}", ha='center', fontsize=9)

    # 右下: 模式分离 (按波长分组)
    ax_mode = fig.add_subplot(gs[1, half:])
    width = 0.8 / max(L, 1)
    mode_idx = np.arange(num_modes)
    for li in range(L):
        offset = (li - (L - 1) / 2.0) * width
        ax_mode.bar(mode_idx + offset, mode_eff[li], width=width * 0.95,
                    color=colors_wl[li], edgecolor="0.3",
                    label=f"λ={wavelengths[li]*1e9:.1f} nm")
    ax_mode.set_ylim(0, 1.05)
    ax_mode.set_xticks(mode_idx)
    ax_mode.set_xticklabels([f"M{i}" for i in mode_idx])
    ax_mode.axhline(1.0 / num_modes, color='r', linestyle=':', alpha=0.5,
                    label=f"Random (1/{num_modes})")
    ax_mode.set_ylabel("Mode separation efficiency\n(target-mode ROI / all-mode ROI in same λ)")
    ax_mode.set_title("Mode separation efficiency")
    ax_mode.grid(True, axis='y', alpha=0.3)
    ax_mode.legend(fontsize=9, loc='lower right')

    mean_wl   = float(np.mean(wl_ratio_per_wl))
    mean_mode = float(np.mean(mode_eff))
    fig.suptitle(
        f"[{scan_label}] half-side a = {a_mm:.3f} mm   |   "
        f"N_layers = {n_layers}   |   "
        f"WL eff (mean) = {mean_wl:.3f}   |   "
        f"Mode eff (mean) = {mean_mode:.3f}",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# 实验 3 主函数 (3 合 1, 方形版本)
# =============================================================================
@torch.no_grad()
def experiment_aperture_full_square(
    model, dataset, *,
    layer_size, pixel_size, z_layers,
    device, evaluation_regions, detect_radius,
    L, num_modes, batch_size, wavelengths,
    n_disk: int, n_ring: int, ring_width_mm: float,
    b_r_inners_mm: list,
    out_dir: Path,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    diag_dir = out_dir / "diagnostic_figures"
    diag_dir.mkdir(parents=True, exist_ok=True)
    n_layers = len(model.layers)

    # 物理孔径半边长 (从中心到边沿的距离)
    a_max_mm = (layer_size / 2.0) * pixel_size * 1e3

    # === 理论临界 a (方形孔径有边长方向 + 对角方向两个估计) ===
    a_critical_edge_mm = None
    a_critical_diag_mm = None
    if L >= 2:
        delta_lam = abs(float(wavelengths[1]) - float(wavelengths[0]))
        if delta_lam > 0:
            lam0 = float(wavelengths[0])
            L_coh = lam0 ** 2 / (2 * delta_lam)
            # 边长方向: a²/(2z) * N >= L_coh -> a >= sqrt(2*z*L_coh/N)
            a_critical_edge_mm = float(np.sqrt(2 * z_layers * L_coh / n_layers) * 1e3)
            # 对角方向: (a*sqrt(2))²/(2z) * N >= L_coh -> a >= sqrt(z*L_coh/N)
            a_critical_diag_mm = float(np.sqrt(z_layers * L_coh / n_layers) * 1e3)
            print(f"\n  Physics (square aperture):")
            print(f"    half-side a_max (physical)        = {a_max_mm:.3f} mm")
            print(f"    coherence length L_c              = {L_coh*1e3:.3f} mm")
            print(f"    critical a (edge-axis estimate)   = {a_critical_edge_mm:.3f} mm")
            print(f"    critical a (diagonal estimate)    = {a_critical_diag_mm:.3f} mm "
                  f"(square corners reach √2× farther)")

    # 备份原始 phase
    original_phases = [layer.phase.detach().clone() for layer in model.layers]

    # === 关键修改: 用切比雪夫距离 max(|y|,|x|) 代替欧几里得 sqrt(x²+y²) ===
    yy, xx = torch.meshgrid(
        torch.arange(layer_size, device=device),
        torch.arange(layer_size, device=device),
        indexing="ij",
    )
    cy = (layer_size - 1) / 2.0
    cx = (layer_size - 1) / 2.0
    chebyshev_dist = torch.maximum(
        torch.abs(yy - cy), torch.abs(xx - cx)
    ).to(torch.float32)   # "方形半径": max(|y|, |x|)

    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    def apply_aperture_and_eval(aperture_2d):
        for layer, orig in zip(model.layers, original_phases):
            layer.phase.data = orig * aperture_2d
        ratio = evaluate_demux_ratio(
            model, test_loader, device=device,
            evaluation_regions=evaluation_regions,
            detect_radius=detect_radius, L=L, num_modes=num_modes,
        )
        mode_eff = evaluate_mode_separation_eff(
            model, dataset, device=device,
            evaluation_regions=evaluation_regions,
            detect_radius=detect_radius, L=L, num_modes=num_modes,
            batch_size=batch_size,
        )
        return ratio, mode_eff

    def get_cropped_phases(aperture_2d):
        ap_np = aperture_2d.detach().cpu().numpy()
        cropped = []
        for orig in original_phases:
            cropped.append((orig * aperture_2d).detach().cpu().numpy())
        return cropped, ap_np

    # =========================================================================
    # A) 方形 cutoff: 只保留 max(|x|,|y|) <= a
    # =========================================================================
    A_a_values = list(np.linspace(0.3, a_max_mm, n_disk))
    A_results = []
    print("\n[A] Square cutoff (keep max(|x|,|y|) <= a):")
    print(f"  {'a(mm)':>10} {'wl_eff':>10} {'mode_eff':>10}   per_wl")
    for k, a_mm in enumerate(A_a_values):
        a_px = a_mm * 1e-3 / pixel_size
        ap = (chebyshev_dist <= a_px).to(torch.float32)
        ratio, mode_eff = apply_aperture_and_eval(ap)
        cropped_phases, ap_np = get_cropped_phases(ap)

        save_aperture_diagnostic_figure(
            cropped_phases=cropped_phases, aperture_2d=ap_np,
            wl_ratio_per_wl=ratio["ratio_per_wl"], mode_eff=mode_eff,
            a_mm=float(a_mm), scan_label=f"A-square[{k+1}/{len(A_a_values)}]",
            wavelengths=wavelengths,
            out_path=diag_dir / f"A_square_{k:02d}_a_{a_mm:.3f}mm.png",
        )

        A_results.append({
            "a_mm": float(a_mm), "a_px": float(a_px),
            "ratio_mean": ratio["ratio_mean"],
            "ratio_per_wl": ratio["ratio_per_wl"],
            "mode_eff": mode_eff.tolist(),
            "mode_eff_mean": float(np.mean(mode_eff)),
        })
        per_wl_str = "[" + ", ".join(f"{v:.3f}" for v in ratio["ratio_per_wl"]) + "]"
        print(f"  {a_mm:10.3f} {ratio['ratio_mean']:10.4f} "
              f"{float(np.mean(mode_eff)):10.4f}   {per_wl_str}")

    # =========================================================================
    # B) 反向方形: 只保留 max(|x|,|y|) >= a_inner
    # =========================================================================
    B_a_inners = [r for r in b_r_inners_mm if r < a_max_mm]
    B_results = []
    print("\n[B] Inverse square (keep max(|x|,|y|) >= a_inner):")
    print(f"  {'a_inner(mm)':>12} {'wl_eff':>10} {'mode_eff':>10}   per_wl")
    for k, a_mm in enumerate(B_a_inners):
        a_px = a_mm * 1e-3 / pixel_size
        ap = (chebyshev_dist >= a_px).to(torch.float32)
        ratio, mode_eff = apply_aperture_and_eval(ap)
        cropped_phases, ap_np = get_cropped_phases(ap)

        save_aperture_diagnostic_figure(
            cropped_phases=cropped_phases, aperture_2d=ap_np,
            wl_ratio_per_wl=ratio["ratio_per_wl"], mode_eff=mode_eff,
            a_mm=float(a_mm), scan_label=f"B-inv[{k+1}/{len(B_a_inners)}]",
            wavelengths=wavelengths,
            out_path=diag_dir / f"B_inv_{k:02d}_ainner_{a_mm:.3f}mm.png",
        )
        B_results.append({
            "a_inner_mm": float(a_mm), "a_inner_px": float(a_px),
            "ratio_mean": ratio["ratio_mean"],
            "ratio_per_wl": ratio["ratio_per_wl"],
            "mode_eff": mode_eff.tolist(),
            "mode_eff_mean": float(np.mean(mode_eff)),
        })
        per_wl_str = "[" + ", ".join(f"{v:.3f}" for v in ratio["ratio_per_wl"]) + "]"
        print(f"  {a_mm:12.3f} {ratio['ratio_mean']:10.4f} "
              f"{float(np.mean(mode_eff)):10.4f}   {per_wl_str}")

    # =========================================================================
    # C) 方形环带扫描: 只保留 a-w/2 <= max(|x|,|y|) <= a+w/2
    # =========================================================================
    margin = ring_width_mm / 2.0
    C_centers = list(np.linspace(margin, a_max_mm - margin, n_ring))
    C_results = []
    print(f"\n[C] Square ring scan (width = {ring_width_mm} mm):")
    print(f"  {'a_center(mm)':>12} {'wl_eff':>10} {'mode_eff':>10}")
    for k, a_c in enumerate(C_centers):
        a_in  = (a_c - margin) * 1e-3 / pixel_size
        a_out = (a_c + margin) * 1e-3 / pixel_size
        ap = ((chebyshev_dist >= a_in) & (chebyshev_dist <= a_out)).to(torch.float32)
        ratio, mode_eff = apply_aperture_and_eval(ap)
        cropped_phases, ap_np = get_cropped_phases(ap)

        save_aperture_diagnostic_figure(
            cropped_phases=cropped_phases, aperture_2d=ap_np,
            wl_ratio_per_wl=ratio["ratio_per_wl"], mode_eff=mode_eff,
            a_mm=float(a_c), scan_label=f"C-ring[{k+1}/{len(C_centers)}]",
            wavelengths=wavelengths,
            out_path=diag_dir / f"C_ring_{k:02d}_ac_{a_c:.3f}mm.png",
        )
        C_results.append({
            "a_center_mm": float(a_c),
            "ratio_mean": ratio["ratio_mean"],
            "ratio_per_wl": ratio["ratio_per_wl"],
            "mode_eff": mode_eff.tolist(),
            "mode_eff_mean": float(np.mean(mode_eff)),
        })
        print(f"  {a_c:12.3f} {ratio['ratio_mean']:10.4f} "
              f"{float(np.mean(mode_eff)):10.4f}")

    # =========================================================================
    # 恢复原始 phase
    # =========================================================================
    for layer, orig in zip(model.layers, original_phases):
        layer.phase.data = orig

    # =========================================================================
    # 总览图
    # =========================================================================
    rs_A = np.array([r["a_mm"]         for r in A_results])
    rA_w = np.array([r["ratio_mean"]   for r in A_results])
    rA_m = np.array([r["mode_eff_mean"] for r in A_results])

    rs_B = np.array([r["a_inner_mm"]   for r in B_results])
    rB_w = np.array([r["ratio_mean"]   for r in B_results])
    rB_m = np.array([r["mode_eff_mean"] for r in B_results])

    rs_C = np.array([r["a_center_mm"]  for r in C_results])
    rC_w = np.array([r["ratio_mean"]   for r in C_results])
    rC_m = np.array([r["mode_eff_mean"] for r in C_results])

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    def _plot_dual(ax, rs, w, m, xlabel, title, mark_critical=False):
        ax.plot(rs, w, "o-", linewidth=2, markersize=7,
                label="WL eff", color="tab:blue")
        ax.plot(rs, m, "s--", linewidth=2, markersize=7,
                label="Mode eff", color="tab:orange")
        ax.axhline(1.0/L, color="b", linestyle=":", alpha=0.4,
                   label=f"WL random (1/{L})")
        ax.axhline(1.0/num_modes, color="orange", linestyle=":", alpha=0.4,
                   label=f"Mode random (1/{num_modes})")
        if mark_critical and a_critical_edge_mm is not None and a_critical_edge_mm < a_max_mm:
            ax.axvline(a_critical_edge_mm, color="g", linestyle="--", alpha=0.6,
                       label=f"a_crit (edge) ≈ {a_critical_edge_mm:.2f}mm")
        if mark_critical and a_critical_diag_mm is not None and a_critical_diag_mm < a_max_mm:
            ax.axvline(a_critical_diag_mm, color="lime", linestyle=":", alpha=0.6,
                       label=f"a_crit (diag) ≈ {a_critical_diag_mm:.2f}mm")
        ax.set_xlabel(xlabel); ax.set_ylabel("Efficiency")
        ax.set_ylim(0, 1.05)
        ax.set_title(title); ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    _plot_dual(axes[0], rs_A, rA_w, rA_m, "Half-side a (mm)",
               "(A) Square cutoff: keep max(|x|,|y|) ≤ a", mark_critical=True)
    _plot_dual(axes[1], rs_B, rB_w, rB_m, "Inner half-side a (mm)",
               "(B) Inverse square: keep max(|x|,|y|) ≥ a")
    _plot_dual(axes[2], rs_C, rC_w, rC_m, "Square ring center a (mm)",
               f"(C) Square ring scan (w={ring_width_mm}mm)", mark_critical=True)

    title_dlam = (f"Δλ = {abs(wavelengths[1]-wavelengths[0])*1e9:.2f} nm"
                  if L >= 2 else "single λ")
    fig.suptitle(f"Square-Aperture Analysis: WL & Mode separation efficiency "
                 f"({title_dlam}, N={n_layers} layers)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    summary_path = out_dir / "aperture_full_analysis.png"
    fig.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 保存 .mat
    # =========================================================================
    pwA  = np.array([r["ratio_per_wl"] for r in A_results])
    meA  = np.array([r["mode_eff"]     for r in A_results])
    pwB  = np.array([r["ratio_per_wl"] for r in B_results])
    meB  = np.array([r["mode_eff"]     for r in B_results])
    pwC  = np.array([r["ratio_per_wl"] for r in C_results])
    meC  = np.array([r["mode_eff"]     for r in C_results])

    savemat(str(out_dir / "aperture_full_analysis.mat"), {
        # A
        "A_a_mm":              rs_A,
        "A_wl_eff_mean":       rA_w,
        "A_wl_eff_per_wl":     pwA,
        "A_mode_eff_mean":     rA_m,
        "A_mode_eff_full":     meA,
        # B
        "B_a_inner_mm":        rs_B,
        "B_wl_eff_mean":       rB_w,
        "B_wl_eff_per_wl":     pwB,
        "B_mode_eff_mean":     rB_m,
        "B_mode_eff_full":     meB,
        # C
        "C_a_center_mm":       rs_C,
        "C_ring_width_mm":     float(ring_width_mm),
        "C_wl_eff_mean":       rC_w,
        "C_wl_eff_per_wl":     pwC,
        "C_mode_eff_mean":     rC_m,
        "C_mode_eff_full":     meC,
        # Physics
        "wavelengths_m":       np.asarray(wavelengths, dtype=np.float64),
        "wavelengths_nm":      np.asarray(wavelengths, dtype=np.float64) * 1e9,
        "n_layers":            np.asarray([n_layers], dtype=np.int32),
        "z_layers_m":          np.asarray([z_layers], dtype=np.float64),
        "pixel_size_m":        np.asarray([pixel_size], dtype=np.float64),
        "a_max_mm":            np.asarray([a_max_mm], dtype=np.float64),
        "a_critical_edge_mm":  np.asarray(
            [a_critical_edge_mm if a_critical_edge_mm is not None else np.nan],
            dtype=np.float64),
        "a_critical_diag_mm":  np.asarray(
            [a_critical_diag_mm if a_critical_diag_mm is not None else np.nan],
            dtype=np.float64),
        "aperture_shape":      np.array(["square"]),
    })

    print(f"\n  ✔ Summary plot   -> {summary_path}")
    print(f"  ✔ Diagnostic figs -> {diag_dir}/  "
          f"({len(A_results)+len(B_results)+len(C_results)} files)")
    print(f"  ✔ Data           -> {out_dir / 'aperture_full_analysis.mat'}")

    return {"A": A_results, "B": B_results, "C": C_results,
            "a_critical_edge_mm": a_critical_edge_mm,
            "a_critical_diag_mm": a_critical_diag_mm,
            "a_max_mm": a_max_mm}


# =============================================================================
# Main
# =============================================================================
def main():
    set_seed(SEED)

    if torch.cuda.is_available() and DEVICE_STR.startswith("cuda"):
        device = torch.device(DEVICE_STR)
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # --- 加载 ckpt ---
    ckpt_path = Path(CKPT_PATH).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    print(f"\nLoading checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)

    meta = ckpt.get("meta", {})

    num_layers = int(meta.get("num_layers", 7))
    L = int(meta.get("num_wavelengths", 2))
    num_modes = int(meta.get("num_modes", PARAMS["num_modes"]))
    wavelengths = np.asarray(
        meta.get("wavelengths", DEFAULT_WAVELENGTHS), dtype=np.float32
    )

    for k in ("layer_size", "z_layers", "z_prop", "pixel_size",
              "padding_ratio", "z_input_to_first", "base_wavelength_idx",
              "circle_focus_radius", "circle_detectsize",
              "field_size", "phase_option", "num_modes"):
        if k in meta and meta[k] is not None:
            PARAMS[k] = meta[k]
    PARAMS["num_modes"] = num_modes

    print(f"\n=== Resolved parameters ===")
    print(f"  num_layers          = {num_layers}")
    print(f"  num_wavelengths (L) = {L}")
    print(f"  num_modes           = {num_modes}")
    print(f"  wavelengths_nm      = {(wavelengths*1e9).tolist()}")
    if L >= 2:
        print(f"  Δλ                  = {abs(wavelengths[1]-wavelengths[0])*1e9:.4f} nm")
    print(f"  layer_size          = {PARAMS['layer_size']}")
    print(f"  z_layers            = {PARAMS['z_layers']*1e3:.3f} mm")
    print(f"  z_prop              = {PARAMS['z_prop']*1e3:.1f} mm")
    print(f"  pixel_size          = {PARAMS['pixel_size']*1e6:.2f} µm")
    print(f"  Aperture shape      = SQUARE (Chebyshev/L∞ norm)")

    if L < 2:
        raise RuntimeError(
            f"This experiment requires L >= 2 (multi-wavelength). "
            f"Got L={L}. Aperture cutoff makes no sense for single λ."
        )

    # --- 构建模型并加载权重 ---
    model = D2NNModelMultiWL(
        num_layers=num_layers,
        layer_size=PARAMS["layer_size"],
        z_layers=PARAMS["z_layers"],
        z_prop=PARAMS["z_prop"],
        pixel_size=PARAMS["pixel_size"],
        wavelengths=wavelengths,
        device=device,
        padding_ratio=PARAMS["padding_ratio"],
        z_input_to_first=float(PARAMS["z_input_to_first"]),
        base_wavelength_idx=PARAMS["base_wavelength_idx"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # --- 输出目录 ---
    out_root = ckpt_path.parent.parent / f"exp3_aperture_SQUARE_{ckpt_path.stem}"
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput -> {out_root}")

    with open(out_root / "aperture_meta.json", "w") as f:
        json.dump({
            "ckpt": str(ckpt_path),
            "aperture_shape": "square",
            "num_layers": num_layers,
            "num_wavelengths": L,
            "num_modes": num_modes,
            "wavelengths_m": wavelengths.tolist(),
            "wavelengths_nm": (wavelengths * 1e9).tolist(),
            "params": {k: (v.tolist() if isinstance(v, np.ndarray)
                           else (float(v) if isinstance(v, (np.floating,))
                                 else (int(v) if isinstance(v, (np.integer,))
                                       else v)))
                       for k, v in PARAMS.items()},
            "scan_config": {
                "n_disk":          N_DISK,
                "n_ring":          N_RING,
                "ring_width_mm":   RING_WIDTH_MM,
                "b_r_inners_mm":   B_R_INNERS_MM,
            },
        }, f, indent=2)

    # --- detector layout ---
    evaluation_regions = generate_detector_patterns_multiwl(
        H=PARAMS["layer_size"], W=PARAMS["layer_size"],
        num_modes=num_modes, num_wavelengths=L,
        radius=PARAMS["circle_focus_radius"], margin_ratio=0.2,
    )
    detect_radius = int(PARAMS["circle_detectsize"] // 2)

    # --- dataset ---
    print("\nBuilding eigenmode dataset...")
    dataset = build_eigenmode_dataset(MODE_FILE, PARAMS)
    print(f"  ✔ {len(dataset)} samples")

    # --- baseline (no aperture) ---
    print("\nBaseline (no aperture cutoff):")
    test_loader = DataLoader(dataset, batch_size=PARAMS["batch_size"], shuffle=False)
    baseline = evaluate_demux_ratio(
        model, test_loader, device=device,
        evaluation_regions=evaluation_regions,
        detect_radius=detect_radius, L=L, num_modes=num_modes,
    )
    baseline_mode_eff = evaluate_mode_separation_eff(
        model, dataset, device=device,
        evaluation_regions=evaluation_regions,
        detect_radius=detect_radius, L=L, num_modes=num_modes,
        batch_size=PARAMS["batch_size"],
    )
    print(f"  Baseline WL eff   = {baseline['ratio_mean']:.4f} "
          f"(per_wl: {baseline['ratio_per_wl']})")
    print(f"  Baseline Mode eff = {float(np.mean(baseline_mode_eff)):.4f}")

    # --- 跑实验 ---
    print("\n" + "=" * 70)
    print("Experiment 3 (SQUARE): Aperture cutoff (3-in-1 with diagnostic figures)")
    print("=" * 70)
    experiment_aperture_full_square(
        model, dataset,
        layer_size=PARAMS["layer_size"],
        pixel_size=PARAMS["pixel_size"],
        z_layers=PARAMS["z_layers"],
        device=device,
        evaluation_regions=evaluation_regions,
        detect_radius=detect_radius,
        L=L, num_modes=num_modes,
        batch_size=PARAMS["batch_size"],
        wavelengths=wavelengths,
        n_disk=N_DISK,
        n_ring=N_RING,
        ring_width_mm=RING_WIDTH_MM,
        b_r_inners_mm=B_R_INNERS_MM,
        out_dir=out_root,
    )

    # --- sanity check ---
    print("\n" + "=" * 70)
    print("Sanity check: original demux ratio after experiment")
    print("=" * 70)
    sanity = evaluate_demux_ratio(
        model, test_loader, device=device,
        evaluation_regions=evaluation_regions,
        detect_radius=detect_radius, L=L, num_modes=num_modes,
    )
    print(f"  Recovered demux ratio = {sanity['ratio_mean']:.4f} "
          f"(per_wl: {sanity['ratio_per_wl']})")
    if abs(sanity['ratio_mean'] - baseline['ratio_mean']) > 1e-4:
        print("  ⚠️  WARNING: phase 未完全恢复, 结果可能受影响")
    else:
        print("  ✔ Phase correctly restored")

    print("\n" + "=" * 70)
    print(f"✅ Experiment 3 (SQUARE) done!")
    print(f"   Results in: {out_root}")
    print("=" * 70)


if __name__ == "__main__":
    main()
