"""
多波长综合 Metrics 模块
计算与单波长 evaluate_spot_metrics 对应的指标，
额外增加 wavelength isolation / mode isolation / target-all-ROI
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy.io import savemat

from odnn_processing import pad_field_to_layer


# ============================================================
# 核心：多波长综合 Metrics
# ============================================================
@torch.no_grad()
def evaluate_multiwl_comprehensive_metrics(
    model: nn.Module,
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    *,
    detect_radius: int,
    device: torch.device,
    num_modes: int,
    num_wavelengths: int,
    wavelengths_m: np.ndarray,
    mmf_modes: torch.Tensor,       # (M, H_field, W_field) complex
    layer_size: int,
) -> Dict[str, Any]:
    """
    多波长 D2NN 综合评估指标（类似单波长 evaluate_spot_metrics）。

    evaluation_regions 索引约定：
        regions[mode_k * L + wl_idx]
        其中 L = num_wavelengths

    输入为每个本征模式（复制到 L 个波长通道），
    模型输出 (1, L, H, W) 强度图。

    Returns
    -------
    dict with keys:
        ── Mode Isolation ──
        mode_isolation_db           : (M, L) 每个 (模式, 波长) 的模式隔离 dB
        mode_isolation_db_mean      : float  平均模式隔离
        mode_isolation_db_per_wl    : (L,)   每个波长的平均模式隔离

        ── Wavelength Isolation ──
        wavelength_isolation_db     : (M, L) 每个 (模式, 波长) 的波长隔离 dB
        wavelength_isolation_db_mean: float  平均波长隔离
        wavelength_isolation_db_per_mode: (M,) 每个模式的平均波长隔离

        ── Target / All ROI ──
        target_all_roi_ratio        : (M,)   每个模式的 target/all ROI 比
        target_all_roi_ratio_mean   : float
        target_all_roi_db           : (M,)   target/all ROI in dB

        ── Crosstalk ──
        crosstalk_matrix_per_wl     : (L, M, M)  每个波长的模式串扰矩阵（归一化）
        crosstalk_matrix_wl         : (M, L, L)  每个模式的波长串扰矩阵（归一化）

        ── Insertion Loss ──
        insertion_loss_db           : (M,)
        insertion_loss_db_mean      : float

        ── SNR ──
        snr_db                      : (M, L)  每个 (模式, 波长) 的 ROI SNR
        snr_db_mean                 : float

        ── Throughput ──
        throughput_per_mode         : (M,)   输出总能量/输入总能量

        ── Raw ──
        energy_matrix               : (M, L, M) [input_mode, output_wl, roi_mode]
        total_energy_output         : (M, L)
        input_energy                : (M,)
    """
    model.eval()
    L = int(num_wavelengths)
    M = int(num_modes)
    eps = 1e-12

    # 实际积分半径（与 evaluate_spot_metrics 保持一致用 detect_radius/2）
    r_eff = max(1, int(round(detect_radius / 2.0)))

    # ── 构建每个 ROI 的圆形 mask（numpy, 预计算） ──
    def _build_circle_mask(x0, x1, y0, y1, radius):
        hh = y1 - y0
        ww = x1 - x0
        yy, xx = np.ogrid[:hh, :ww]
        cy = (hh - 1) / 2.0
        cx = (ww - 1) / 2.0
        return ((yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2)

    roi_masks = []
    for reg in evaluation_regions:
        x0, x1, y0, y1 = reg
        roi_masks.append(_build_circle_mask(x0, x1, y0, y1, r_eff))

    # ── 能量矩阵 E[input_mode, output_wl, roi_mode] ──
    energy_matrix = np.zeros((M, L, M), dtype=np.float64)
    total_energy_output = np.zeros((M, L), dtype=np.float64)
    input_energy = np.zeros(M, dtype=np.float64)

    for m_idx in range(M):
        # 准备输入: eigenmode m → pad → (1, L, H, W)
        mode_field = mmf_modes[m_idx].to(device=device, dtype=torch.complex64)
        padded = pad_field_to_layer(mode_field, layer_size)
        input_energy[m_idx] = float((padded.abs() ** 2).sum().item())

        x = padded[None, None, ...].repeat(1, L, 1, 1).contiguous()  # (1, L, H, W)

        # 前向传播
        I_blhw = model(x)  # (1, L, H, W) float intensity
        I_lhw = I_blhw[0].detach().cpu().numpy().astype(np.float64)  # (L, H, W)

        for l_idx in range(L):
            I_hw = I_lhw[l_idx]
            total_energy_output[m_idx, l_idx] = float(I_hw.sum())

            # 计算该模式输入、该波长输出面上每个模式 ROI 的能量
            for j in range(M):
                roi_idx = j * L + l_idx
                x0, x1, y0, y1 = evaluation_regions[roi_idx]
                patch = I_hw[y0:y1, x0:x1]
                mask = roi_masks[roi_idx]
                # 确保 mask 和 patch 形状一致
                h_p, w_p = patch.shape
                h_m, w_m = mask.shape
                h_use = min(h_p, h_m)
                w_use = min(w_p, w_m)
                energy_matrix[m_idx, l_idx, j] = float(
                    patch[:h_use, :w_use][mask[:h_use, :w_use]].sum()
                )

    # ================================================================
    # 1. Mode Isolation (模式隔离)
    #    对于输入模式 m，在波长 l 的输出面上：
    #    iso = E[m,l,m] / Σ_{j≠m} E[m,l,j]
    # ================================================================
    mode_isolation_db = np.zeros((M, L), dtype=np.float64)
    for m in range(M):
        for l in range(L):
            signal = energy_matrix[m, l, m]
            noise = energy_matrix[m, l, :].sum() - signal
            ratio = signal / max(noise, eps)
            mode_isolation_db[m, l] = 10.0 * np.log10(max(ratio, eps))

    mode_isolation_db_mean = float(np.mean(mode_isolation_db))
    mode_isolation_db_per_wl = np.mean(mode_isolation_db, axis=0)  # (L,)

    # ================================================================
    # 2. Wavelength Isolation (波长隔离)
    #    对于输入模式 m，在目标波长 l：
    #    iso = E[m,l,m] / Σ_{l'≠l} E[m,l',m]
    #    即：该模式能量在目标波长 vs 其他波长的同一模式 ROI
    # ================================================================
    wavelength_isolation_db = np.zeros((M, L), dtype=np.float64)
    for m in range(M):
        for l in range(L):
            signal = energy_matrix[m, l, m]
            noise = sum(energy_matrix[m, lp, m] for lp in range(L) if lp != l)
            if L == 1:
                # 单波长时无意义，设为 inf
                wavelength_isolation_db[m, l] = float('inf')
            else:
                ratio = signal / max(noise, eps)
                wavelength_isolation_db[m, l] = 10.0 * np.log10(max(ratio, eps))

    # 对 inf 做特殊处理
    finite_mask = np.isfinite(wavelength_isolation_db)
    wavelength_isolation_db_mean = (
        float(np.mean(wavelength_isolation_db[finite_mask]))
        if np.any(finite_mask) else float('inf')
    )
    wavelength_isolation_db_per_mode = np.array([
        float(np.mean(wavelength_isolation_db[m][np.isfinite(wavelength_isolation_db[m])]))
        if np.any(np.isfinite(wavelength_isolation_db[m])) else float('inf')
        for m in range(M)
    ], dtype=np.float64)

    # ================================================================
    # 3. Target / All ROI
    #    对于输入模式 m：
    #    target_energy = Σ_l E[m,l,m] (所有波长上目标 ROI 的总能量)
    #    all_roi_energy = Σ_{j,l} E[m,l,j] (所有 ROI 的总能量)
    #    ratio = target / all
    # ================================================================
    target_all_roi_ratio = np.zeros(M, dtype=np.float64)
    for m in range(M):
        target_sum = sum(energy_matrix[m, l, m] for l in range(L))
        all_roi_sum = float(energy_matrix[m, :, :].sum())
        target_all_roi_ratio[m] = target_sum / max(all_roi_sum, eps)

    target_all_roi_ratio_mean = float(np.mean(target_all_roi_ratio))
    # 转 dB: 10*log10(target / (all - target))
    target_all_roi_db = np.zeros(M, dtype=np.float64)
    for m in range(M):
        r = target_all_roi_ratio[m]
        r_clip = np.clip(r, eps, 1.0 - eps)
        target_all_roi_db[m] = 10.0 * np.log10(r_clip / (1.0 - r_clip))

    # ================================================================
    # 4. Crosstalk 矩阵
    # ================================================================
    # (a) 模式串扰矩阵 (per wavelength): (L, M, M)
    #     crosstalk[l, m_in, m_roi] = E[m_in, l, m_roi] / Σ_j E[m_in, l, j]
    crosstalk_matrix_per_wl = np.zeros((L, M, M), dtype=np.float64)
    for l in range(L):
        for m in range(M):
            row_sum = energy_matrix[m, l, :].sum()
            if row_sum > eps:
                crosstalk_matrix_per_wl[l, m, :] = energy_matrix[m, l, :] / row_sum

    # (b) 波长串扰矩阵 (per mode): (M, L, L)
    #     crosstalk_wl[m, l_target, l_source] = E[m, l_source, m] / Σ_{l'} E[m, l', m]
    crosstalk_matrix_wl = np.zeros((M, L, L), dtype=np.float64)
    for m in range(M):
        wl_energies = np.array([energy_matrix[m, l, m] for l in range(L)], dtype=np.float64)
        total_wl = wl_energies.sum()
        if total_wl > eps:
            crosstalk_matrix_wl[m, :, :] = np.outer(
                np.ones(L), wl_energies / total_wl
            )
            # 实际上每行相同 → 改为对角占比
            for l in range(L):
                crosstalk_matrix_wl[m, l, :] = wl_energies / total_wl

    # ================================================================
    # 5. Insertion Loss
    #    IL = -10*log10( E_out_total / (E_in * L) )
    #    因为输入被复制到 L 个波长通道，总输入能量 = E_in * L
    # ================================================================
    output_total_per_mode = total_energy_output.sum(axis=1)  # (M,)
    il_ratio = output_total_per_mode / np.clip(input_energy * L, eps, None)
    insertion_loss_db = -10.0 * np.log10(np.clip(il_ratio, eps, None))
    insertion_loss_db_mean = float(np.mean(insertion_loss_db))

    # ================================================================
    # 6. SNR (per mode, per wavelength)
    #    SNR = E_target_roi / (E_total_plane - E_target_roi)
    # ================================================================
    snr_db = np.zeros((M, L), dtype=np.float64)
    for m in range(M):
        for l in range(L):
            signal = energy_matrix[m, l, m]
            total = total_energy_output[m, l]
            noise = max(total - signal, eps)
            snr_db[m, l] = 10.0 * np.log10(max(signal / noise, eps))

    snr_db_mean = float(np.mean(snr_db))

    # ================================================================
    # 7. Throughput
    # ================================================================
    throughput_per_mode = output_total_per_mode / np.clip(input_energy * L, eps, None)

    # ================================================================
    # 打印报告
    # ================================================================
    wls_nm = wavelengths_m * 1e9
    print("\n" + "=" * 80)
    print("📊 Multi-Wavelength Comprehensive Metrics")
    print("=" * 80)

    print(f"\n{'='*40} Mode Isolation {'='*40}")
    print(f"  Mean Mode Isolation: {mode_isolation_db_mean:.2f} dB")
    print(f"  Per-wavelength mean:")
    for l in range(L):
        print(f"    λ={wls_nm[l]:.1f} nm: {mode_isolation_db_per_wl[l]:.2f} dB")
    print(f"  Per-(mode, wavelength) matrix (dB):")
    header = "  Mode\\λ(nm) " + " ".join([f"{wls_nm[l]:>8.1f}" for l in range(L)])
    print(header)
    for m in range(M):
        row = f"  Mode {m+1:>3d}   " + " ".join([f"{mode_isolation_db[m,l]:>8.2f}" for l in range(L)])
        print(row)

    print(f"\n{'='*40} Wavelength Isolation {'='*40}")
    print(f"  Mean Wavelength Isolation: {wavelength_isolation_db_mean:.2f} dB")
    print(f"  Per-mode mean:")
    for m in range(M):
        print(f"    Mode {m+1}: {wavelength_isolation_db_per_mode[m]:.2f} dB")

    print(f"\n{'='*40} Target / All ROI {'='*40}")
    print(f"  Mean Target/All ROI ratio: {target_all_roi_ratio_mean:.4f} "
          f"({10*np.log10(max(target_all_roi_ratio_mean/(1-target_all_roi_ratio_mean+eps), eps)):.2f} dB)")
    for m in range(M):
        print(f"    Mode {m+1}: {target_all_roi_ratio[m]:.4f} ({target_all_roi_db[m]:.2f} dB)")

    print(f"\n{'='*40} Insertion Loss {'='*40}")
    print(f"  Mean IL: {insertion_loss_db_mean:.2f} dB")
    for m in range(M):
        print(f"    Mode {m+1}: {insertion_loss_db[m]:.2f} dB")

    print(f"\n{'='*40} SNR {'='*40}")
    print(f"  Mean SNR: {snr_db_mean:.2f} dB")

    print(f"\n{'='*40} Throughput {'='*40}")
    print(f"  Mean: {float(throughput_per_mode.mean()):.4f}")
    print("=" * 80 + "\n")

    return {
        # Mode isolation
        "mode_isolation_db": mode_isolation_db,
        "mode_isolation_db_mean": mode_isolation_db_mean,
        "mode_isolation_db_per_wl": mode_isolation_db_per_wl,
        # Wavelength isolation
        "wavelength_isolation_db": wavelength_isolation_db,
        "wavelength_isolation_db_mean": wavelength_isolation_db_mean,
        "wavelength_isolation_db_per_mode": wavelength_isolation_db_per_mode,
        # Target / All ROI
        "target_all_roi_ratio": target_all_roi_ratio,
        "target_all_roi_ratio_mean": target_all_roi_ratio_mean,
        "target_all_roi_db": target_all_roi_db,
        # Crosstalk
        "crosstalk_matrix_per_wl": crosstalk_matrix_per_wl,
        "crosstalk_matrix_wl": crosstalk_matrix_wl,
        # Insertion loss
        "insertion_loss_db": insertion_loss_db,
        "insertion_loss_db_mean": insertion_loss_db_mean,
        # SNR
        "snr_db": snr_db,
        "snr_db_mean": snr_db_mean,
        # Throughput
        "throughput_per_mode": throughput_per_mode,
        # Raw
        "energy_matrix": energy_matrix,
        "total_energy_output": total_energy_output,
        "input_energy": input_energy,
        "wavelengths_m": np.asarray(wavelengths_m, dtype=np.float64),
    }


# ============================================================
# 可视化：多波长 Metrics 热图 + 保存 .mat
# ============================================================
def plot_and_save_multiwl_metrics(
    metrics: Dict[str, Any],
    *,
    output_dir: Path,
    tag: str,
    num_modes: int,
    num_wavelengths: int,
) -> Dict[str, str]:
    """
    将 evaluate_multiwl_comprehensive_metrics 的结果可视化并保存。
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    wls_nm = metrics["wavelengths_m"] * 1e9
    M = num_modes
    L = num_wavelengths

    saved_paths = {}

    # ── 1. Mode Isolation Heatmap ──
    fig, ax = plt.subplots(figsize=(max(6, L * 1.5), max(4, M * 0.8)))
    im = ax.imshow(metrics["mode_isolation_db"], cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"{wls_nm[l]:.1f}" for l in range(L)], rotation=45)
    ax.set_yticks(range(M))
    ax.set_yticklabels([f"Mode {m+1}" for m in range(M)])
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Input Mode")
    ax.set_title(f"Mode Isolation (dB) | mean={metrics['mode_isolation_db_mean']:.2f} dB")
    for m in range(M):
        for l in range(L):
            val = metrics["mode_isolation_db"][m, l]
            color = "white" if abs(val) > 10 else "black"
            ax.text(l, m, f"{val:.1f}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    p = output_dir / f"mode_isolation_heatmap_{tag}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["mode_isolation_fig"] = str(p)

    # ── 2. Wavelength Isolation Heatmap ──
    wl_iso = metrics["wavelength_isolation_db"].copy()
    wl_iso_clipped = np.clip(wl_iso, -30, 30)  # clip inf for display

    fig, ax = plt.subplots(figsize=(max(6, L * 1.5), max(4, M * 0.8)))
    im = ax.imshow(wl_iso_clipped, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(L))
    ax.set_xticklabels([f"{wls_nm[l]:.1f}" for l in range(L)], rotation=45)
    ax.set_yticks(range(M))
    ax.set_yticklabels([f"Mode {m+1}" for m in range(M)])
    ax.set_xlabel("Target Wavelength (nm)")
    ax.set_ylabel("Input Mode")
    ax.set_title(f"Wavelength Isolation (dB) | mean={metrics['wavelength_isolation_db_mean']:.2f} dB")
    for m in range(M):
        for l in range(L):
            val = wl_iso[m, l]
            txt = f"{val:.1f}" if np.isfinite(val) else "∞"
            color = "white" if abs(wl_iso_clipped[m, l]) > 10 else "black"
            ax.text(l, m, txt, ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    p = output_dir / f"wavelength_isolation_heatmap_{tag}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["wavelength_isolation_fig"] = str(p)

    # ── 3. Mode Crosstalk Heatmaps (per wavelength) ──
    crosstalk_dir = output_dir / "crosstalk_heatmaps"
    crosstalk_dir.mkdir(parents=True, exist_ok=True)
    ct_mat = metrics["crosstalk_matrix_per_wl"]  # (L, M, M)

    for l in range(L):
        M_ct = ct_mat[l]
        M_db = 10.0 * np.log10(np.clip(M_ct, 1e-6, None))

        fig_ct, axes_ct = plt.subplots(1, 2, figsize=(11, 4.5))
        im0 = axes_ct[0].imshow(M_ct, cmap="viridis", vmin=0, vmax=1)
        axes_ct[0].set_title(f"Mode Crosstalk (linear)\nλ={wls_nm[l]:.1f} nm")
        axes_ct[0].set_xlabel("ROI mode"); axes_ct[0].set_ylabel("Input mode")
        fig_ct.colorbar(im0, ax=axes_ct[0], fraction=0.046, pad=0.04)
        for r in range(M):
            for c in range(M):
                axes_ct[0].text(c, r, f"{M_ct[r,c]:.2f}",
                    ha="center", va="center",
                    color="white" if M_ct[r,c] < 0.5 else "black", fontsize=8)

        im1 = axes_ct[1].imshow(M_db, cmap="magma", vmin=-30, vmax=0)
        axes_ct[1].set_title(f"Mode Crosstalk (dB)\nλ={wls_nm[l]:.1f} nm")
        axes_ct[1].set_xlabel("ROI mode"); axes_ct[1].set_ylabel("Input mode")
        fig_ct.colorbar(im1, ax=axes_ct[1], fraction=0.046, pad=0.04)
        for r in range(M):
            for c in range(M):
                axes_ct[1].text(c, r, f"{M_db[r,c]:.0f}",
                    ha="center", va="center",
                    color="white" if M_db[r,c] < -15 else "black", fontsize=8)

        fig_ct.tight_layout()
        ct_path = crosstalk_dir / f"crosstalk_wl{l:02d}_{wls_nm[l]:.1f}nm_{tag}.png"
        fig_ct.savefig(ct_path, dpi=300, bbox_inches="tight")
        plt.close(fig_ct)

    saved_paths["crosstalk_dir"] = str(crosstalk_dir)

    # ── 4. Bar chart: Target/All ROI ──
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(M)
    ax.bar(x, metrics["target_all_roi_ratio"], color="tab:blue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Mode {m+1}" for m in range(M)])
    ax.set_ylabel("Target / All ROI Ratio")
    ax.set_title(f"Target vs All ROI Energy Ratio | mean={metrics['target_all_roi_ratio_mean']:.4f}")
    ax.set_ylim(0, 1.05)
    ax.axhline(metrics["target_all_roi_ratio_mean"], color="red", linestyle="--",
               label=f"mean={metrics['target_all_roi_ratio_mean']:.4f}")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = output_dir / f"target_all_roi_bar_{tag}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["target_all_roi_fig"] = str(p)

    # ── 5. Summary radar / combined plot ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # IL per mode
    axes[0].bar(range(M), metrics["insertion_loss_db"], color="tab:green", alpha=0.8)
    axes[0].set_xticks(range(M))
    axes[0].set_xticklabels([f"M{m+1}" for m in range(M)])
    axes[0].set_ylabel("IL (dB)")
    axes[0].set_title(f"Insertion Loss | mean={metrics['insertion_loss_db_mean']:.2f} dB")
    axes[0].grid(axis="y", alpha=0.3)

    # SNR heatmap
    im_snr = axes[1].imshow(metrics["snr_db"], cmap="YlOrRd", aspect="auto")
    axes[1].set_xticks(range(L))
    axes[1].set_xticklabels([f"{wls_nm[l]:.0f}" for l in range(L)], fontsize=8)
    axes[1].set_yticks(range(M))
    axes[1].set_yticklabels([f"M{m+1}" for m in range(M)])
    axes[1].set_title(f"SNR (dB) | mean={metrics['snr_db_mean']:.2f} dB")
    fig.colorbar(im_snr, ax=axes[1], fraction=0.046, pad=0.04)

    # Throughput per mode
    axes[2].bar(range(M), metrics["throughput_per_mode"], color="tab:purple", alpha=0.8)
    axes[2].set_xticks(range(M))
    axes[2].set_xticklabels([f"M{m+1}" for m in range(M)])
    axes[2].set_ylabel("Throughput (E_out/E_in)")
    axes[2].set_title(f"Throughput | mean={float(metrics['throughput_per_mode'].mean()):.4f}")
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    p = output_dir / f"summary_metrics_{tag}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight")
    plt.close(fig)
    saved_paths["summary_fig"] = str(p)

    # ── 6. Save .mat ──
    mat_path = output_dir / f"multiwl_metrics_{tag}.mat"
    savemat(str(mat_path), {
        "wavelengths_nm": wls_nm,
        "wavelengths_m": metrics["wavelengths_m"],
        "num_modes": np.array([M], dtype=np.int32),
        "num_wavelengths": np.array([L], dtype=np.int32),
        # Mode isolation
        "mode_isolation_db": metrics["mode_isolation_db"],
        "mode_isolation_db_mean": np.array([metrics["mode_isolation_db_mean"]]),
        "mode_isolation_db_per_wl": metrics["mode_isolation_db_per_wl"],
        # Wavelength isolation
        "wavelength_isolation_db": np.where(
            np.isfinite(metrics["wavelength_isolation_db"]),
            metrics["wavelength_isolation_db"], 999.0
        ),
        "wavelength_isolation_db_mean": np.array([metrics["wavelength_isolation_db_mean"]]),
        "wavelength_isolation_db_per_mode": metrics["wavelength_isolation_db_per_mode"],
        # Target/All ROI
        "target_all_roi_ratio": metrics["target_all_roi_ratio"],
        "target_all_roi_ratio_mean": np.array([metrics["target_all_roi_ratio_mean"]]),
        "target_all_roi_db": metrics["target_all_roi_db"],
        # Crosstalk
        "crosstalk_matrix_per_wl": metrics["crosstalk_matrix_per_wl"],
        "crosstalk_matrix_wl": metrics["crosstalk_matrix_wl"],
        # IL
        "insertion_loss_db": metrics["insertion_loss_db"],
        "insertion_loss_db_mean": np.array([metrics["insertion_loss_db_mean"]]),
        # SNR
        "snr_db": metrics["snr_db"],
        "snr_db_mean": np.array([metrics["snr_db_mean"]]),
        # Throughput
        "throughput_per_mode": metrics["throughput_per_mode"],
        # Raw energy
        "energy_matrix": metrics["energy_matrix"],
        "total_energy_output": metrics["total_energy_output"],
        "input_energy": metrics["input_energy"],
    })
    saved_paths["mat_path"] = str(mat_path)

    print(f"✔ MultiWL metrics figures saved -> {output_dir}")
    print(f"✔ MultiWL metrics data (.mat) -> {mat_path}")

    return saved_paths
