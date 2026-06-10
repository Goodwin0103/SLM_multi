from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import torch
from scipy.io import savemat

from odnn_training_eval import forward_full_intensity, spot_energy_ratios_circle
from odnn_model import complex_crop, complex_pad, propagation
from odnn_processing import pad_field_to_layer
from datetime import datetime  # for save_detection_overlap_debug timestamp
from matplotlib.patches import Rectangle  # for save_detection_overlap_debug

# ============================================================
# Helper: 把 (H_slm, W_slm) 的 complex field 居中零填充到 out_size 画布
# 用于支持 D2NNModel 中 out_size > layer_size 的情形。
# 当 model 没有 out_size 字段或两者相等时，原样返回。
# ============================================================
def _embed_field_to_out_canvas(
    field_2d: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    """field_2d: (H, W) complex；返回 (out_size, out_size) complex。"""
    if not hasattr(model, "out_size") or not hasattr(model, "layer_size"):
        return field_2d
    out_size = int(model.out_size)
    layer_size = int(model.layer_size)
    if out_size == layer_size:
        return field_2d

    H, W = field_2d.shape[-2], field_2d.shape[-1]
    if H == out_size and W == out_size:
        return field_2d  # 已经是大画布
    if H != layer_size or W != layer_size:
        # 安全兜底：大小不符预期就居中填充到 out_size
        pass

    diff_h = out_size - H
    diff_w = out_size - W
    pt = diff_h // 2; pb = diff_h - pt
    pl = diff_w // 2; pr = diff_w - pl
    from odnn_model import complex_pad_asymm
    return complex_pad_asymm(field_2d, pt, pb, pl, pr)


def _embed_field_b1hw_to_out_canvas(
    field_b1hw: torch.Tensor,
    model: torch.nn.Module,
) -> torch.Tensor:
    """同上但接受 (B,1,H,W)。"""
    if not hasattr(model, "out_size") or not hasattr(model, "layer_size"):
        return field_b1hw
    out_size = int(model.out_size)
    layer_size = int(model.layer_size)
    if out_size == layer_size:
        return field_b1hw

    plane = field_b1hw.squeeze(1).squeeze(0)   # 2D
    plane = _embed_field_to_out_canvas(plane, model)
    return plane.unsqueeze(0).unsqueeze(0)

def _overlay_detector_masks(
    ax,
    evaluation_regions: Sequence[Tuple[int, int, int, int]] | None,
    detect_radius: int | None,
    *,
    color: str = "cyan",
    linewidth: float = 1.2,
) -> None:
    """
    Draw circular detector ROI boundaries on top of an axes.
    """
    if not evaluation_regions or detect_radius is None:
        return

    r_draw = float(detect_radius)
    # For consistency with evaluate_spot_metrics, the passed-in detect_radius is often detectsize,
    # and energy is integrated with radius = detectsize/2. Draw using the effective energy radius.
    r_draw = r_draw / 2.0

    for (x0, x1, y0, y1) in evaluation_regions:
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        circle = Circle(
            (cx, cy),
            radius=r_draw,
            fill=False,
            color=color,
            linewidth=linewidth,
        )
        circle.set_clip_on(True)
        ax.add_patch(circle)

def plot_reconstruction_vs_input(
    image_test_data: torch.Tensor,
    reconstructed_fields: Sequence[np.ndarray],
    *,
    sample_idx: int = 0,
    model_idx: int = 0,
    save_path: str | Path = "results/plots/Reconstruction_vs_Input.png",
) -> None:
    """
    Compare original complex input with reconstructed field for a selected sample.
    """
    x_true = image_test_data[sample_idx, 0].detach().cpu().numpy()
    x_recon = reconstructed_fields[model_idx][sample_idx]

    amp_true, amp_recon = np.abs(x_true), np.abs(x_recon)
    phs_true, phs_recon = np.angle(x_true), np.angle(x_recon)

    def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
        a = a.ravel()
        b = b.ravel()
        if a.size == 0 or b.size == 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    rmse_amp = float(np.sqrt(np.mean((amp_true - amp_recon) ** 2)))
    rmse_phs = float(np.sqrt(np.mean((phs_true - phs_recon) ** 2)))
    cc_amp = corrcoef(amp_true, amp_recon)
    cc_phs = corrcoef(phs_true, phs_recon)

    vmin_amp, vmax_amp = 0.0, max(1.0, float(amp_true.max()), float(amp_recon.max()))
    vmin_phs, vmax_phs = -np.pi, np.pi

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    im = axes[0, 0].imshow(amp_true, vmin=vmin_amp, vmax=vmax_amp)
    axes[0, 0].set_title("Input Amplitude")
    axes[0, 0].axis("off")
    fig.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im = axes[0, 1].imshow(amp_recon, vmin=vmin_amp, vmax=vmax_amp)
    axes[0, 1].set_title("Reconstructed Amplitude")
    axes[0, 1].axis("off")
    fig.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im = axes[0, 2].imshow(np.abs(amp_recon - amp_true))
    axes[0, 2].set_title("|Δ Amp|")
    axes[0, 2].axis("off")
    fig.colorbar(im, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im = axes[1, 0].imshow(phs_true, vmin=vmin_phs, vmax=vmax_phs)
    axes[1, 0].set_title("Input Phase")
    axes[1, 0].axis("off")
    fig.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(phs_recon, vmin=vmin_phs, vmax=vmax_phs)
    axes[1, 1].set_title("Reconstructed Phase")
    axes[1, 1].axis("off")
    fig.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im = axes[1, 2].imshow(np.angle(np.exp(1j * (phs_recon - phs_true))))
    axes[1, 2].set_title("Wrapped Phase Diff")
    axes[1, 2].axis("off")
    fig.colorbar(im, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Reconstruction vs Input | RMSE_amp={rmse_amp:.4e}, CC_amp={cc_amp:.3f}, "
        f"RMSE_phs={rmse_phs:.4e}, CC_phs={cc_phs:.3f}"
    )
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print("✔ Saved:", Path(save_path).resolve())

def save_superposition_visuals(
    label_map: torch.Tensor,
    predicted_map: torch.Tensor,
    amplitudes: np.ndarray,
    evaluation_regions,
    detect_radius: int,
    output_dir: Path,
    tag: str,
) -> dict:
    """
    Save standalone label/prediction heatmaps and a comparison figure with weight bars.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    label_np = label_map.cpu().numpy()
    pred_np = predicted_map.cpu().numpy()

    predicted_energy, _ = spot_energy_ratios_circle(
        pred_np,
        evaluation_regions,
        detect_radius,
        offset=(0, 0),
        eps=1e-12,
    )
    predicted_weights = predicted_energy.astype(np.float64)
    pred_norm = np.linalg.norm(predicted_weights)
    if pred_norm > 0:
        predicted_weights /= pred_norm

    label_weights = amplitudes.astype(np.float64)
    label_norm = np.linalg.norm(label_weights)
    if label_norm > 0:
        label_weights /= label_norm

    vmin = float(min(label_np.min(), pred_np.min()))
    vmax = float(max(label_np.max(), pred_np.max()))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(label_np.min())
        vmax = vmin + 1e-6

    simple_label_path = output_dir / f"debug_super_label_{tag}.png"
    plt.figure(figsize=(5, 4))
    plt.imshow(label_np, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("super_label_map (ideal output)")
    plt.savefig(simple_label_path, dpi=200)
    plt.close()

    simple_pred_path = output_dir / f"debug_super_pred_{tag}.png"
    plt.figure(figsize=(5, 4))
    plt.imshow(pred_np, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.title("model predicted intensity (super_output)")
    plt.savefig(simple_pred_path, dpi=200)
    plt.close()

    bar_path = output_dir / f"superposition_sys_vs_label_{tag}.png"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im_label = axes[0].imshow(label_np, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[0].set_title("super_label_map (ideal output)")
    axes[0].axis("off")
    fig.colorbar(im_label, ax=axes[0], fraction=0.046, pad=0.04)

    im_pred = axes[1].imshow(pred_np, cmap="viridis", vmin=vmin, vmax=vmax)
    axes[1].set_title("model predicted intensity")
    axes[1].axis("off")
    fig.colorbar(im_pred, ax=axes[1], fraction=0.046, pad=0.04)

    mode_indices = np.arange(len(label_weights))
    bar_width = 0.4
    axes[2].bar(
        mode_indices - bar_width / 2,
        label_weights,
        width=bar_width,
        label="Label amplitude",
    )
    axes[2].bar(
        mode_indices + bar_width / 2,
        predicted_weights,
        width=bar_width,
        label="Predicted weight",
    )
    axes[2].set_xticks(mode_indices)
    axes[2].set_xticklabels([f"Mode {int(i) + 1}" for i in mode_indices])
    axes[2].set_ylabel("Normalized weight")
    axes[2].set_title("Mode weight comparison")
    axes[2].legend()
    axes[2].grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(bar_path, dpi=200)
    plt.close(fig)

    return {
        "label_path": simple_label_path,
        "prediction_path": simple_pred_path,
        "comparison_path": bar_path,
        "predicted_weights": predicted_weights,
        "label_weights": label_weights,
    }

@torch.no_grad()
def plot_sys_vs_label_strict(
    model: torch.nn.Module,
    dataset,
    sample_idx: int,
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    detect_radius: int,
    save_path: str | Path,
    *,
    device: torch.device,
    use_big_canvas: bool = False,
    clip_pct: float = 99.5,
    mask_roi_for_scale: bool = True,
    show_signed: bool = True,
    sys_scale: str = "bg_pct",
    sys_pct: float = 99.5,
) -> None:
    """
    Compare raw system output with label target for a selected sample.
    """
    if isinstance(dataset, list):
        image_complex, label = dataset[sample_idx]
    else:
        image_complex = dataset.tensors[0][sample_idx]
        label = dataset.tensors[1][sample_idx]

    image_complex = image_complex.to(device, dtype=torch.complex64)
    label = label.to(device, dtype=torch.float32)

    intensity_crop_t, intensity_full_t = forward_full_intensity(model, image_complex.unsqueeze(0))
    system_output = (intensity_full_t if use_big_canvas else intensity_crop_t)[0].cpu().numpy()

    label_hw = label[0].cpu().numpy()
    if use_big_canvas:
        pad = int(model.propagation.pad_px)
        height, width = system_output.shape
        label_canvas = np.zeros((height, width), dtype=label_hw.dtype)
        h_label, w_label = label_hw.shape
        label_canvas[pad : pad + h_label, pad : pad + w_label] = label_hw
        label_use = label_canvas
    else:
        label_use = label_hw

    if sys_scale == "same_max":
        vmax_sys = float(max(system_output.max(), label_use.max(), 1e-12))
    elif sys_scale == "sys_pct":
        vmax_sys = float(np.percentile(system_output, sys_pct))
    elif sys_scale == "bg_pct":
        radius = int(detect_radius)
        height, width = system_output.shape
        pad = int(model.propagation.pad_px) if use_big_canvas else 0
        yy, xx = np.ogrid[:height, :width]
        union = np.zeros((height, width), dtype=bool)
        for x0, x1, y0, y1 in evaluation_regions:
            cx = int(round((x0 + x1) / 2.0 + pad))
            cy = int(round((y0 + y1) / 2.0 + pad))
            union |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        background = system_output[~union]
        if background.size:
            vmax_sys = float(np.percentile(background, sys_pct))
        else:
            vmax_sys = float(np.percentile(system_output, sys_pct))
    else:
        vmax_sys = float(max(system_output.max(), 1e-12))

    diff = system_output - label_use
    diff_abs = np.abs(diff)

    if mask_roi_for_scale:
        radius = int(detect_radius)
        height, width = system_output.shape
        pad = int(model.propagation.pad_px) if use_big_canvas else 0
        yy, xx = np.ogrid[:height, :width]
        union = np.zeros((height, width), dtype=bool)
        for x0, x1, y0, y1 in evaluation_regions:
            cx = int(round((x0 + x1) / 2.0 + pad))
            cy = int(round((y0 + y1) / 2.0 + pad))
            union |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        bg_abs = diff_abs[~union]
        bg_signed = np.abs(diff[~union])
        vmax_diff_abs = float(np.percentile(bg_abs, clip_pct)) if bg_abs.size else float(np.percentile(diff_abs, clip_pct))
        vmax_signed = (
            float(np.percentile(bg_signed, clip_pct))
            if bg_signed.size
            else float(np.percentile(np.abs(diff), clip_pct))
        )
    else:
        vmax_diff_abs = float(np.percentile(diff_abs, clip_pct))
        vmax_signed = float(np.percentile(np.abs(diff), clip_pct))

    vmax_diff_abs = max(vmax_diff_abs, 1e-12)
    vmax_signed = max(vmax_signed, 1e-12)

    columns = 3 if show_signed else 2
    fig, axes = plt.subplots(1, columns, figsize=(4.8 * columns, 4.0))

    im0 = axes[0].imshow(system_output, vmin=0, vmax=vmax_sys)
    axes[0].set_title("System Output (raw view)")
    axes[0].axis("off")

    im1 = axes[1].imshow(label_use, vmin=0, vmax=vmax_sys)
    axes[1].set_title("Label / Ideal Output")
    axes[1].axis("off")

    ims = [im0, im1]
    if show_signed:
        im2 = axes[2].imshow(np.clip(diff, -vmax_signed, vmax_signed), cmap="bwr", vmin=-vmax_signed, vmax=vmax_signed)
        axes[2].set_title(f"Signed Diff (bwr, ≤p{clip_pct} bg)")
        axes[2].axis("off")
        ims.append(im2)

    for im, ax in zip(ims, axes):
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print("✔ Saved:", Path(save_path).resolve())

def plot_sys_vs_label_strict_separate_scale(
    model: torch.nn.Module,
    dataset,
    sample_idx: int,
    evaluation_regions,
    detect_radius: int,
    save_path: str | Path,
    *,
    device: torch.device,
    use_big_canvas: bool = False,
    clip_pct: float = 99.5,
    mask_roi_for_scale: bool = True,
    show_signed: bool = True,
    sys_scale: str = "bg_pct",
    sys_pct: float = 99.5,
):
    """
    Same idea as plot_sys_vs_label_strict, but:
    - system_output and label_use each get their own vmax
    - diff still clipped symmetrically
    """
    # ==== copy of the data extraction ====
    if isinstance(dataset, list):
        image_complex, label = dataset[sample_idx]
    else:
        image_complex = dataset.tensors[0][sample_idx]
        label = dataset.tensors[1][sample_idx]

    image_complex = image_complex.to(device, dtype=torch.complex64)
    label = label.to(device, dtype=torch.float32)

    intensity_crop_t, intensity_full_t = forward_full_intensity(model, image_complex.unsqueeze(0))
    system_output = (intensity_full_t if use_big_canvas else intensity_crop_t)[0].detach().cpu().numpy()

    label_hw = label[0].detach().cpu().numpy()
    if use_big_canvas:
        pad = int(model.propagation.pad_px)
        height, width = system_output.shape
        label_canvas = np.zeros((height, width), dtype=label_hw.dtype)
        h_label, w_label = label_hw.shape
        label_canvas[pad : pad + h_label, pad : pad + w_label] = label_hw
        label_use = label_canvas
    else:
        label_use = label_hw

    # ==== scaling for system ====
    if sys_scale == "same_max":
        vmax_sys = float(max(system_output.max(), 1e-12))
    elif sys_scale == "sys_pct":
        vmax_sys = float(np.percentile(system_output, sys_pct))
    elif sys_scale == "bg_pct":
        radius = int(detect_radius)
        height, width = system_output.shape
        pad = int(model.propagation.pad_px) if use_big_canvas else 0
        yy, xx = np.ogrid[:height, :width]
        union = np.zeros((height, width), dtype=bool)
        for x0, x1, y0, y1 in evaluation_regions:
            cx = int(round((x0 + x1) / 2.0 + pad))
            cy = int(round((y0 + y1) / 2.0 + pad))
            union |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        background = system_output[~union]
        if background.size:
            vmax_sys = float(np.percentile(background, sys_pct))
        else:
            vmax_sys = float(np.percentile(system_output, sys_pct))
    else:
        vmax_sys = float(max(system_output.max(), 1e-12))

    vmax_sys = max(vmax_sys, 1e-12)

    # ==== scaling for label (its OWN vmax, not vmax_sys) ====
    vmax_label = float(np.percentile(label_use, 99.5))
    vmax_label = max(vmax_label, 1e-12)

    # ==== diff scaling ====
    diff = system_output - label_use
    diff_abs = np.abs(diff)

    if mask_roi_for_scale:
        radius = int(detect_radius)
        height, width = system_output.shape
        pad = int(model.propagation.pad_px) if use_big_canvas else 0
        yy, xx = np.ogrid[:height, :width]
        union = np.zeros((height, width), dtype=bool)
        for x0, x1, y0, y1 in evaluation_regions:
            cx = int(round((x0 + x1) / 2.0 + pad))
            cy = int(round((y0 + y1) / 2.0 + pad))
            union |= (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        bg_signed = np.abs(diff[~union])
        if bg_signed.size:
            vmax_signed = float(np.percentile(bg_signed, clip_pct))
        else:
            vmax_signed = float(np.percentile(np.abs(diff), clip_pct))
    else:
        vmax_signed = float(np.percentile(np.abs(diff), clip_pct))

    vmax_signed = max(vmax_signed, 1e-12)

    # ==== plotting ====
    columns = 3 if show_signed else 2
    fig, axes = plt.subplots(1, columns, figsize=(4.8 * columns, 4.0))

    im0 = axes[0].imshow(system_output, vmin=0, vmax=vmax_sys)
    axes[0].set_title("System Output (own scale)")
    axes[0].axis("off")

    im1 = axes[1].imshow(label_use, vmin=0, vmax=vmax_label)
    axes[1].set_title("Label / Ideal Output (own scale)")
    axes[1].axis("off")

    ims = [im0, im1]
    if show_signed:
        im2 = axes[2].imshow(
            np.clip(diff, -vmax_signed, vmax_signed),
            cmap="bwr",
            vmin=-vmax_signed,
            vmax=vmax_signed,
        )
        axes[2].set_title("Signed Diff")
        axes[2].axis("off")
        ims.append(im2)

    for im, ax in zip(ims, axes):
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

    print("✔ Saved (separate scale):", Path(save_path).resolve())

def plot_propagated_field_padded(
    field: torch.Tensor,
    *,
    z_start: float,
    z_end: float,
    z_step: float,
    pixel_size: float,
    wavelength: float,
    pad_px: int = 0,
    plot: bool = False,
    kmax: int = 12,
    ncols: int = 5,
    save_path: str | Path | None = None,
    mode: str = "intensity",
    dpi: int = 300,
    cmap: str = "turbo",
    add_colorbar: bool = True,
) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Propagate a complex field along z and optionally visualise sampled slices.
    """
    if not torch.is_complex(field):
        raise ValueError("field must be complex")

    device = field.device
    if z_step <= 0:
        raise ValueError("z_step must be > 0")

    steps = int(np.floor((z_end - z_start) / z_step)) + 1
    z_values = np.linspace(z_start, z_start + (steps - 1) * z_step, steps)

    height, width = field.shape[-2:]
    frames = []

    if pad_px and pad_px > 0:
        canvas = height + 2 * pad_px
        fx = torch.fft.fftshift(torch.fft.fftfreq(canvas, d=pixel_size)).to(device)
        fxx, fyy = torch.meshgrid(fx, fx, indexing="ij")
        argument = (2 * torch.pi) ** 2 * ((1.0 / wavelength) ** 2 - fxx**2 - fyy**2)
        kz = torch.where(argument >= 0, torch.sqrt(torch.abs(argument)), 1j * torch.sqrt(torch.abs(argument))).to(
            torch.complex64
        )

        padded = complex_pad(field, pad_px, pad_px)
        for z in z_values:
            spectrum = torch.fft.fftshift(torch.fft.fft2(padded.to(torch.complex64)))
            propagated = torch.fft.ifft2(torch.fft.ifftshift(spectrum * torch.exp(1j * kz * float(z))))
            cropped = complex_crop(propagated, height, width, pad_px, pad_px)
            frames.append(cropped.detach().cpu())
    else:
        canvas = width
        for z in z_values:
            propagated = propagation(field, float(z), wavelength, canvas, pixel_size, device)
            frames.append(propagated.detach().cpu())

    fields = torch.stack(frames, dim=0)

    if plot or save_path:
        total = fields.shape[0]
        count = min(total, int(kmax))
        indices = np.linspace(0, total - 1, count, dtype=int)
        show = fields[indices].numpy()
        show = np.abs(show) ** 2 if mode == "intensity" else np.abs(show)

        p99 = np.percentile(show, 99.0)
        if p99 > 0:
            show = np.clip(show / p99, 0, 1)

        columns = max(1, int(ncols))
        rows = (count + columns - 1) // columns
        fig, axes = plt.subplots(rows, columns, figsize=(2.2 * columns, 2.2 * rows))
        axes = np.array(axes).reshape(-1)

        for idx, sample_idx in enumerate(indices):
            axes[idx].imshow(show[idx], cmap=cmap, vmin=0, vmax=1)
            axes[idx].set_title(f"z={z_values[sample_idx] * 1e6:.0f} µm", fontsize=8)
            axes[idx].axis("off")
        for idx in range(count, len(axes)):
            axes[idx].axis("off")

        if add_colorbar and count > 0:
            fig.colorbar(
                axes[0].images[0],
                ax=axes[:count].tolist(),
                fraction=0.02,
                pad=0.02,
                label="Normalized " + ("Intensity" if mode == "intensity" else "Amplitude"),
            )

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=dpi)
            print("Saved figure ->", Path(save_path).resolve())
        plt.close(fig)

    return fields, z_values

def plot_amplitude_comparison_grid(
    image_test_data: torch.Tensor,
    reconstructed_images: np.ndarray,
    cc_values: np.ndarray,
    *,
    max_samples: int,
    save_path: str | Path,
    title: str,
) -> None:
    """
    Plot amplitude comparison between ground truth and reconstructed fields for several samples.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    num_available = min(
        max_samples,
        image_test_data.shape[0],
        reconstructed_images.shape[0],
    )
    if num_available == 0:
        return

    image_abs = torch.abs(image_test_data[:num_available]).cpu().numpy()
    recon_abs = np.abs(reconstructed_images[:num_available])
    cc_values = cc_values[:num_available]

    fig, axes = plt.subplots(num_available, 2, figsize=(8, 4 * num_available))
    axes = np.atleast_2d(axes)

    for idx in range(num_available):
        axes[idx, 0].imshow(image_abs[idx, 0], vmin=0, vmax=1)
        axes[idx, 0].set_title(f"Sample {idx + 1}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(recon_abs[idx], vmin=0, vmax=1)
        axes[idx, 1].set_title(f"reconstruct CC: {cc_values[idx]:.4f}")
        axes[idx, 1].axis("off")

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print("✔ Saved:", Path(save_path).resolve())

def visualize_model_slices(
    model: torch.nn.Module,
    phase_layers: Sequence[np.ndarray],
    input_field: torch.Tensor,
    *,
    output_dir: str | Path,
    sample_tag: str,
    z_input_to_first: float,
    z_layers: float,
    z_prop_plus: float,
    z_step: float,
    pixel_size: float,
    wavelength: float,
    kmax: int = 25,
    ncols: int = 5,
    cmap: str = "RdBu_r",
) -> Tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """
    Generate propagation slice plots for each layer and return the sampled stacks.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = input_field.device
    scans: dict[str, dict[str, np.ndarray]] = {}

    scan_input_stack, scan_input_z = plot_propagated_field_padded(
        input_field,
        z_start=0.0,
        z_end=z_input_to_first,
        z_step=z_step,
        pixel_size=pixel_size,
        wavelength=wavelength,
        pad_px=int(model.propagation.pad_px),
        kmax=kmax,
        ncols=ncols,
        save_path=output_dir / f"{sample_tag}_scan_input.png",
        cmap=cmap,
    )
    scans["scan_input"] = {
        "stack": scan_input_stack.detach().cpu().numpy(),
        "z": scan_input_z.copy(),
    }

    field = input_field
    if abs(z_input_to_first) > 0:
        first_layer = model.layers[0]
        pad = int(first_layer.pad_px)
        field = complex_pad(field, pad, pad)
        field = first_layer._propagate(field, first_layer.kz_pad, z_input_to_first)
        field = complex_crop(field, first_layer.units, first_layer.units, pad, pad)

    for idx, phase_np in enumerate(phase_layers):
        phase_tensor = torch.from_numpy(phase_np).to(device=device, dtype=torch.float32)
        field = field * torch.exp(1j * phase_tensor)

        scan_name = f"scan_layer{idx + 1}"
        scan_stack, scan_z = plot_propagated_field_padded(
            field,
            z_start=0.0,
            z_end=z_layers,
            z_step=z_step,
            pixel_size=pixel_size,
            wavelength=wavelength,
            pad_px=int(model.layers[idx].pad_px),
            kmax=kmax,
            ncols=ncols,
            save_path=output_dir / f"{sample_tag}_{scan_name}.png",
            cmap=cmap,
        )
        scans[scan_name] = {
            "stack": scan_stack.detach().cpu().numpy(),
            "z": scan_z.copy(),
        }

        layer = model.layers[idx]
        pad = int(layer.pad_px)
        field = complex_pad(field, pad, pad)
        field = layer._propagate(field, layer.kz_pad, z_layers)
        field = complex_crop(field, layer.units, layer.units, pad, pad)

    # ★ scan_to_camera 段在 out_size 大画布上做（model.propagation.units = out_size）
    field = _embed_field_to_out_canvas(field, model)

    scan_stack2, scan_z2 = plot_propagated_field_padded(
        field,
        z_start=0.0,
        z_end=z_prop_plus,
        z_step=z_step,
        pixel_size=pixel_size,
        wavelength=wavelength,
        pad_px=int(model.propagation.pad_px),
        kmax=kmax,
        ncols=ncols,
        save_path=output_dir / f"{sample_tag}_scan_to_camera.png",
        cmap=cmap,
    )
    scans["scan_to_camera"] = {
        "stack": scan_stack2.detach().cpu().numpy(),
        "z": scan_z2.copy(),
    }

    # ★ 最终相机面也用 out_size 网格
    prop = model.propagation
    pad_cam = int(prop.pad_px)
    cam_units = int(prop.units)         # = out_size
    field_cam = complex_pad(field, pad_cam, pad_cam)
    field_cam = prop._propagate(field_cam, prop.kz_pad, prop.z)
    field_cam = complex_crop(field_cam, cam_units, cam_units, pad_cam, pad_cam)

    return scans, field_cam.detach().cpu().numpy()

@torch.no_grad()
def capture_eigenmode_propagation(
    model: torch.nn.Module,
    eigenmode_field: torch.Tensor,
    *,
    mode_index: int,
    layer_size: int,
    z_input_to_first: float,
    z_layers: float,
    z_prop: float,
    pixel_size: float,
    wavelength: float,
    output_dir: str | Path,
    tag: str,
    fractions_between_layers: Sequence[Sequence[float]] | None = None,
    output_fractions: Sequence[float] | None = None,
) -> dict[str, object]:
    """
    Capture specific propagation snapshots for a given eigenmode input and save plots/data.

    NEW: also computes & plots remaining energy at each LAYER plane:
         input_plane / layer{k}_arrival / after_last_layer / output_field
    """
    device = next(model.parameters()).device
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    num_layers = len(getattr(model, "layers", []))
    if num_layers == 0:
        raise ValueError("Model does not define diffraction layers to inspect.")

    if fractions_between_layers is None:
        default_fractions: list[tuple[float, ...]] = []
        for idx in range(num_layers):
            if idx < num_layers - 1:
                default_fractions.append((1.0 / 3.0, 2.0 / 3.0))
            else:
                default_fractions.append(())
        fractions_between_layers = tuple(default_fractions)

    if output_fractions is None:
        output_fractions = (0.2, 0.4, 0.6, 0.8)

    mode_field = eigenmode_field.to(device=device, dtype=torch.complex64)
    padded_field = pad_field_to_layer(mode_field, layer_size)
    input_field = padded_field.unsqueeze(0).unsqueeze(0)

    records: list[dict[str, object]] = []

    def tensor_to_np(field_tensor: torch.Tensor) -> np.ndarray:
        field_cpu = field_tensor.detach().squeeze().to("cpu").numpy()
        return np.asarray(field_cpu, dtype=np.complex64)

    # ★ 新增：用于 stack 前的统一尺寸
    def pad_np_to(arr: np.ndarray, target: int) -> np.ndarray:
        H, W = arr.shape[-2], arr.shape[-1]
        if H == target and W == target:
            return arr
        diff_h = target - H; pt = diff_h // 2; pb = diff_h - pt
        diff_w = target - W; pl = diff_w // 2; pr = diff_w - pl
        return np.pad(arr, ((pt, pb), (pl, pr)), mode="constant", constant_values=0)

    def add_record(key: str, description: str, field_tensor: torch.Tensor, z_value: float) -> None:
        records.append(
            {
                "key": key,
                "description": description,
                "z": float(z_value),
                "field": tensor_to_np(field_tensor),
            }
        )

    def propagate_through_layer(
        field_before_layer: torch.Tensor,
        layer_module,
        fractions: Sequence[float],
        stage_label: str,
        current_z: float,
    ) -> tuple[torch.Tensor, float]:
        plane = field_before_layer.squeeze(0).squeeze(0)
        units = int(layer_module.units)
        pad_px = int(layer_module.pad_px)
        phase = torch.exp(1j * layer_module.phase.to(device=plane.device, dtype=torch.float32)).to(torch.complex64)

        if pad_px > 0:
            padded = complex_pad(plane, pad_px, pad_px)
            mask_canvas = torch.ones(units + 2 * pad_px, units + 2 * pad_px, dtype=torch.complex64, device=plane.device)
            mask_canvas[pad_px : pad_px + units, pad_px : pad_px + units] = phase
            masked_padded = padded * mask_canvas
            kz = layer_module.kz_pad
        else:
            masked_padded = plane * phase
            kz = layer_module.kz_base

        for idx, frac in enumerate(sorted(fractions), start=1):
            distance = float(layer_module.z) * float(frac)
            propagated = layer_module._propagate(masked_padded, kz, distance)
            if pad_px > 0:
                cropped = complex_crop(propagated, units, units, pad_px, pad_px)
            else:
                cropped = propagated
            add_record(
                key=f"{stage_label}_prop{idx}",
                description=f"{stage_label} propagation ({frac:.2f}·z)",
                field_tensor=cropped,
                z_value=current_z + distance,
            )

        propagated_full = layer_module._propagate(masked_padded, kz, float(layer_module.z))
        if pad_px > 0:
            next_plane = complex_crop(propagated_full, units, units, pad_px, pad_px)
        else:
            next_plane = propagated_full

        current_z += float(layer_module.z)
        next_field = next_plane.unsqueeze(0).unsqueeze(0)
        return next_field, current_z

    def propagate_module(
        field_input: torch.Tensor,
        prop_module,
        fractions: Sequence[float],
        stage_label: str,
        current_z: float,
    ) -> tuple[torch.Tensor, float]:
        plane = field_input.squeeze(0).squeeze(0)
        units = int(prop_module.units)
        pad_px = int(prop_module.pad_px)

        if pad_px > 0:
            padded = complex_pad(plane, pad_px, pad_px)
            kz = prop_module.kz_pad
        else:
            padded = plane
            kz = prop_module.kz_base

        for idx, frac in enumerate(sorted(fractions), start=1):
            distance = float(prop_module.z) * float(frac)
            propagated = prop_module._propagate(padded, kz, distance)
            if pad_px > 0:
                cropped = complex_crop(propagated, units, units, pad_px, pad_px)
            else:
                cropped = propagated
            add_record(
                key=f"{stage_label}_prop{idx}",
                description=f"{stage_label} propagation ({frac:.2f}·z)",
                field_tensor=cropped,
                z_value=current_z + distance,
            )

        propagated_full = prop_module._propagate(padded, kz, float(prop_module.z))
        if pad_px > 0:
            output_plane = complex_crop(propagated_full, units, units, pad_px, pad_px)
        else:
            output_plane = propagated_full

        current_z += float(prop_module.z)
        next_field = output_plane.unsqueeze(0).unsqueeze(0)
        return next_field, current_z

    # ----------------------------
    # forward sweep with snapshots
    # ----------------------------
    current_z = 0.0
    add_record(
        key="input_plane",
        description=f"Input eigenmode {mode_index + 1} (padded)",
        field_tensor=input_field,
        z_value=current_z,
    )

    field = model.pre_propagation(input_field)
    current_z += float(z_input_to_first)
    add_record(
        key="layer1_arrival",
        description="Arrival at layer 1 (before mask)",
        field_tensor=field,
        z_value=current_z,
    )

    for layer_idx, layer in enumerate(model.layers):
        fractions = fractions_between_layers[layer_idx] if layer_idx < len(fractions_between_layers) else ()
        stage_label = (
            f"L{layer_idx + 1}_to_L{layer_idx + 2}" if layer_idx < num_layers - 1
            else f"L{layer_idx + 1}_internal"
        )
        field, current_z = propagate_through_layer(field, layer, fractions, stage_label, current_z)
        arrival_label = (
            f"layer{layer_idx + 2}_arrival" if layer_idx < num_layers - 1 else "after_last_layer"
        )
        add_record(
            key=arrival_label,
            description=(
                f"Arrival at layer {layer_idx + 2}" if layer_idx < num_layers - 1 else "After last layer"
            ),
            field_tensor=field,
            z_value=current_z,
        )

    # ★★★ 关键：进入 out_size 大画布 ★★★
    field = _embed_field_b1hw_to_out_canvas(field, model)

    field, current_z = propagate_module(field, model.propagation, output_fractions, "layers_to_output", current_z)
    add_record(
        key="output_field",
        description="Output plane (before detector)",
        field_tensor=field,
        z_value=current_z,
    )

    output_intensity = model.regression(field).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

    # ----------------------------
    # overview figure (existing)
    # ----------------------------
    # ★ 把所有 record 统一 pad 到 out_size，再 stack
    target_size = int(getattr(model, "out_size", layer_size))
    fields_aligned = [pad_np_to(rec["field"], target_size) for rec in records]
    intensity_stack = (np.abs(np.stack(fields_aligned, axis=0)) ** 2).astype(np.float32)

    vmax = float(np.percentile(intensity_stack, 99.5))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax_candidate = float(np.max(intensity_stack)) if intensity_stack.size else 0.0
        vmax = vmax_candidate if vmax_candidate > 0 else 1.0

    num_records = len(records)
    ncols = 4
    nrows = (num_records + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.array(axes).reshape(-1)
    last_im = None

    for idx, rec in enumerate(records):
        img = pad_np_to(rec["field"], target_size)
        im = axes[idx].imshow(np.abs(img) ** 2, cmap="inferno", vmin=0, vmax=vmax)   # ★ 改用 img
        axes[idx].set_title(f"{idx + 1}. {rec['description']}", fontsize=8)
        axes[idx].axis("off")
        last_im = im
        
    for idx in range(num_records, len(axes)):
        axes[idx].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes[:num_records], fraction=0.025, pad=0.02)

    fig.suptitle(f"Propagation snapshots | mode {mode_index + 1}")
    fig_path = output_dir / f"propagation_mode{mode_index + 1}_{tag}.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    # ============================================================
    # NEW: energy at each LAYER plane
    # Pick: input_plane / layer{k}_arrival / after_last_layer / output_field
    # ============================================================
    wanted_keys = ["input_plane"] \
        + [f"layer{k+1}_arrival" for k in range(num_layers)] \
        + ["after_last_layer", "output_field"]

    plane_records: list[dict[str, object]] = []
    for k in wanted_keys:
        for r in records:
            if r["key"] == k:
                plane_records.append(r)
                break

    # 每个平面的总能量
    energy_planes = np.array(
        [float((np.abs(r["field"]) ** 2).sum()) for r in plane_records],
        dtype=np.float64,
    )                                                                # (Nplanes,)

    E0 = float(energy_planes[0]) if energy_planes.size > 0 else 0.0
    energy_planes_norm = energy_planes / (E0 if E0 > 0 else 1.0)     # (Nplanes,)

    plane_keys = [r["key"] for r in plane_records]
    plane_z_m = np.array([r["z"] for r in plane_records], dtype=np.float64)

    # 折线图
    fig_e, ax_e = plt.subplots(figsize=(max(7.0, 0.9 * len(plane_keys)), 4.5))
    xs = np.arange(len(plane_keys))
    ax_e.plot(xs, energy_planes_norm, marker="o",
              color="tab:blue", label=f"λ={wavelength*1e9:.1f} nm")
    ax_e.set_xticks(xs)
    ax_e.set_xticklabels(plane_keys, rotation=30, ha="right", fontsize=8)
    ax_e.set_ylabel("Energy (normalized to input)")
    ax_e.set_title(f"Energy at each layer plane | mode {mode_index+1}")
    ax_e.grid(True, alpha=0.3)
    ax_e.set_ylim(bottom=0.0)
    ax_e.legend(fontsize=9, loc="best")
    fig_e.tight_layout()
    energy_fig_path = output_dir / f"energy_at_layers_mode{mode_index + 1}_{tag}.png"
    fig_e.savefig(energy_fig_path, dpi=250, bbox_inches="tight")
    plt.close(fig_e)

    # 终端打印一份表格
    print(f"\n[Energy at each plane | mode {mode_index+1} | tag={tag}]")
    print(f"  {'plane':<20s} {'z(mm)':>8s} {'E_abs':>14s} {'E_rel(%)':>12s}")
    for i, k in enumerate(plane_keys):
        print(
            f"  {k:<20s} {plane_z_m[i]*1e3:>8.2f} "
            f"{energy_planes[i]:>14.4e} {energy_planes_norm[i]*100:>11.2f}"
        )

    # ----------------------------
    # MAT export (existing + new)
    # ----------------------------
    slice_names = np.array([rec["key"] for rec in records], dtype=object)
    slice_descriptions = np.array([rec["description"] for rec in records], dtype=object)
    z_positions = np.array([rec["z"] for rec in records], dtype=np.float64)
    field_stack = np.stack(fields_aligned, axis=0).astype(np.complex64)
    energy_trace = intensity_stack.reshape(intensity_stack.shape[0], -1).sum(axis=1).astype(np.float64)

    mat_path = output_dir / f"propagation_mode{mode_index + 1}_{tag}.mat"
    savemat(
        str(mat_path),
        {
            "fields": field_stack,
            "intensities": intensity_stack.astype(np.float32),
            "energies": energy_trace,
            "slice_names": slice_names,
            "slice_descriptions": slice_descriptions,
            "z_positions_m": z_positions,
            "output_intensity": output_intensity,
            "mode_index": np.array([mode_index + 1], dtype=np.int32),
            "layer_size": np.array([layer_size], dtype=np.int32),
            "pixel_size": np.array([pixel_size], dtype=np.float64),
            "wavelength": np.array([wavelength], dtype=np.float64),
            "z_input_to_first": np.array([z_input_to_first], dtype=np.float64),
            "z_layers": np.array([z_layers], dtype=np.float64),
            "z_prop": np.array([z_prop], dtype=np.float64),

            # ---- new: per-layer energy bookkeeping ----
            "plane_keys":          np.array(plane_keys, dtype=object),
            "plane_z_m":           plane_z_m,
            "energy_at_planes":      energy_planes,           # (Nplanes,)
            "energy_at_planes_norm": energy_planes_norm,      # (Nplanes,) normalized to input
        },
    )

    if was_training:
        model.train()

    return {
        "fig_path":         str(fig_path),
        "mat_path":         str(mat_path),
        "energy_fig_path":  str(energy_fig_path),
        "energies":         energy_trace,
        "z_positions":      z_positions,
        "plane_keys":       plane_keys,
        "plane_z_m":        plane_z_m,
        "energy_at_planes":      energy_planes,
        "energy_at_planes_norm": energy_planes_norm,
    }

@torch.no_grad()
def save_mode_triptych(
    model: torch.nn.Module,
    mode_index: int,
    eigenmode_field: torch.Tensor,
    label_field: torch.Tensor,
    *,
    layer_size: int,
    output_dir: str | Path,
    tag: str,
    evaluation_regions: Sequence[Tuple[int, int, int, int]] | None = None,
    detect_radius: int | None = None,
    show_mask_overlays: bool = False,
    overlay_on_input: bool = False,
) -> dict[str, str]:
    """
    Save a triptych (input, model output, label) for a specific eigenmode and export data to MAT.
    """
    device = next(model.parameters()).device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    eigenmode_field = eigenmode_field.to(device=device, dtype=torch.complex64)
    padded_field = pad_field_to_layer(eigenmode_field, layer_size)
    input_batch = padded_field.unsqueeze(0).unsqueeze(0)

    output_intensity = model(input_batch).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)
    input_intensity = torch.abs(padded_field).square().detach().cpu().numpy().astype(np.float32)
    label_np = label_field.detach().cpu().numpy().astype(np.float32)

    # ★ 大画布对齐：output 与 label 在 out_size 画布上，input 仍在 SLM 画布上 → 居中填充
    out_h, out_w = output_intensity.shape[-2], output_intensity.shape[-1]
    in_h, in_w   = input_intensity.shape[-2],  input_intensity.shape[-1]
    if (in_h, in_w) != (out_h, out_w):
        diff_h = out_h - in_h
        diff_w = out_w - in_w
        pt = diff_h // 2; pb = diff_h - pt
        pl = diff_w // 2; pr = diff_w - pl
        input_intensity = np.pad(
            input_intensity,
            ((pt, pb), (pl, pr)),
            mode="constant",
            constant_values=0.0,
        )

    # label 一般已经是 out_size；做一个对称的兜底
    lab_h, lab_w = label_np.shape[-2], label_np.shape[-1]
    if (lab_h, lab_w) != (out_h, out_w):
        diff_h = out_h - lab_h
        diff_w = out_w - lab_w
        pt = diff_h // 2; pb = diff_h - pt
        pl = diff_w // 2; pr = diff_w - pl
        label_np = np.pad(
            label_np,
            ((pt, pb), (pl, pr)),
            mode="constant",
            constant_values=0.0,
        )

    vmax = float(np.max([input_intensity.max(), output_intensity.max(), label_np.max(), 1e-8]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(input_intensity, cmap="inferno", vmin=0, vmax=vmax)
    axes[0].set_title(f"Mode {mode_index + 1} Input")
    axes[0].axis("off")
    if show_mask_overlays and overlay_on_input:
        _overlay_detector_masks(axes[0], evaluation_regions, detect_radius)

    im1 = axes[1].imshow(output_intensity, cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title("Model Output")
    axes[1].axis("off")
    if show_mask_overlays:
        _overlay_detector_masks(axes[1], evaluation_regions, detect_radius)

    im2 = axes[2].imshow(label_np, cmap="inferno", vmin=0, vmax=vmax)
    axes[2].set_title("Label")
    axes[2].axis("off")
    if show_mask_overlays:
        _overlay_detector_masks(axes[2], evaluation_regions, detect_radius)

    for ax, im in zip(axes, [im0, im1, im2]):
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_path = output_dir / f"mode{mode_index + 1}_{tag}.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    mat_path = output_dir / f"mode{mode_index + 1}_{tag}.mat"
    savemat(
        str(mat_path),
        {
            "mode_index":      np.array([mode_index + 1], dtype=np.int32),
            "input_field":     padded_field.detach().cpu().numpy().astype(np.complex64),  # SLM 网格
            "input_intensity": input_intensity,            # ★ 现在是 out_size 画布
            "model_output":    output_intensity,           # out_size 画布
            "label":           label_np,                   # out_size 画布
        },
    )


    if was_training:
        model.train()

    return {"fig_path": str(fig_path), "mat_path": str(mat_path)}

@torch.no_grad()
def save_superposition_triptych(
    input_field: torch.Tensor,
    output_intensity_map: torch.Tensor | np.ndarray,
    *,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    complex_weights: np.ndarray,
    label_map: torch.Tensor,
    evaluation_regions,
    detect_radius: int,
    output_dir: str | Path,
    tag: str,
    save_plot: bool = True,
) -> dict[str, str]:
    """
    Save input/output/bar comparison for a superposition sample and export supporting data to MAT.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare input intensity (complex field expected with shape [1, H, W] or [H, W])
    if input_field.ndim == 3:
        input_plane = input_field.squeeze(0)
    else:
        input_plane = input_field
    input_complex = input_plane.detach().cpu()
    input_intensity = torch.abs(input_complex).square().numpy().astype(np.float32)

    # Predicted output intensity map -> numpy
    if isinstance(output_intensity_map, torch.Tensor):
        output_np = output_intensity_map.detach().cpu().numpy().astype(np.float32)
    else:
        output_np = np.asarray(output_intensity_map, dtype=np.float32)

    # Label map (for completeness in MAT)
    label_np = label_map.detach().cpu().numpy().astype(np.float32)

    # Compute predicted weights from detector regions
    eff_radius = max(1, int(round(detect_radius / 2.0)))
    predicted_energy, _ = spot_energy_ratios_circle(
        output_np,
        evaluation_regions,
        eff_radius,
        offset=(0, 0),
        eps=1e-12,
    )
    predicted_energy = predicted_energy.astype(np.float64)
    predicted_amplitudes = np.sqrt(np.clip(predicted_energy, 0.0, None))
    amp_norm = np.linalg.norm(predicted_amplitudes)
    if amp_norm > 0:
        predicted_weights = predicted_amplitudes / amp_norm
    else:
        predicted_weights = predicted_amplitudes

    # Label amplitudes (默认label已经做了归一化，没做就把删掉注释)
    # amplitudes = np.asarray(amplitudes, dtype=np.float64)
    # label_weights = amplitudes / (np.linalg.norm(amplitudes) + 1e-12)
    label_weights = amplitudes
    label_norm_sq = float(np.sum(label_weights**2))
    pred_norm_sq = float(np.sum(predicted_weights**2))
    

    vmax_input = float(np.percentile(input_intensity, 99.0))
    vmax_input = vmax_input if np.isfinite(vmax_input) and vmax_input > 0 else float(input_intensity.max(initial=1.0))
    vmax_output = float(np.percentile(output_np, 99.0))
    vmax_output = vmax_output if np.isfinite(vmax_output) and vmax_output > 0 else float(output_np.max(initial=1.0))
    vmax_input = max(vmax_input, 1e-12)
    vmax_output = max(vmax_output, 1e-12)

    fig_path = output_dir / f"super_triptych_{tag}.png"
    if save_plot:
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        im0 = axes[0].imshow(input_intensity, cmap="inferno", vmin=0, vmax=vmax_input)
        axes[0].set_title("Superposition Input Intensity")
        axes[0].axis("off")

        im1 = axes[1].imshow(output_np, cmap="inferno", vmin=0, vmax=vmax_output)
        axes[1].set_title("Model Output Intensity")
        axes[1].axis("off")

        vmax_label = float(np.percentile(label_np, 99.0))
        vmax_label = vmax_label if np.isfinite(vmax_label) and vmax_label > 0 else float(label_np.max(initial=1.0))
        vmax_label = max(vmax_label, 1e-12)
        im2 = axes[2].imshow(label_np, cmap="inferno", vmin=0, vmax=vmax_label)
        axes[2].set_title("Label Intensity")
        axes[2].axis("off")

        mode_indices = np.arange(len(label_weights))
        bar_width = 0.4
        axes[3].bar(
            mode_indices - bar_width / 2,
            label_weights,
            width=bar_width,
            label="Label weight",
        )
        axes[3].bar(
            mode_indices + bar_width / 2,
            predicted_weights,
            width=bar_width,
            label="Predicted weight",
        )
        axes[3].set_xticks(mode_indices)
        axes[3].set_xticklabels([f"Mode {int(i) + 1}" for i in mode_indices])
        axes[3].set_ylabel("Weight")
        axes[3].set_ylim(0, max(float(label_weights.max()), float(predicted_weights.max()), 1e-3) * 1.15)
        axes[3].set_title("Mode Weight Comparison")
        axes[3].legend()
        axes[3].grid(axis="y", alpha=0.3)

        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.savefig(fig_path, dpi=300)
        plt.close(fig)
        fig_path_str = str(fig_path)
    else:
        fig_path_str = ""
        if fig_path.exists():
            fig_path.unlink()

    mat_path = output_dir / f"super_triptych_{tag}.mat"
    savemat(
        str(mat_path),
        {
            "input_field": input_complex.numpy().astype(np.complex64),
            "input_intensity": input_intensity,
            "output_intensity": output_np,
            "label_map": label_np,
            "label_weights": label_weights.astype(np.float64),
            "predicted_weights": predicted_weights.astype(np.float64),
            "predicted_energy": predicted_energy.astype(np.float64),
            "amplitudes": amplitudes.astype(np.float32),
            "phases": np.asarray(phases, dtype=np.float32),
            "complex_weights": np.asarray(complex_weights, dtype=np.complex64),
            "evaluation_regions": np.asarray(evaluation_regions, dtype=np.int32),
            "detect_radius": np.array([detect_radius], dtype=np.int32),
        },
    )

    return {"fig_path": fig_path_str, "mat_path": str(mat_path)}

def export_superposition_slices(
    model: torch.nn.Module,
    phase_masks_list: Sequence[Sequence[np.ndarray]],
    input_field: torch.Tensor,
    output_dir: Path,
    *,
    sample_tag: str,
    z_input_to_first: float,
    z_layers: float,
    z_prop: float,
    z_step: float,
    pixel_size: float,
    wavelength: float,
) -> None:
    """
    Export propagation slices for the provided complex field across all stored phase masks.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i_model, phase_masks in enumerate(phase_masks_list, start=1):
        model_dir = output_dir / f"m{i_model}"
        model_dir.mkdir(parents=True, exist_ok=True)
        _scans, camera_field = visualize_model_slices(
            model,
            phase_masks,
            input_field,
            output_dir=model_dir,
            sample_tag=f"{sample_tag}_m{i_model}",
            z_input_to_first=z_input_to_first,
            z_layers=z_layers,
            z_prop_plus=z_prop,
            z_step=z_step,
            pixel_size=pixel_size,
            wavelength=wavelength,
        )
        np.savez(model_dir / "camera_field_superposition.npz", camera_field=camera_field)
        print(f"Superposition slices saved -> {model_dir.resolve()}")

# =========================
# Multi-wavelength additions
# =========================

from typing import Dict, Any, List  # noqa: E402


def _make_kz_stack_multiwl(N: int, dx: float, wavelengths: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    kz: (L, N, N) complex64
    """
    wl = torch.tensor(wavelengths, dtype=torch.float32, device=device)  # (L,)
    fx = torch.fft.fftshift(torch.fft.fftfreq(N, d=dx)).to(device)
    fxx, fyy = torch.meshgrid(fx, fx, indexing="ij")  # (N,N)

    inv_lam2 = (1.0 / wl)[:, None, None] ** 2
    argument = (2 * torch.pi) ** 2 * (inv_lam2 - fxx[None] ** 2 - fyy[None] ** 2)

    tmp = torch.sqrt(torch.abs(argument))
    kz = torch.where(argument >= 0, tmp, 1j * tmp).to(torch.complex64)
    return kz


def _propagate_multiwl_kz(E_blhw: torch.Tensor, kz_lnn: torch.Tensor, z: float) -> torch.Tensor:
    """
    E_blhw: (B,L,N,N) complex
    kz_lnn: (L,N,N) complex
    """
    E_blhw = E_blhw.to(torch.complex64)
    C = torch.fft.fftshift(torch.fft.fft2(E_blhw), dim=(-2, -1))
    return torch.fft.ifft2(torch.fft.ifftshift(C * torch.exp(1j * kz_lnn[None] * float(z)), dim=(-2, -1)))


def _complex_pad_blhw(E_blhw: torch.Tensor, pad_px: int) -> torch.Tensor:
    """
    pad complex tensor (B,L,H,W) with zeros.
    """
    if pad_px <= 0:
        return E_blhw
    Er = torch.view_as_real(E_blhw)  # (B,L,H,W,2)
    Erp = torch.nn.functional.pad(Er, (0, 0, pad_px, pad_px, pad_px, pad_px), mode="constant", value=0)
    return torch.view_as_complex(Erp.contiguous())


def _complex_crop_blhw(Epad_blhw: torch.Tensor, H: int, W: int, pad_px: int) -> torch.Tensor:
    if pad_px <= 0:
        return Epad_blhw
    p = int(pad_px)
    return Epad_blhw[..., p : p + H, p : p + W].contiguous()


def _save_intensity_frames_multiwl(
    E_blhw: torch.Tensor,
    *,
    out_dir: Path,
    tag: str,
    frame_name: str,
    wavelengths: np.ndarray,
    dpi: int = 250,
    cmap: str = "inferno",
) -> None:
    """
    Save per-wavelength intensity images for a given complex field (B=1 expected).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    wls_nm = (np.asarray(wavelengths, dtype=np.float64) * 1e9)
    I = (E_blhw.abs() ** 2)[0].detach().cpu().numpy()  # (L,H,W)

    for li in range(I.shape[0]):
        fig, ax = plt.subplots(1, 1, figsize=(5.5, 5))
        im = ax.imshow(I[li], cmap=cmap)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(f"{tag} | {frame_name} | λ idx={li}, {wls_nm[li]:.1f} nm")
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_dir / f"{tag}_{frame_name}_l{li}_{wls_nm[li]:.1f}nm.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)
@torch.no_grad()
def visualize_model_slices_multiwl(
    model: torch.nn.Module,
    input_field: torch.Tensor,
    *,
    output_dir: str | Path,
    sample_tag: str,
    z_input_to_first: float,
    z_layers: float,
    z_prop_plus: float,
    z_step: float,
    pixel_size: float,
    wavelengths: np.ndarray,
    kmax: int = 25,
    cmap: str = "inferno",
) -> tuple[dict[str, dict[str, np.ndarray]], np.ndarray]:
    """
    MultiWL version of visualize_model_slices:
    - input_field: (H,W) or (1,H,W) complex
    - saves intensity slices for EACH wavelength at sampled z positions
    - returns scans (np stacks) + camera_field (complex) as numpy

    Notes:
    - Uses the SAME physics as D2NNModelMultiWL:
      pre_propagation -> [phase(mask scaled per λ) + propagate]xN -> propagation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    if input_field.ndim == 3:
        plane_hw = input_field.squeeze(0)
    else:
        plane_hw = input_field
    plane_hw = plane_hw.to(device=device, dtype=torch.complex64)

    # infer sizes
    H, W = int(plane_hw.shape[-2]), int(plane_hw.shape[-1])
    L = int(len(wavelengths))

    # pad px: follow your model's modules
    pad_px = int(getattr(getattr(model, "pre_propagation", None), "pad_px", 0))

    # prepare (B=1,L,H,W)
    field = plane_hw[None, None, ...].repeat(1, L, 1, 1).contiguous()

    # kz for padded canvas
    Np = H + 2 * pad_px
    kz_pad = _make_kz_stack_multiwl(Np, float(pixel_size), np.asarray(wavelengths, dtype=np.float32), device)

    scans: dict[str, dict[str, np.ndarray]] = {}

    def sample_segment(
        E_in: torch.Tensor,
        *,
        z_total: float,
        seg_name: str,
    ) -> torch.Tensor:
        # build sampled z list (cap by kmax for saving)
        if z_step <= 0:
            raise ValueError("z_step must be > 0")
        steps = int(np.floor(z_total / z_step)) + 1 if z_total > 0 else 1
        z_values = np.linspace(0.0, max(0.0, z_total), steps)

        # pad once, propagate many times
        Epad = _complex_pad_blhw(E_in, pad_px) if pad_px > 0 else E_in

        frames: list[np.ndarray] = []
        z_kept: list[float] = []

        # choose indices to save (up to kmax)
        total = len(z_values)
        keep = min(total, int(kmax))
        keep_idx = np.linspace(0, total - 1, keep, dtype=int) if total > 1 else np.array([0], dtype=int)

        for idx in keep_idx:
            z = float(z_values[idx])
            Etmp = _propagate_multiwl_kz(Epad, kz_pad, z)
            Eshow = _complex_crop_blhw(Etmp, H, W, pad_px) if pad_px > 0 else Etmp
            _save_intensity_frames_multiwl(
                Eshow,
                out_dir=output_dir,
                tag=sample_tag,
                frame_name=f"{seg_name}_z{z*1e6:.1f}um",
                wavelengths=wavelengths,
                cmap=cmap,
            )
            frames.append(Eshow.detach().cpu().numpy().astype(np.complex64))  # (1,L,H,W)
            z_kept.append(z)

        # store stacks
        stack = np.stack(frames, axis=0)  # (K,1,L,H,W)
        scans[seg_name] = {
            "stack": stack,
            "z": np.asarray(z_kept, dtype=np.float64),
        }

        # propagate to the END of segment
        Epad_out = _propagate_multiwl_kz(Epad, kz_pad, float(z_total))
        E_out = _complex_crop_blhw(Epad_out, H, W, pad_px) if pad_px > 0 else Epad_out
        return E_out

    # 1) input -> first layer
    field = sample_segment(field, z_total=float(z_input_to_first), seg_name="scan_input_to_L1")

    # 2) each diffraction layer: apply scaled mask then propagate z_layers
    layers = getattr(model, "layers", None)
    if layers is None:
        raise ValueError("model.layers not found. This function expects D2NNModelMultiWL-like modules.")

    for li, layer in enumerate(layers):
        # build scaled phase (same as DiffractionLayerMultiWL.forward)
        # scale: (L,)
        lam0 = layer.lam0.to(device=device)
        wl_buf = layer.wavelengths.to(device=device)  # (L,)
        scale = (lam0 / wl_buf)  # (L,)

        phi0 = layer.phase.to(device=device, dtype=torch.float32)  # (H,W)
        phi = phi0[None, :, :] * scale[:, None, None]            # (L,H,W)
        phase_c = torch.exp(1j * phi).to(torch.complex64)         # (L,H,W)

        field = field * phase_c[None, ...]                        # (1,L,H,W)
        field = sample_segment(field, z_total=float(z_layers), seg_name=f"scan_after_mask_L{li+1:02d}")

    # 3) last -> camera
    field = sample_segment(field, z_total=float(z_prop_plus), seg_name="scan_to_camera")

    # return camera complex field (L,H,W) as numpy
    camera_field = field[0].detach().cpu().numpy().astype(np.complex64)  # (L,H,W)
    return scans, camera_field

@torch.no_grad()
def capture_eigenmode_propagation_multiwl(
    model: torch.nn.Module,
    eigenmode_field: torch.Tensor,
    *,
    mode_index: int,
    layer_size: int,
    z_input_to_first: float,
    z_layers: float,
    z_prop: float,
    pixel_size: float,
    wavelengths: np.ndarray,
    output_dir: str | Path,
    tag: str,
    base_wavelength_idx: int = 0,
    fractions_between_layers: Sequence[Sequence[float]] | None = None,
    output_fractions: Sequence[float] | None = None,
) -> dict[str, str]:
    """
    MultiWL version (dense snapshots supported):
    - records: input, arrival before L1, per-layer propagation snapshots, after last layer, output plane
    - figure: show base_wavelength_idx only (to avoid huge figure)
    - mat: save ALL wavelengths complex fields/intensities: fields(K,L,H,W)
    """
    device = next(model.parameters()).device
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- prepare input (H,W) -> (1,L,H,W)
    mode_field = eigenmode_field.to(device=device, dtype=torch.complex64)
    padded_field = pad_field_to_layer(mode_field, layer_size)  # (H,W)
    H, W = int(padded_field.shape[-2]), int(padded_field.shape[-1])
    L = int(len(wavelengths))

    field = padded_field[None, None, ...].repeat(1, L, 1, 1).contiguous()  # (1,L,H,W)

    # ---- defaults for snapshot fractions
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("model.layers not found or empty.")

    num_layers = len(layers)
    if fractions_between_layers is None:
        # mimic your single-wl defaults: internal points for each layer segment, last can be ()
        default: list[tuple[float, ...]] = []
        for li in range(num_layers):
            if li < num_layers - 1:
                default.append((1.0 / 3.0, 2.0 / 3.0))
            else:
                default.append((1.0 / 3.0, 2.0 / 3.0))  # last layer segment too (feel free to make () )
        fractions_between_layers = tuple(default)

    if output_fractions is None:
        output_fractions = (0.2, 0.4, 0.6, 0.8)

    # ---- pad & kz (use pre_propagation pad if exists)
    pad_px = int(getattr(getattr(model, "pre_propagation", None), "pad_px", 0))
    Np = H + 2 * pad_px
    kz_pad = _make_kz_stack_multiwl(Np, float(pixel_size), np.asarray(wavelengths, dtype=np.float32), device)

    records: list[dict[str, Any]] = []

    def add_record(key: str, desc: str, E_blhw: torch.Tensor, z_value: float) -> None:
        E_np = E_blhw[0].detach().cpu().numpy().astype(np.complex64)  # (L,H,W)
        records.append({"key": key, "description": desc, "z": float(z_value), "field": E_np})

    def propagate_full(E_in: torch.Tensor, z_total: float) -> torch.Tensor:
        Epad = _complex_pad_blhw(E_in, pad_px) if pad_px > 0 else E_in
        Epad_out = _propagate_multiwl_kz(Epad, kz_pad, float(z_total))
        return _complex_crop_blhw(Epad_out, H, W, pad_px) if pad_px > 0 else Epad_out

    def propagate_with_fractions(
        E_in: torch.Tensor,
        *,
        z_total: float,
        fractions: Sequence[float],
        stage_label: str,
        z_base: float,
    ) -> torch.Tensor:
        # use sorted unique fractions within (0,1)
        fr = [float(f) for f in fractions]
        fr = [f for f in fr if 0.0 < f < 1.0]
        fr = sorted(set(fr))

        if fr:
            Epad = _complex_pad_blhw(E_in, pad_px) if pad_px > 0 else E_in
            for idx, f in enumerate(fr, start=1):
                dist = float(z_total) * float(f)
                Etmp = _propagate_multiwl_kz(Epad, kz_pad, dist)
                Eshow = _complex_crop_blhw(Etmp, H, W, pad_px) if pad_px > 0 else Etmp
                add_record(
                    f"{stage_label}_prop{idx}",
                    f"{stage_label} propagation ({f:.2f}·z)",
                    Eshow,
                    z_base + dist,
                )

        return propagate_full(E_in, float(z_total))

    # ----------------------------
    # record input
    # ----------------------------
    z_cur = 0.0
    add_record("input", f"Input eigenmode {mode_index + 1} (padded)", field, z_cur)

    # input -> L1
    field = propagate_full(field, float(z_input_to_first))
    z_cur += float(z_input_to_first)
    add_record("arrive_L1", "Arrival at layer 1 (before mask)", field, z_cur)

    # ----------------------------
    # layers: (mask) + (prop with snapshots)
    # ----------------------------
    for li, layer in enumerate(layers):
        # apply scaled mask (same as your multiwl layer)
        lam0 = layer.lam0.to(device=device)
        wl_buf = layer.wavelengths.to(device=device)
        scale = (lam0 / wl_buf)  # (L,)

        phi0 = layer.phase.to(device=device, dtype=torch.float32)      # (H,W)
        phi = phi0[None, :, :] * scale[:, None, None]                 # (L,H,W)
        phase_c = torch.exp(1j * phi).to(torch.complex64)              # (L,H,W)

        field = field * phase_c[None, ...]
        add_record(f"after_mask_L{li+1}", f"After mask L{li+1}", field, z_cur)

        fracs = fractions_between_layers[li] if li < len(fractions_between_layers) else ()
        stage_label = f"L{li+1}_to_L{li+2}" if li < num_layers - 1 else f"L{li+1}_internal"

        z_base = z_cur
        field = propagate_with_fractions(
            field,
            z_total=float(z_layers),
            fractions=fracs,
            stage_label=stage_label,
            z_base=z_base,
        )
        z_cur += float(z_layers)

        add_record(
            f"arrive_L{li+2}" if li < num_layers - 1 else "after_last_layer",
            f"Arrival at layer {li+2}" if li < num_layers - 1 else "After last layer",
            field,
            z_cur,
        )

    # ----------------------------
    # last -> output (with snapshots)
    # ----------------------------
    z_base = z_cur
    field = propagate_with_fractions(
        field,
        z_total=float(z_prop),
        fractions=output_fractions,
        stage_label="layers_to_output",
        z_base=z_base,
    )
    z_cur += float(z_prop)
    add_record("output_plane", "Output plane (before detector)", field, z_cur)

    # detector intensity (full L)
    output_intensity = (field.abs() ** 2)[0].detach().cpu().numpy().astype(np.float32)  # (L,H,W)

    # --- overview figure using base_wavelength_idx only
    base_idx = int(np.clip(base_wavelength_idx, 0, L - 1))
    intensity_stack_base = np.stack([np.abs(r["field"][base_idx]) ** 2 for r in records], axis=0)  # (K,H,W)
    vmax = float(np.percentile(intensity_stack_base, 99.5))
    vmax = vmax if np.isfinite(vmax) and vmax > 0 else float(np.max(intensity_stack_base) if intensity_stack_base.size else 1.0)

    K = len(records)
    ncols = 4
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.0 * nrows))
    axes = np.array(axes).reshape(-1)

    last_im = None
    for i, rec in enumerate(records):
        im = axes[i].imshow(np.abs(rec["field"][base_idx]) ** 2, cmap="inferno", vmin=0, vmax=vmax)
        axes[i].set_title(f"{i+1}. {rec['description']}", fontsize=8)
        axes[i].axis("off")
        last_im = im
    for i in range(K, len(axes)):
        axes[i].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes[:K], fraction=0.025, pad=0.02)

    fig.suptitle(f"MultiWL propagation snapshots | mode {mode_index+1} | base λ idx={base_idx}")
    fig_path = output_dir / f"propagation_multiwl_mode{mode_index+1}_{tag}.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    # --- MAT export (store all L)
    slice_names = np.array([r["key"] for r in records], dtype=object)
    slice_desc = np.array([r["description"] for r in records], dtype=object)
    z_positions = np.array([r["z"] for r in records], dtype=np.float64)
    field_stack = np.stack([r["field"] for r in records], axis=0).astype(np.complex64)  # (K,L,H,W)
    intensity_stack_all = (np.abs(field_stack) ** 2).astype(np.float32)                  # (K,L,H,W)
    energy_trace = intensity_stack_all.reshape(intensity_stack_all.shape[0], -1).sum(axis=1).astype(np.float64)

    mat_path = output_dir / f"propagation_multiwl_mode{mode_index+1}_{tag}.mat"
    savemat(
        str(mat_path),
        {
            "fields": field_stack,
            "intensities": intensity_stack_all,
            "energies": energy_trace,
            "slice_names": slice_names,
            "slice_descriptions": slice_desc,
            "z_positions_m": z_positions,
            "output_intensity": output_intensity,
            "mode_index": np.array([mode_index + 1], dtype=np.int32),
            "layer_size": np.array([layer_size], dtype=np.int32),
            "pixel_size": np.array([pixel_size], dtype=np.float64),
            "wavelengths_m": np.asarray(wavelengths, dtype=np.float64),
            "z_input_to_first": np.array([z_input_to_first], dtype=np.float64),
            "z_layers": np.array([z_layers], dtype=np.float64),
            "z_prop": np.array([z_prop], dtype=np.float64),
            "base_wavelength_idx": np.array([base_idx], dtype=np.int32),
        },
    )

    return {"fig_path": str(fig_path), "mat_path": str(mat_path)}


# ============================================================
# ★ NEW (#8): build_uniform_fractions —— 切片分数生成
# ============================================================
def build_uniform_fractions(count: int) -> tuple[float, ...]:
    """
    Generate `count` evenly spaced fractions in (0, 1), exclusive at both ends.
    Used by capture_eigenmode_propagation as fractions_between_layers / output_fractions.

    Examples:
        build_uniform_fractions(0) -> ()
        build_uniform_fractions(1) -> (0.5,)
        build_uniform_fractions(3) -> (0.25, 0.5, 0.75)
    """
    if count <= 0:
        return ()
    fractions = np.linspace(1.0 / (count + 1), count / (count + 1), count, dtype=float)
    return tuple(float(f) for f in fractions)


# ============================================================
# ★ NEW (#4): save_detection_overlap_debug —— 检测区布局诊断图
# ============================================================
def save_detection_overlap_debug(
    *,
    evaluation_regions: Sequence[Tuple[int, int, int, int]],
    detection_masks: np.ndarray | None,
    label_data: torch.Tensor | None,
    out_size: int,
    focus_radius: int,
    output_dir: str | Path,
    sample_idx: int = 0,
    tag: str | None = None,
) -> Path:
    """
    Render a side-by-side diagnostic of detector layout and coverage overlap.

    Args:
        evaluation_regions: list of (x0, x1, y0, y1) bounding boxes per detector.
        detection_masks: optional (N, H, W) float array of per-detector masks
            (used only in `mixed` label mode; pass None to use rectangular regions).
        label_data: optional (B, 1, H, W) tensor; one sample is shown as the background.
        out_size: square canvas side (pixels).
        focus_radius: radius (in pixels) of the dashed reference circle drawn at each region center.
        output_dir: where to save the PNG.
        sample_idx: which sample of label_data to display.
        tag: optional filename suffix; default = current timestamp.

    Returns:
        Path of the saved PNG.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- compute overlap map ----
    if detection_masks is not None:
        overlap_map = np.sum(detection_masks > 0.5, axis=0).astype(np.float32)
    else:
        overlap_map = np.zeros((out_size, out_size), dtype=np.float32)
        for (x0, x1, y0, y1) in evaluation_regions:
            overlap_map[y0:y1, x0:x1] += 1.0
    overlap_pixels = int(np.count_nonzero(overlap_map > 1.0 + 1e-6))
    max_overlap = float(overlap_map.max()) if overlap_map.size else 0.0

    # ---- sample to display behind the detector overlay ----
    label_sample_np = None
    if isinstance(label_data, torch.Tensor) and label_data.shape[0] > 0:
        idx = min(max(0, int(sample_idx)), label_data.shape[0] - 1)
        label_sample_np = label_data[idx, 0].detach().cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ---- left panel: detector layout over (optional) label ----
    if label_sample_np is not None:
        im0 = axes[0].imshow(label_sample_np, cmap="inferno")
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
        axes[0].set_title(f"Label sample #{idx + 1} with detectors")
    else:
        axes[0].imshow(np.zeros((out_size, out_size), dtype=np.float32), cmap="Greys")
        axes[0].set_title("Detector layout (no label sample)")
    axes[0].set_axis_off()

    if detection_masks is not None:
        for i, mask in enumerate(detection_masks):
            color = plt.cm.tab20(i % 20)
            axes[0].contour(mask, levels=[0.5], colors=[color], linewidths=1.0)
            ys, xs = np.where(mask > 0.5)
            if xs.size > 0:
                axes[0].text(
                    float(xs.mean()), float(ys.mean()),
                    f"M{i + 1}",
                    color=color, fontsize=8, weight="bold",
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                              alpha=0.4, edgecolor="none"),
                )
    else:
        for i, (x0, x1, y0, y1) in enumerate(evaluation_regions):
            color = plt.cm.tab20(i % 20)
            axes[0].add_patch(Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=1.0, edgecolor=color, facecolor="none",
            ))
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            axes[0].add_patch(Circle(
                (cx, cy), radius=focus_radius,
                linewidth=1.0, edgecolor=color, linestyle="--", fill=False,
            ))
            axes[0].text(
                x0 + 1, y0 + 4,
                f"M{i + 1}",
                color=color, fontsize=8, weight="bold",
                ha="left", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black",
                          alpha=0.4, edgecolor="none"),
            )

    # ---- right panel: coverage overlap heatmap ----
    im1 = axes[1].imshow(overlap_map, cmap="viridis")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_title("Detector coverage count (overlap map)")
    axes[1].set_axis_off()

    if tag is None:
        tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"detection_overlap_{tag}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---- console feedback ----
    if overlap_pixels > 0:
        print(f"⚠ Detection regions overlap detected: "
              f"{overlap_pixels} pixels have >1 coverage (max {max_overlap:.1f}).")
    else:
        print("✔ No overlap detected between evaluation regions.")
    print(f"✔ Detection region debug plot saved -> {out_path}")
    return out_path

@torch.no_grad()
def compute_per_plane_energy(
    model: torch.nn.Module,
    eigenmode_field: torch.Tensor,
    *,
    layer_size: int,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """
    Lightweight per-plane energy tracer (no plots, no MAT).
    Mirrors the geometry inside capture_eigenmode_propagation but only
    records |E|^2 sum at:
      input_plane, layer{k}_arrival, after_last_layer, output_field.

    Returns
    -------
    plane_keys  : list[str]
    plane_z_m   : (Nplanes,) float64
    energy      : (Nplanes,) float64    # absolute |E|^2 sum
    energy_norm : (Nplanes,) float64    # / energy[0]
    """
    device = next(model.parameters()).device
    model.eval()

    mode_field   = eigenmode_field.to(device=device, dtype=torch.complex64)
    padded_field = pad_field_to_layer(mode_field, layer_size)
    field        = padded_field.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)

    keys: list[str]   = []
    zs:   list[float] = []
    Es:   list[float] = []

    def _record(key: str, f_b1hw: torch.Tensor, z_val: float) -> None:
        keys.append(key)
        zs.append(float(z_val))
        Es.append(float((f_b1hw.abs() ** 2).sum().item()))

    # input plane
    z_cur = 0.0
    _record("input_plane", field, z_cur)

    # input -> layer 1
    field = model.pre_propagation(field)
    z_cur += float(model.pre_propagation.z)
    _record("layer1_arrival", field, z_cur)

    # diffraction layers
    num_layers = len(model.layers)
    for li, layer in enumerate(model.layers):
        units  = int(layer.units)
        pad_px = int(layer.pad_px)
        plane  = field.squeeze(0).squeeze(0)
        phase_c = torch.exp(
            1j * layer.phase.to(device=device, dtype=torch.float32)
        ).to(torch.complex64)

        if pad_px > 0:
            padded = complex_pad(plane, pad_px, pad_px)
            mask_canvas = torch.ones(
                units + 2 * pad_px, units + 2 * pad_px,
                dtype=torch.complex64, device=device,
            )
            mask_canvas[pad_px:pad_px + units, pad_px:pad_px + units] = phase_c
            masked = padded * mask_canvas
            propagated = layer._propagate(masked, layer.kz_pad, float(layer.z))
            plane_next = complex_crop(propagated, units, units, pad_px, pad_px)
        else:
            masked = plane * phase_c
            plane_next = layer._propagate(masked, layer.kz_base, float(layer.z))

        field = plane_next.unsqueeze(0).unsqueeze(0)
        z_cur += float(layer.z)
        key = f"layer{li + 2}_arrival" if li < num_layers - 1 else "after_last_layer"
        _record(key, field, z_cur)

    # last layer -> detector (out_size canvas)
    field = _embed_field_b1hw_to_out_canvas(field, model)
    prop = model.propagation
    cam_units = int(prop.units)
    pad_cam   = int(prop.pad_px)
    plane = field.squeeze(0).squeeze(0)
    if pad_cam > 0:
        padded = complex_pad(plane, pad_cam, pad_cam)
        propagated = prop._propagate(padded, prop.kz_pad, float(prop.z))
        plane = complex_crop(propagated, cam_units, cam_units, pad_cam, pad_cam)
    else:
        plane = prop._propagate(plane, prop.kz_base, float(prop.z))
    field = plane.unsqueeze(0).unsqueeze(0)
    z_cur += float(prop.z)
    _record("output_field", field, z_cur)

    energy      = np.asarray(Es, dtype=np.float64)
    energy_norm = energy / (energy[0] if energy[0] > 0 else 1.0)
    return keys, np.asarray(zs, dtype=np.float64), energy, energy_norm
