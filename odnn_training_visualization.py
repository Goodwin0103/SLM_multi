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

    prop = model.propagation
    pad_cam = int(prop.pad_px)
    field_cam = complex_pad(field, pad_cam, pad_cam)
    field_cam = prop._propagate(field_cam, prop.kz_pad, prop.z)
    field_cam = complex_crop(field_cam, model.layers[0].units, model.layers[0].units, pad_cam, pad_cam)

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
) -> dict[str, str]:
    """
    Capture specific propagation snapshots for a given eigenmode input and save plots/data.
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
        stage_label = f"L{layer_idx + 1}_to_L{layer_idx + 2}" if layer_idx < num_layers - 1 else f"L{layer_idx + 1}_internal"
        field, current_z = propagate_through_layer(field, layer, fractions, stage_label, current_z)
        arrival_label = (
            f"layer{layer_idx + 2}_arrival" if layer_idx < num_layers - 1 else "after_last_layer"
        )
        add_record(
            key=arrival_label,
            description=f"Arrival at layer {layer_idx + 2}" if layer_idx < num_layers - 1 else "After last layer",
            field_tensor=field,
            z_value=current_z,
        )

    field, current_z = propagate_module(field, model.propagation, output_fractions, "layers_to_output", current_z)
    add_record(
        key="output_field",
        description="Output plane (before detector)",
        field_tensor=field,
        z_value=current_z,
    )

    output_intensity = model.regression(field).squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.float32)

    intensity_stack = np.stack([np.abs(rec["field"]) ** 2 for rec in records], axis=0)
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
        im = axes[idx].imshow(np.abs(rec["field"]) ** 2, cmap="inferno", vmin=0, vmax=vmax)
        axes[idx].set_title(f"{idx + 1}. {rec['description']}", fontsize=8)
        axes[idx].axis("off")
        last_im = im
    for idx in range(num_records, len(axes)):
        axes[idx].axis("off")

    if last_im is not None:
        fig.colorbar(last_im, ax=axes[:num_records], fraction=0.025, pad=0.02)

    fig.suptitle(f"Propagation snapshots | mode {mode_index + 1}")
    #plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = output_dir / f"propagation_mode{mode_index + 1}_{tag}.png"
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    slice_names = np.array([rec["key"] for rec in records], dtype=object)
    slice_descriptions = np.array([rec["description"] for rec in records], dtype=object)
    z_positions = np.array([rec["z"] for rec in records], dtype=np.float64)
    field_stack = np.stack([rec["field"] for rec in records], axis=0).astype(np.complex64)
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
        },
    )

    if was_training:
        model.train()

    return {
        "fig_path": str(fig_path),
        "mat_path": str(mat_path),
        "energies": energy_trace,
        "z_positions": z_positions,
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

    vmax = float(np.max([input_intensity.max(), output_intensity.max(), label_np.max(), 1e-8]))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(input_intensity, cmap="inferno", vmin=0, vmax=vmax)
    axes[0].set_title(f"Mode {mode_index + 1} Input")
    axes[0].axis("off")
    if show_mask_overlays:
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
            "mode_index": np.array([mode_index + 1], dtype=np.int32),
            "input_field": padded_field.detach().cpu().numpy().astype(np.complex64),
            "input_intensity": input_intensity,
            "model_output": output_intensity,
            "label": label_np,
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
    out_size: int | None = None,
    padding_ratio_out: float | None = None,
) -> dict[str, object]:
    """
    MultiWL propagation snapshots — matches model.forward() exactly.
    Uses model's own modules and registered buffers for bit-exact results.
    Supports out_size != layer_size.
    """
    from typing import Any
    from odnn_multiwl_model import complex_pad, complex_crop

    device = next(model.parameters()).device
    model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # infer out_size from model
    if out_size is None:
        out_size = int(getattr(model, 'out_size', layer_size))

    # prepare input
    mode_field = eigenmode_field.to(device=device, dtype=torch.complex64)
    padded_field = pad_field_to_layer(mode_field, layer_size)
    H, W = int(padded_field.shape[-2]), int(padded_field.shape[-1])
    L = int(len(wavelengths))

    field = padded_field[None, None, ...].repeat(1, L, 1, 1).contiguous()

    # layer info
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("model.layers not found or empty.")
    num_layers = len(layers)

    if fractions_between_layers is None:
        fractions_between_layers = tuple((1.0/3.0, 2.0/3.0) for _ in range(num_layers))
    if output_fractions is None:
        output_fractions = (0.2, 0.4, 0.6, 0.8)

    # records
    records: list[dict[str, Any]] = []

    def add_record(key: str, desc: str, E_blhw: torch.Tensor, z_value: float) -> None:
        E_np = E_blhw[0].detach().cpu().numpy().astype(np.complex64)
        records.append({"key": key, "description": desc, "z": float(z_value), "field": E_np})

    # ===================================================================
    # Helper: propagate using model's propagation module (bit-exact)
    # ===================================================================
    def propagate_exact(E_blhw: torch.Tensor, prop_module, z_dist: float) -> torch.Tensor:
        B, Lf, Hf, Wf = E_blhw.shape
        p = int(prop_module.pad_px)
        if p > 0:
            Ein = complex_pad(E_blhw, p, p)
            kz = prop_module.kz_pad
            Eout = prop_module._propagate(Ein, kz, z_dist)
            return complex_crop(Eout, Hf, Wf, p, p)
        else:
            kz = prop_module.kz_base
            return prop_module._propagate(E_blhw, kz, z_dist)

    # ===================================================================
    # Helper: apply diffraction layer EXACTLY as model does
    # ===================================================================
    def apply_layer_exact(E_blhw: torch.Tensor, layer_module) -> torch.Tensor:
        """Matches DiffractionLayerMultiWL.forward() exactly."""
        B, Lf, Hf, Wf = E_blhw.shape
        p = int(layer_module.pad_px)

        lam0 = layer_module.lam0.to(device)
        wl_buf = layer_module.wavelengths.to(device)
        scale = (lam0 / wl_buf)
        phi0 = layer_module.phase.to(device, dtype=torch.float32)
        phi = phi0[None, :, :] * scale[:, None, None]
        phase_c = torch.exp(1j * phi).to(torch.complex64)

        if p > 0:
            Ein = complex_pad(E_blhw, p, p)
            phase_big = torch.ones(
                Lf, Hf + 2*p, Wf + 2*p, dtype=torch.complex64, device=device
            )
            phase_big[:, p:p+Hf, p:p+Wf] = phase_c
            Ein = Ein * phase_big[None]
            Eout = layer_module._propagate(Ein, layer_module.kz_pad, float(layer_module.z))
            return complex_crop(Eout, Hf, Wf, p, p)
        else:
            Ein = E_blhw * phase_c[None]
            return layer_module._propagate(Ein, layer_module.kz_base, float(layer_module.z))

    def get_field_after_mask_only(E_blhw: torch.Tensor, layer_module) -> torch.Tensor:
        """Field after phase mask only (no propagation) — for visualization."""
        lam0 = layer_module.lam0.to(device)
        wl_buf = layer_module.wavelengths.to(device)
        scale = (lam0 / wl_buf)
        phi0 = layer_module.phase.to(device, dtype=torch.float32)
        phi = phi0[None, :, :] * scale[:, None, None]
        phase_c = torch.exp(1j * phi).to(torch.complex64)
        return E_blhw * phase_c[None]

    def get_intermediate_snapshots(E_blhw, layer_module, fractions, stage_label, z_base):
        """Record intermediate propagation snapshots (matching model's padded logic)."""
        B, Lf, Hf, Wf = E_blhw.shape
        p = int(layer_module.pad_px)
        z_total = float(layer_module.z)

        fr = sorted(set(f for f in fractions if 0.0 < f < 1.0))
        if not fr:
            return

        lam0 = layer_module.lam0.to(device)
        wl_buf = layer_module.wavelengths.to(device)
        scale = (lam0 / wl_buf)
        phi0 = layer_module.phase.to(device, dtype=torch.float32)
        phi = phi0[None, :, :] * scale[:, None, None]
        phase_c = torch.exp(1j * phi).to(torch.complex64)

        if p > 0:
            Ein = complex_pad(E_blhw, p, p)
            phase_big = torch.ones(
                Lf, Hf + 2*p, Wf + 2*p, dtype=torch.complex64, device=device
            )
            phase_big[:, p:p+Hf, p:p+Wf] = phase_c
            Ein_masked = Ein * phase_big[None]
            kz = layer_module.kz_pad

            for idx, f in enumerate(fr, start=1):
                dist = z_total * f
                Etmp = layer_module._propagate(Ein_masked, kz, dist)
                Eshow = complex_crop(Etmp, Hf, Wf, p, p)
                add_record(f"{stage_label}_prop{idx}",
                           f"{stage_label} propagation ({f:.2f}·z)",
                           Eshow, z_base + dist)
        else:
            Ein_masked = E_blhw * phase_c[None]
            kz = layer_module.kz_base
            for idx, f in enumerate(fr, start=1):
                dist = z_total * f
                Etmp = layer_module._propagate(Ein_masked, kz, dist)
                add_record(f"{stage_label}_prop{idx}",
                           f"{stage_label} propagation ({f:.2f}·z)",
                           Etmp, z_base + dist)

    def get_output_snapshots(E_blhw, prop_module, fractions, stage_label, z_base, Hf, Wf):
        """Record intermediate output propagation snapshots."""
        p = int(prop_module.pad_px)
        z_total = float(prop_module.z)
        fr = sorted(set(f for f in fractions if 0.0 < f < 1.0))
        if not fr:
            return

        if p > 0:
            Ein = complex_pad(E_blhw, p, p)
            kz = prop_module.kz_pad
            for idx, f in enumerate(fr, start=1):
                dist = z_total * f
                Etmp = prop_module._propagate(Ein, kz, dist)
                Eshow = complex_crop(Etmp, Hf, Wf, p, p)
                add_record(f"{stage_label}_prop{idx}",
                           f"{stage_label} propagation ({f:.2f}·z)",
                           Eshow, z_base + dist)
        else:
            kz = prop_module.kz_base
            for idx, f in enumerate(fr, start=1):
                dist = z_total * f
                Etmp = prop_module._propagate(E_blhw, kz, dist)
                add_record(f"{stage_label}_prop{idx}",
                           f"{stage_label} propagation ({f:.2f}·z)",
                           Etmp, z_base + dist)

    # ===================================================================
    # START
    # ===================================================================
    z_cur = 0.0
    add_record("input", f"Input eigenmode {mode_index + 1} (padded)", field, z_cur)

    # pre_propagation
    field = propagate_exact(field, model.pre_propagation, float(z_input_to_first))
    z_cur += float(z_input_to_first)
    add_record("arrive_L1", "Arrival at layer 1 (before mask)", field, z_cur)

    # ===================================================================
    # LAYERS
    # ===================================================================
    for li, layer in enumerate(layers):
        # Record after mask (visualization only)
        field_after_mask = get_field_after_mask_only(field, layer)
        add_record(f"after_mask_L{li+1}", f"After mask L{li+1}", field_after_mask, z_cur)

        # Intermediate snapshots (matching model's padded logic)
        fracs = fractions_between_layers[li] if li < len(fractions_between_layers) else ()
        stage_label = f"L{li+1}_to_L{li+2}" if li < num_layers - 1 else f"L{li+1}_internal"
        get_intermediate_snapshots(field, layer, fracs, stage_label, z_cur)

        # Apply layer exactly as model does (actual state update)
        field = apply_layer_exact(field, layer)
        z_cur += float(layer.z)

        arrival_key = f"arrive_L{li+2}" if li < num_layers - 1 else "after_last_layer"
        arrival_desc = f"Arrival at layer {li+2}" if li < num_layers - 1 else "After last layer"
        add_record(arrival_key, arrival_desc, field, z_cur)

    # ===================================================================
    # EMBED TO OUT
    # ===================================================================
    if out_size != layer_size:
        if hasattr(model, '_embed_to_out_canvas'):
            field = model._embed_to_out_canvas(field)
        else:
            B_f, L_f, H_f, W_f = field.shape
            diff_h = out_size - H_f
            diff_w = out_size - W_f
            if diff_h >= 0 and diff_w >= 0:
                pt = diff_h // 2; pb = diff_h - pt
                pl = diff_w // 2; pr = diff_w - pl
                Er = torch.view_as_real(field)
                Er_pad = torch.nn.functional.pad(Er, (0, 0, pl, pr, pt, pb), mode='constant', value=0)
                field = torch.view_as_complex(Er_pad.contiguous())
            else:
                crop_h = (-diff_h) // 2
                crop_w = (-diff_w) // 2
                field = field[:, :, crop_h:crop_h+out_size, crop_w:crop_w+out_size].contiguous()
        add_record("embed_to_out", f"Embedded to out_size={out_size}", field, z_cur)

    # ===================================================================
    # OUTPUT PROPAGATION
    # ===================================================================
    prop_module = model.propagation
    H_out, W_out = field.shape[-2], field.shape[-1]
    z_base = z_cur

    get_output_snapshots(field, prop_module, output_fractions, "layers_to_output", z_base, H_out, W_out)

    field = propagate_exact(field, prop_module, float(prop_module.z))
    z_cur += float(prop_module.z)
    add_record("output_plane", "Output plane (before detector)", field, z_cur)

    # Detector intensity
    output_intensity = (field.abs() ** 2)[0].detach().cpu().numpy().astype(np.float32)

    # ===================================================================
    # VERIFY against model.forward()
    # ===================================================================
    x_verify = padded_field[None, None, ...].repeat(1, L, 1, 1).contiguous().to(device)
    I_model = model(x_verify)
    I_model_np = I_model[0].detach().cpu().numpy().astype(np.float32)
    max_diff = float(np.abs(output_intensity - I_model_np).max())
    match_status = '✔ MATCH' if max_diff < 1e-4 else '⚠ MISMATCH'
    print(f"    [verify] max |I_slices - I_model| = {max_diff:.2e} ({match_status})")

    # ===================================================================
    # OVERVIEW FIGURE
    # ===================================================================
    base_idx = int(np.clip(base_wavelength_idx, 0, L - 1))
    K = len(records)

    panel_w = 3.0
    panel_h = 3.0
    fig, axes = plt.subplots(K, L, figsize=(panel_w * L + 1.0, panel_h * K), squeeze=False)

    last_im = None
    for i, rec in enumerate(records):
        I_rec = np.abs(rec["field"]) ** 2
        row_vmax = float(I_rec.max()) if I_rec.size else 1.0
        if (not np.isfinite(row_vmax)) or row_vmax <= 0:
            row_vmax = float(I_rec.max() if I_rec.size else 1.0)

        for li in range(L):
            ax = axes[i, li]
            I_li = I_rec[li]
            im = ax.imshow(I_li, cmap="inferno", vmin=0, vmax=row_vmax)
            wl_nm_val = float(wavelengths[li]) * 1e9
            tag_base = " (base)" if li == base_idx else ""
            ax.set_title(f"{i+1}. {rec['description']}\nλ={wl_nm_val:.1f} nm{tag_base}", fontsize=7)
            ax.axis("off")
            last_im = im

        if last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes[i, :].tolist(), fraction=0.025, pad=0.01)
            cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f"MultiWL propagation | mode {mode_index+1} | layer_size={layer_size}, out_size={out_size}",
        fontsize=12,
    )
    fig_path = output_dir / f"propagation_multiwl_mode{mode_index+1}_{tag}.png"
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ===================================================================
    # ENERGY PLOT
    # ===================================================================
    wanted_keys = ["input"] + [f"arrive_L{k+1}" for k in range(num_layers)] + ["after_last_layer"]
    if out_size != layer_size:
        wanted_keys.append("embed_to_out")
    wanted_keys.append("output_plane")

    plane_records: list[dict[str, Any]] = []
    for k in wanted_keys:
        for r in records:
            if r["key"] == k:
                plane_records.append(r)
                break

    energy_planes_per_wl = np.stack(
        [(np.abs(r["field"]) ** 2).reshape(L, -1).sum(axis=1) for r in plane_records], axis=0,
    ).astype(np.float64)
    energy_planes_total = energy_planes_per_wl.sum(axis=1)

    E0_wl = energy_planes_per_wl[0].copy()
    E0_total = float(energy_planes_total[0])
    E0_wl_safe = np.where(E0_wl > 0, E0_wl, 1.0)
    energy_planes_per_wl_norm = energy_planes_per_wl / E0_wl_safe[None, :]
    energy_planes_total_norm = energy_planes_total / (E0_total if E0_total > 0 else 1.0)

    plane_keys = [r["key"] for r in plane_records]
    plane_z_m = np.array([r["z"] for r in plane_records], dtype=np.float64)
    wls_nm = (np.asarray(wavelengths, dtype=np.float64) * 1e9)

    fig_e, ax_e = plt.subplots(figsize=(max(7.0, 0.9 * len(plane_keys)), 4.5))
    xs = np.arange(len(plane_keys))
    for li in range(L):
        ax_e.plot(xs, energy_planes_per_wl_norm[:, li], marker="o", label=f"λ={wls_nm[li]:.1f} nm")
    ax_e.plot(xs, energy_planes_total_norm, marker="s", linestyle="--", color="k", label="total")
    ax_e.set_xticks(xs)
    ax_e.set_xticklabels(plane_keys, rotation=30, ha="right", fontsize=8)
    ax_e.set_ylabel("Energy (normalized to input)")
    ax_e.set_title(f"Energy at each plane | mode {mode_index+1}")
    ax_e.grid(True, alpha=0.3)
    ax_e.set_ylim(bottom=0.0)
    ax_e.legend(fontsize=9, loc="best")
    fig_e.tight_layout()
    energy_fig_path = output_dir / f"energy_at_layers_mode{mode_index+1}_{tag}.png"
    fig_e.savefig(energy_fig_path, dpi=250, bbox_inches="tight")
    plt.close(fig_e)

    # ===================================================================
    # MAT EXPORT
    # ===================================================================
    slice_names = np.array([r["key"] for r in records], dtype=object)
    slice_desc = np.array([r["description"] for r in records], dtype=object)
    z_positions = np.array([r["z"] for r in records], dtype=np.float64)

    mat_path = output_dir / f"propagation_multiwl_mode{mode_index+1}_{tag}.mat"
    savemat(str(mat_path), {
        "output_intensity": output_intensity,
        "slice_names": slice_names,
        "slice_descriptions": slice_desc,
        "z_positions_m": z_positions,
        "mode_index": np.array([mode_index + 1], dtype=np.int32),
        "layer_size": np.array([layer_size], dtype=np.int32),
        "out_size": np.array([out_size], dtype=np.int32),
        "pixel_size": np.array([pixel_size], dtype=np.float64),
        "wavelengths_m": np.asarray(wavelengths, dtype=np.float64),
        "z_input_to_first": np.array([z_input_to_first], dtype=np.float64),
        "z_layers": np.array([z_layers], dtype=np.float64),
        "z_prop": np.array([z_prop], dtype=np.float64),
        "base_wavelength_idx": np.array([base_wavelength_idx], dtype=np.int32),
        "plane_keys": np.array(plane_keys, dtype=object),
        "plane_z_m": plane_z_m,
        "energy_planes_per_wl": energy_planes_per_wl,
        "energy_planes_total": energy_planes_total,
        "energy_planes_per_wl_norm": energy_planes_per_wl_norm,
        "energy_planes_total_norm": energy_planes_total_norm,
        "max_diff_vs_model": np.array([max_diff], dtype=np.float64),
    })

    return {
        "fig_path": str(fig_path),
        "mat_path": str(mat_path),
        "energy_fig_path": str(energy_fig_path),
        "plane_keys": plane_keys,
        "plane_z_m": plane_z_m,
        "energy_planes_per_wl_norm": energy_planes_per_wl_norm,
        "energy_planes_total_norm": energy_planes_total_norm,
        "max_diff_vs_model": max_diff,
    }

@torch.no_grad()
def save_mode_triptych_multiwl(
    model: torch.nn.Module,
    mode_index: int,
    eigenmode_field: torch.Tensor,
    label_field: torch.Tensor,
    *,
    layer_size: int,
    wavelengths: np.ndarray,
    output_dir: str | Path,
    tag: str,
    evaluation_regions: Sequence[Tuple[int, int, int, int]] | None = None,
    detect_radius: int | None = None,
    show_mask_overlays: bool = False,
    overlay_on_input: bool = False,          # 默认 Input 不画圈
    base_wavelength_idx: int = 0,            # 总览图里用哪一个 λ
    per_wavelength_figs: bool = True,        # 是否另外给每个 λ 单独存一张三联图
    per_wavelength_label: bool = False,      # 是否对每个 λ 的 label 也做缩放（默认共享同一 label）
) -> dict[str, object]:
    """
    Multi-wavelength triptych for one eigenmode.
    - Saves a 'grid' figure: rows = wavelengths, columns = [Input, Output, Label]
    - Optionally saves per-wavelength 3-panel PNG (same style as save_mode_triptych)
    - Exports a MAT file with per-λ input/output intensities and label.
    """
    device = next(model.parameters()).device
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    was_training = model.training
    model.eval()

    # ---- input: (H,W) -> padded -> (1,L,H,W)
    plane_hw = eigenmode_field.to(device=device, dtype=torch.complex64)
    padded_field = pad_field_to_layer(plane_hw, layer_size)       # (H,W)
    H, W = int(padded_field.shape[-2]), int(padded_field.shape[-1])
    L = int(len(wavelengths))

    input_batch = padded_field[None, None, ...].repeat(1, L, 1, 1).contiguous()  # (1,L,H,W)

    # ---- forward (assumes D2NNModelMultiWL that takes (1,L,H,W) complex and returns intensity)
    out = model(input_batch)                                       # (1,L,H,W) or (L,H,W)
    if out.ndim == 4:
        output_intensity = out[0].detach().cpu().numpy().astype(np.float32)  # (L,H,W)
    else:
        output_intensity = out.detach().cpu().numpy().astype(np.float32)      # (L,H,W)

    # per-λ input intensity (for eigenmode input, all λ share the same amp^2)
    input_intensity = (torch.abs(padded_field) ** 2).detach().cpu().numpy().astype(np.float32)  # (H,W)

    # label
    label_np = label_field.detach().cpu().numpy().astype(np.float32)                             # (H,W)

    wls_nm = np.asarray(wavelengths, dtype=np.float64) * 1e9
    base_idx = int(np.clip(base_wavelength_idx, 0, L - 1))

    # ---------- (A) GRID figure: rows = wavelengths, cols = [Input, Output, Label] ----------
    fig, axes = plt.subplots(L, 3, figsize=(12, 4 * L), squeeze=False)
    for li in range(L):
        vmax = float(max(
            input_intensity.max(),
            output_intensity[li].max(),
            label_np.max(),
            1e-8,
        ))

        im0 = axes[li, 0].imshow(input_intensity, cmap="inferno", vmin=0, vmax=vmax)
        axes[li, 0].set_title(f"Mode {mode_index+1} Input | λ={wls_nm[li]:.1f} nm")
        axes[li, 0].axis("off")
        if show_mask_overlays and overlay_on_input:
            _overlay_detector_masks(axes[li, 0], evaluation_regions, detect_radius)

        im1 = axes[li, 1].imshow(output_intensity[li], cmap="inferno", vmin=0, vmax=vmax)
        axes[li, 1].set_title(f"Model Output | λ={wls_nm[li]:.1f} nm")
        axes[li, 1].axis("off")
        if show_mask_overlays:
            _overlay_detector_masks(axes[li, 1], evaluation_regions, detect_radius)

        im2 = axes[li, 2].imshow(label_np, cmap="inferno", vmin=0, vmax=vmax)
        axes[li, 2].set_title("Label")
        axes[li, 2].axis("off")
        if show_mask_overlays:
            _overlay_detector_masks(axes[li, 2], evaluation_regions, detect_radius)

        for ax, im in zip(axes[li], [im0, im1, im2]):
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    grid_fig_path = output_dir / f"mode{mode_index+1}_{tag}_multiwl_grid.png"
    plt.savefig(grid_fig_path, dpi=300)
    plt.close(fig)

    # ---------- (B) Per-wavelength individual triptychs ----------
    per_wl_fig_paths: list[str] = []
    if per_wavelength_figs:
        per_wl_dir = output_dir / "per_wavelength"
        per_wl_dir.mkdir(parents=True, exist_ok=True)
        for li in range(L):
            vmax = float(max(input_intensity.max(),
                             output_intensity[li].max(),
                             label_np.max(), 1e-8))
            fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4))
            im0 = ax2[0].imshow(input_intensity, cmap="inferno", vmin=0, vmax=vmax)
            ax2[0].set_title(f"Mode {mode_index+1} Input | λ={wls_nm[li]:.1f} nm")
            ax2[0].axis("off")
            if show_mask_overlays and overlay_on_input:
                _overlay_detector_masks(ax2[0], evaluation_regions, detect_radius)

            im1 = ax2[1].imshow(output_intensity[li], cmap="inferno", vmin=0, vmax=vmax)
            ax2[1].set_title(f"Model Output | λ={wls_nm[li]:.1f} nm")
            ax2[1].axis("off")
            if show_mask_overlays:
                _overlay_detector_masks(ax2[1], evaluation_regions, detect_radius)

            im2 = ax2[2].imshow(label_np, cmap="inferno", vmin=0, vmax=vmax)
            ax2[2].set_title("Label")
            ax2[2].axis("off")
            if show_mask_overlays:
                _overlay_detector_masks(ax2[2], evaluation_regions, detect_radius)

            for ax, im in zip(ax2, [im0, im1, im2]):
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            p = per_wl_dir / f"mode{mode_index+1}_{tag}_l{li:02d}_{wls_nm[li]:.1f}nm.png"
            plt.savefig(p, dpi=300)
            plt.close(fig2)
            per_wl_fig_paths.append(str(p))

    # ---------- (C) MAT export ----------
    mat_path = output_dir / f"mode{mode_index+1}_{tag}_multiwl.mat"
    savemat(
        str(mat_path),
        {
            "mode_index": np.array([mode_index + 1], dtype=np.int32),
            "input_field": padded_field.detach().cpu().numpy().astype(np.complex64),  # (H,W)
            "input_intensity": input_intensity,                                        # (H,W)
            "model_output": output_intensity,                                          # (L,H,W)
            "label": label_np,                                                         # (H,W)
            "wavelengths_m": np.asarray(wavelengths, dtype=np.float64),
            "base_wavelength_idx": np.array([base_idx], dtype=np.int32),
            "layer_size": np.array([layer_size], dtype=np.int32),
        },
    )

    if was_training:
        model.train()

    return {
        "grid_fig_path": str(grid_fig_path),
        "per_wavelength_figs": per_wl_fig_paths,
        "mat_path": str(mat_path),
    }

def visualize_phase_masks(
    phase_masks: list[np.ndarray],
    out_dir: Path | str,
    base_name: str = "phase_mask",
    save_degree: bool = False,
    dpi: int = 300,
    cmap: str = "twilight",
    show_stats: bool = True,
) -> list[Path]:
    """
    为每层 phase mask 生成 PNG 可视化。
    
    Parameters
    ----------
    phase_masks : list[np.ndarray]
        每层的相位掩模列表
    out_dir : Path | str
        输出目录
    base_name : str
        文件名前缀
    save_degree : bool
        是否转换为角度显示
    dpi : int
        图像分辨率
    cmap : str
        色图名称（推荐: "twilight", "hsv", "twilight_shifted"）
    show_stats : bool
        是否显示统计信息
    
    Returns
    -------
    png_paths : list[Path]
        保存的 PNG 文件路径列表
    """
    import matplotlib.pyplot as plt
    
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    png_paths = []
    
    for layer_idx, mask in enumerate(phase_masks, start=1):
        # 转换单位
        if save_degree:
            mask_display = np.rad2deg(mask)
            unit = "degrees"
            vmin, vmax = 0, 360
        else:
            mask_display = mask
            unit = "radians"
            vmin, vmax = 0, 2 * np.pi
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # 显示相位掩模
        im = ax.imshow(mask_display, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax)
        
        # 色条
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(f'Phase ({unit})', fontsize=11)
        
        # 标题
        ax.set_title(
            f"Phase Mask - Layer {layer_idx}", 
            fontsize=13, 
            fontweight='bold',
            pad=15
        )
        ax.set_xlabel("X (pixels)", fontsize=10)
        ax.set_ylabel("Y (pixels)", fontsize=10)
        
        # 统计信息
        if show_stats:
            stats_text = (
                f"Shape: {mask.shape[0]}×{mask.shape[1]}\n"
                f"Min: {mask_display.min():.3f} {unit}\n"
                f"Max: {mask_display.max():.3f} {unit}\n"
                f"Mean: {mask_display.mean():.3f} {unit}\n"
                f"Std: {mask_display.std():.3f} {unit}"
            )
            ax.text(
                0.02, 0.98, stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.85, edgecolor='gray')
            )
        
        # 保存
        png_filename = f"{base_name}_layer{layer_idx}.png"
        png_path = out_dir / png_filename
        fig.tight_layout()
        fig.savefig(png_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)
        
        png_paths.append(png_path)
        print(f"✔ Saved phase mask visualization -> {png_path}")
    
    return png_paths
