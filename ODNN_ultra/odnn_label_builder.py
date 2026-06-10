"""Unified label / dataset builder for single- and multi-wavelength D2NN."""
from __future__ import annotations
import math
from typing import Optional

import numpy as np
import torch
from torch.utils.data import TensorDataset

from odnn_generate_label import (
    compute_label_centers,
    compose_labels_from_patterns,
    generate_detector_patterns,
)
from ODNN_functions import generate_complex_weights, generate_fields_ts
from odnn_processing import prepare_sample, pad_field_to_layer


# =======================================================================
# Multi-wavelength label generator (column-layout, non-overlapping ROIs)
# (kept identical to mainfor6_wl.py to preserve behavior)
# =======================================================================
def _generate_detector_patterns_multiwl(
    H: int, W: int,
    num_modes: int, num_wavelengths: int,
    radius: int,
    pattern_mode: str = "circle",
    margin_ratio: float = 0.1,
):
    total = num_modes * num_wavelengths
    num_rows, num_cols = num_modes, num_wavelengths

    margin_x = max(int(W * margin_ratio), radius + 5)
    margin_y = max(int(H * margin_ratio), radius + 5)
    xs = np.linspace(margin_x, W - 1 - margin_x, num_cols)
    ys = np.linspace(margin_y, H - 1 - margin_y, num_rows)

    centers = []
    for mode_idx in range(num_rows):
        for wl_idx in range(num_cols):
            cx = int(round(xs[wl_idx])); cy = int(round(ys[mode_idx]))
            centers.append((cy, cx))

    if pattern_mode != "circle":
        raise NotImplementedError(f"Unsupported pattern_mode: {pattern_mode}")
    patterns = np.zeros((H, W, total), dtype=np.float32)
    for idx, (cy, cx) in enumerate(centers):
        yy, xx = np.ogrid[:H, :W]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        patterns[:, :, idx] = mask.astype(np.float32)

    eval_regions = []
    for cy, cx in centers:
        eval_regions.append((max(0, int(cx - radius)),
                             min(W, int(cx + radius)),
                             max(0, int(cy - radius)),
                             min(H, int(cy + radius))))
    return patterns, eval_regions, centers


# =======================================================================
# Public API: labels + ROIs
# =======================================================================
def build_labels_and_regions(
    *,
    geom,
    num_modes: int,
    out_size: int,
    label_pattern_mode: str,
    eigenmode_detectsize: int,
    eigenmode_focus_radius: int,
    circle_focus_radius: int,
    circle_detectsize: int,
):
    """Single source of truth for label+ROI construction.

    Returns dict:
        MMF_Label_data       : torch.Tensor (H,W,K)  K=num_modes (single) or num_modes*L (multi)
        evaluation_regions   : list[(x0,x1,y0,y1)]  flat list of ROI tuples
        centers              : list[(cy,cx)] | None
        detectsize, focus_radius : int
        num_label_channels   : int
    """
    L = geom.L

    # ----------- Multi-wavelength branch -----------
    if geom.is_multiwavelength:
        radius = eigenmode_detectsize // 2 if label_pattern_mode == "eigenmode" else circle_focus_radius
        patterns_3d, eval_regions, centers = _generate_detector_patterns_multiwl(
            H=out_size, W=out_size,
            num_modes=num_modes, num_wavelengths=L,
            radius=radius, pattern_mode="circle",
        )
        return dict(
            MMF_Label_data     = torch.from_numpy(patterns_3d.astype(np.float32)),
            evaluation_regions = eval_regions,
            centers            = centers,
            focus_radius       = radius,
            detectsize         = eigenmode_detectsize if label_pattern_mode == "eigenmode" else circle_detectsize,
            num_label_channels = num_modes * L,
        )

    # ----------- Single-wavelength branch (verbatim from main.py) -----------
    if label_pattern_mode == "eigenmode":
        pattern_size = eigenmode_detectsize + (eigenmode_detectsize % 2 == 0)
        pattern_stack = generate_detector_patterns(
            pattern_size, pattern_size, num_modes, shape="circle"
        )
        layout_radius = math.ceil(pattern_size / 2)
        focus_radius  = eigenmode_focus_radius
        detectsize    = eigenmode_detectsize
    elif label_pattern_mode == "circle":
        pattern_size  = circle_focus_radius * 2 + 1
        pattern_stack = generate_detector_patterns(
            pattern_size, pattern_size, num_modes, shape="circle"
        )
        layout_radius = circle_focus_radius
        focus_radius  = circle_focus_radius
        detectsize    = circle_detectsize
    else:
        raise ValueError(f"Unknown label_pattern_mode: {label_pattern_mode}")

    centers, _, _ = compute_label_centers(out_size, out_size, num_modes, layout_radius)
    label_maps = [
        compose_labels_from_patterns(out_size, out_size, pattern_stack, centers,
                                     Index=i + 1, visualize=False)
        for i in range(num_modes)
    ]
    MMF_Label_data = torch.from_numpy(np.stack(label_maps, axis=2).astype(np.float32))

    eval_regions = []
    half = detectsize // 2
    for (cy, cx) in centers:
        eval_regions.append((max(0, cx - half), min(out_size, cx + half),
                             max(0, cy - half), min(out_size, cy + half)))
    return dict(
        MMF_Label_data     = MMF_Label_data,
        evaluation_regions = eval_regions,
        centers            = centers,
        focus_radius       = focus_radius,
        detectsize         = detectsize,
        num_label_channels = num_modes,
    )


# =======================================================================
# Public API: training datasets (eigenmode-mode-only)
# =======================================================================
def build_train_dataset(
    *,
    geom,
    num_modes: int,
    field_size: int,
    layer_size: int,
    out_size: int,
    mmf_modes: torch.Tensor,
    MMF_Label_data: torch.Tensor,
    phase_option: int,
    base_amplitudes: np.ndarray,
    base_phases: np.ndarray,
):
    """Returns:
       single-wl: TensorDataset
       multi-wl : list[TensorDataset]   (one per wavelength)
    """
    L = geom.L

    if phase_option == 4:
        n = num_modes
        amplitudes = base_amplitudes[:n]
        phases     = base_phases[:n]
    else:
        amplitudes, phases = base_amplitudes, base_phases
        n = amplitudes.shape[0]

    cw = (amplitudes * np.exp(1j * phases)).astype(np.complex64)
    cw_ts = torch.from_numpy(cw)
    image_data = generate_fields_ts(cw_ts, mmf_modes, n, num_modes, field_size).to(torch.complex64)

    # ----------- helper: pad image only, keep label as-is -----------
    def _pad_image_only(img_complex: torch.Tensor, slm_size: int) -> torch.Tensor:
        img_padded = pad_field_to_layer(img_complex, slm_size)
        if img_padded.ndim == 2:
            img_t = img_padded.unsqueeze(0)
        elif img_padded.ndim == 3:
            img_t = img_padded
        else:
            raise ValueError(f"Unexpected pad_field_to_layer output shape: {img_padded.shape}")
        return img_t.to(torch.complex64)

    # ----------- Multi-wavelength branch -----------
    if geom.is_multiwavelength:
        H, W, total = MMF_Label_data.shape
        assert total == num_modes * L

        imgs = torch.stack([_pad_image_only(image_data[i], layer_size) for i in range(n)], dim=0)
        amp_ts = torch.from_numpy(amplitudes.astype(np.float32))
        energy = amp_ts ** 2

        datasets_per_wl: list[TensorDataset] = []
        for wl_idx in range(L):
            label_indices = [k * L + wl_idx for k in range(num_modes)]
            wl_patterns = MMF_Label_data[:, :, label_indices]                       # (H, W, num_modes)
            label_img = torch.einsum("nm,hwm->nhw", energy, wl_patterns).unsqueeze(1).contiguous()
            datasets_per_wl.append(TensorDataset(imgs, label_img, amp_ts))
        return datasets_per_wl

    # ----------- Single-wavelength branch -----------
    label_data = torch.zeros([n, 1, out_size, out_size], dtype=torch.float32)
    energy = torch.from_numpy(amplitudes.astype(np.float32)) ** 2
    label_data[:, 0] = (energy[:, None, None, :] * MMF_Label_data.unsqueeze(0)).sum(dim=3)

    # ★ 只对 image pad；label 保持 (1, out_size, out_size)
    imgs   = torch.stack([_pad_image_only(image_data[i], layer_size) for i in range(n)], dim=0)
    labels = label_data.contiguous()
    return TensorDataset(imgs, labels)

