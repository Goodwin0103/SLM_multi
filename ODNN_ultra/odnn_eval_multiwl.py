"""
Multi-wavelength evaluation utilities for D2NN.

Provides:
    - evaluate_spot_metrics_multiwl
    - evaluate_target_wl_over_all_wl_roi_ratio
    - evaluate_snr_isolation_crosstalk_multiwl

All assume:
    model output shape : (B, L, H, W) intensity
    evaluation_regions : flat list of L*num_modes ROI tuples (x0,x1,y0,y1)
                         indexed as  region_idx = mode_k * L + wl_idx
"""
from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np
import torch


# ============================================================
# helpers
# ============================================================
def _make_circle_mask(h: int, w: int, r: float, device: torch.device) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing="ij",
    )
    cy = (h - 1) / 2.0
    cx = (w - 1) / 2.0
    mask = ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)
    return mask.to(torch.float32)


def _patch_circle_energy(
    src: torch.Tensor, x0: int, x1: int, y0: int, y1: int, radius: float
) -> torch.Tensor:
    """src: (B,H,W) -> energy in circle inside the (x0,x1,y0,y1) box, shape (B,)."""
    patch = src[:, y0:y1, x0:x1]
    hh, ww = patch.shape[-2], patch.shape[-1]
    cmask = _make_circle_mask(hh, ww, float(radius), device=src.device)
    return (patch * cmask.unsqueeze(0)).sum(dim=(-1, -2))


def _per_sample_corrcoef(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = np.sqrt((a0 * a0).sum() + eps) * np.sqrt((b0 * b0).sum() + eps)
    return float((a0 * b0).sum() / denom)


# ============================================================
# 1) per-wavelength amplitude metrics  (avg_amp_err, rel_err, cc_amp)
# ============================================================
@torch.no_grad()
def evaluate_spot_metrics_multiwl(
    model,
    loader,
    *,
    device: torch.device,
    evaluation_regions: List[Tuple[int, int, int, int]],
    detect_radius: float,
    wl_idx: int,
    L: int,
    num_modes: int,
) -> Dict[str, float | np.ndarray]:
    """
    For wavelength index wl_idx, compute per-sample amplitude metrics:
       - avg_amplitudes_diff  (mean |pred - true|)
       - avg_relative_amp_err (mean |pred - true| / |true|)
       - cc_recon_amp         (per-sample Pearson correlation, shape (N,))
       - amplitudes_diff      (raw differences (N, num_modes))

    Each sample's "true" amplitude is sqrt(amp**2 / sum(amp**2)).
    """
    model.eval()
    pred_list, true_list = [], []

    for batch in loader:
        if len(batch) == 3:
            images, _label_img, amp = batch
        else:
            images, _label_img = batch
            # If amp is missing, fall back to one-hot (eigenmode case)
            B = images.shape[0]
            amp = torch.eye(num_modes, dtype=torch.float32)[:B]

        images = images.to(device, dtype=torch.complex64, non_blocking=True)
        amp    = amp.to(device, dtype=torch.float32, non_blocking=True)
        if images.ndim == 3:
            images = images.unsqueeze(1)

        # broadcast (B,1,H,W) -> (B,L,H,W) if needed
        x = images if images.shape[1] == L else images.repeat(1, L, 1, 1).contiguous()
        I_blhw = model(x)                                           # (B,L,H,W)
        I_bhw  = I_blhw[:, wl_idx].to(torch.float32).contiguous()    # (B,H,W)

        # true amplitude fractions
        amp2 = amp ** 2
        true_frac = amp2 / (amp2.sum(dim=1, keepdim=True) + 1e-12)
        true_amp  = torch.sqrt(true_frac + 1e-12)
        true_list.append(true_amp.detach().cpu())

        # pred energies in this wl's ROIs
        regions_this_wl = [evaluation_regions[k * L + wl_idx] for k in range(num_modes)]
        E_modes = torch.zeros((I_bhw.shape[0], num_modes),
                              device=device, dtype=torch.float32)
        for mk, (x0, x1, y0, y1) in enumerate(regions_this_wl):
            E_modes[:, mk] = _patch_circle_energy(I_bhw, x0, x1, y0, y1, detect_radius)

        E_sum = E_modes.sum(dim=1, keepdim=True) + 1e-12
        pred_amp = torch.sqrt(E_modes / E_sum + 1e-12)
        pred_list.append(pred_amp.detach().cpu())

    pred = torch.cat(pred_list, dim=0).numpy()
    true = torch.cat(true_list, dim=0).numpy()
    diff = pred - true
    abs_diff = np.abs(diff)
    rel_err  = abs_diff / (np.abs(true) + 1e-12)
    cc = np.asarray(
        [_per_sample_corrcoef(pred[i], true[i]) for i in range(pred.shape[0])],
        dtype=np.float64,
    )

    return {
        "avg_amplitudes_diff":  float(abs_diff.mean()),
        "avg_relative_amp_err": float(rel_err.mean()),
        "cc_recon_amp":         cc,
        "amplitudes_diff":      diff,
    }


# ============================================================
# 2) Target-WL energy / All-WL ROI energy ratio
# ============================================================
@torch.no_grad()
def evaluate_target_wl_over_all_wl_roi_ratio(
    model,
    loader,
    *,
    device: torch.device,
    evaluation_regions: List[Tuple[int, int, int, int]],
    detect_radius: float,
    L: int,
    num_modes: int,
) -> Dict[str, float | np.ndarray]:
    """
    For each wavelength channel s (model output) and each target wavelength t,
    compute energy in t's ROIs.  Diagonal element / row-sum gives the
    "target-WL / all-WL" ratio (the higher, the better demux).
    """
    model.eval()
    ratio_list = []

    for batch in loader:
        if len(batch) == 3:
            images, _label_img, _amp = batch
        else:
            images, _label_img = batch

        images = images.to(device, dtype=torch.complex64, non_blocking=True)
        if images.ndim == 3:
            images = images.unsqueeze(1)
        x = images if images.shape[1] == L else images.repeat(1, L, 1, 1).contiguous()
        I_blhw = model(x).to(torch.float32)            # (B,L,H,W)
        B = I_blhw.shape[0]

        ratios = torch.zeros((B, L), device=device, dtype=torch.float32)
        for s in range(L):
            src = I_blhw[:, s]                          # (B,H,W) channel s
            E_per_t = torch.zeros((B, L), device=device, dtype=torch.float32)
            for t in range(L):
                t_regions = [evaluation_regions[mk * L + t] for mk in range(num_modes)]
                tot = torch.zeros((B,), device=device, dtype=torch.float32)
                for (x0, x1, y0, y1) in t_regions:
                    tot = tot + _patch_circle_energy(src, x0, x1, y0, y1, detect_radius)
                E_per_t[:, t] = tot
            denom = E_per_t.sum(dim=1) + 1e-12
            ratios[:, s] = E_per_t[:, s] / denom        # diagonal
        ratio_list.append(ratios.detach().cpu())

    ratio_all = torch.cat(ratio_list, dim=0).numpy()    # (N_total, L)
    return {
        "ratio_mean":   float(ratio_all.mean()),
        "ratio_per_wl": ratio_all.mean(axis=0),         # (L,)
        "ratio_all":    ratio_all,
    }


# ============================================================
# 3) Per-wavelength SNR / Isolation / Crosstalk
# ============================================================
@torch.no_grad()
def evaluate_snr_isolation_crosstalk_multiwl(
    model,
    loader,
    *,
    device: torch.device,
    evaluation_regions: List[Tuple[int, int, int, int]],
    detect_radius: float,
    wl_idx: int,
    L: int,
    num_modes: int,
) -> Dict[str, float | np.ndarray]:
    """
    Evaluate at wavelength wl_idx the standard mode-isolation/SNR triplet:
       - isolation_db_mean      : iso vs. same-WL leakage
       - isolation_db_mean_allroi : iso vs. ALL ROIs (same+cross WL)
       - snr_db_full            : signal in target ROI / noise outside any ROI
       - crosstalk_matrix       : (num_modes, num_modes) for same WL
    """
    model.eval()
    iso_same_list   = []
    iso_all_list    = []
    snr_list        = []
    cross_acc       = torch.zeros((num_modes, num_modes), device=device, dtype=torch.float32)
    cross_count     = 0

    same_wl_regions = [evaluation_regions[mk * L + wl_idx] for mk in range(num_modes)]

    for batch in loader:
        if len(batch) == 3:
            images, _label_img, amp = batch
        else:
            images, _label_img = batch
            B = images.shape[0]
            amp = torch.eye(num_modes, dtype=torch.float32)[:B]

        images = images.to(device, dtype=torch.complex64, non_blocking=True)
        amp    = amp.to(device, dtype=torch.float32, non_blocking=True)
        if images.ndim == 3:
            images = images.unsqueeze(1)
        x = images if images.shape[1] == L else images.repeat(1, L, 1, 1).contiguous()

        I_blhw = model(x).to(torch.float32)              # (B,L,H,W)
        I_bhw  = I_blhw[:, wl_idx]                        # (B,H,W)
        B, H, W = I_bhw.shape

        # which mode is the "target" for this sample (largest amp component)
        target_idx = amp.argmax(dim=1)                    # (B,)

        # Energies in same-WL ROIs (B, num_modes)
        E_same = torch.stack([
            _patch_circle_energy(I_bhw, x0, x1, y0, y1, detect_radius)
            for (x0, x1, y0, y1) in same_wl_regions
        ], dim=1)

        # Energies in ALL ROIs across all wavelengths (B, num_modes*L)
        E_all = torch.stack([
            _patch_circle_energy(I_bhw, x0, x1, y0, y1, detect_radius)
            for (x0, x1, y0, y1) in evaluation_regions
        ], dim=1)

        for b in range(B):
            tk = int(target_idx[b].item())

            # ---- same-WL isolation ----
            sig = float(E_same[b, tk].item())
            leak_same = float(E_same[b].sum().item()) - sig
            ratio_same = sig / max(leak_same, 1e-12)
            iso_same_list.append(10.0 * np.log10(max(ratio_same, 1e-12)))

            # ---- all-ROI isolation (signal vs leakage to other WL/mode ROIs) ----
            sig_idx_in_all = tk * L + wl_idx
            sig_all = float(E_all[b, sig_idx_in_all].item())
            leak_all = float(E_all[b].sum().item()) - sig_all
            ratio_all = sig_all / max(leak_all, 1e-12)
            iso_all_list.append(10.0 * np.log10(max(ratio_all, 1e-12)))

            # ---- SNR: signal vs noise outside any ROI ----
            mask_all_rois = torch.zeros((H, W), device=device, dtype=torch.float32)
            for (x0, x1, y0, y1) in evaluation_regions:
                hh, ww = (y1 - y0), (x1 - x0)
                cmask  = _make_circle_mask(hh, ww, float(detect_radius), device=device)
                mask_all_rois[y0:y1, x0:x1] = torch.maximum(
                    mask_all_rois[y0:y1, x0:x1], cmask
                )
            full_energy = float(I_bhw[b].sum().item())
            roi_energy  = float((I_bhw[b] * mask_all_rois).sum().item())
            noise = max(full_energy - roi_energy, 1e-12)
            snr_list.append(10.0 * np.log10(max(sig_all / noise, 1e-12)))

            # ---- crosstalk row (only same-WL block here) ----
            row = E_same[b].clone()
            row = row / (row.sum() + 1e-12)
            cross_acc[tk] += row
            cross_count += 1 if tk == 0 else 0   # accumulate count per mode separately below

    # crosstalk: average per row (rows that received data)
    # use a simpler accumulator: count per row
    # (above counter logic was off — recompute properly:)
    # Just normalize each row by its current sum.
    row_sums = cross_acc.sum(dim=1, keepdim=True) + 1e-12
    crosstalk_matrix = (cross_acc / row_sums).cpu().numpy()

    return {
        "isolation_db_mean":         float(np.mean(iso_same_list)) if iso_same_list else float("nan"),
        "isolation_db_mean_allroi":  float(np.mean(iso_all_list))  if iso_all_list  else float("nan"),
        "snr_db_full":               float(np.mean(snr_list))      if snr_list      else float("nan"),
        "crosstalk_matrix":          crosstalk_matrix,
    }
