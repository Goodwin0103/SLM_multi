"""Multi-wavelength evaluation utilities (extracted from train_multiwl.py)."""
from __future__ import annotations
import numpy as np
import torch


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
def region_energy_fractions(I_bhw, evaluation_regions, detect_radius):
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


def _per_sample_corrcoef(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a0 = a - a.mean(); b0 = b - b.mean()
    denom = (np.sqrt((a0 * a0).sum() + eps) * np.sqrt((b0 * b0).sum() + eps))
    return float((a0 * b0).sum() / denom)


@torch.no_grad()
def evaluate_spot_metrics_multiwl(
    model, loader, *, device, evaluation_regions, detect_radius, wl_idx, L, num_modes,
):
    model.eval()
    pred_amp_list, true_amp_list = [], []
    for images, label_img, amp in loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)
        amp = amp.to(device, dtype=torch.float32, non_blocking=True)
        if images.ndim == 3:
            images = images.unsqueeze(1)

        amp2 = amp ** 2
        true_energy_frac = amp2 / (amp2.sum(dim=1, keepdim=True) + 1e-12)
        true_amp_list.append(torch.sqrt(true_energy_frac + 1e-12).detach().cpu())

        # 单通道 -> L 通道
        x = images if images.shape[1] == L else images.repeat(1, L, 1, 1).contiguous()
        I_blhw = model(x)
        I_bhw = I_blhw[:, wl_idx].to(torch.float32)

        wl_regions = [evaluation_regions[k * L + wl_idx] for k in range(num_modes)]
        pred_energy_frac = region_energy_fractions(I_bhw, wl_regions, detect_radius=detect_radius)
        pred_amp_list.append(torch.sqrt(pred_energy_frac + 1e-12).detach().cpu())

    pred = torch.cat(pred_amp_list, dim=0).numpy()
    true = torch.cat(true_amp_list, dim=0).numpy()
    diff = pred - true
    abs_diff = np.abs(diff)
    rel = abs_diff / (np.abs(true) + 1e-12)
    cc = np.asarray(
        [_per_sample_corrcoef(pred[i], true[i]) for i in range(pred.shape[0])],
        dtype=np.float64,
    )
    return {
        "avg_amplitudes_diff":  float(abs_diff.mean()),
        "avg_relative_amp_err": float(rel.mean()),
        "cc_recon_amp":         cc,
        "amplitudes_diff":      diff,
    }


@torch.no_grad()
def evaluate_target_wl_over_all_wl_roi_ratio(
    model, loader, *, device, evaluation_regions, detect_radius, L, num_modes,
):
    model.eval()
    ratio_list = []
    for images, _label_img, _amp in loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)
        if images.ndim == 3:
            images = images.unsqueeze(1)
        x = images if images.shape[1] == L else images.repeat(1, L, 1, 1).contiguous()
        I_blhw = model(x).to(torch.float32)
        B = I_blhw.shape[0]
        ratios = torch.zeros((B, L), device=device, dtype=torch.float32)
        for s in range(L):
            src = I_blhw[:, s]
            E = torch.zeros((B, L), device=device, dtype=torch.float32)
            for t in range(L):
                t_regions = [evaluation_regions[m * L + t] for m in range(num_modes)]
                tot = torch.zeros((B,), device=device, dtype=torch.float32)
                for (x0, x1, y0, y1) in t_regions:
                    patch = src[:, y0:y1, x0:x1]
                    hh, ww = patch.shape[-2], patch.shape[-1]
                    cmask = _make_circle_mask(hh, ww, float(detect_radius), device=device)
                    tot += (patch * cmask.unsqueeze(0)).sum(dim=(-1, -2))
                E[:, t] = tot
            denom = E.sum(dim=1) + 1e-12
            ratios[:, s] = E[:, s] / denom
        ratio_list.append(ratios.detach().cpu())
    ratio_all = torch.cat(ratio_list, dim=0).numpy()
    return {
        "ratio_mean":   float(ratio_all.mean()),
        "ratio_per_wl": ratio_all.mean(axis=0),
    }
