# eval_metrics_all_layers.py
import os
import re
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import savemat
from torch.utils.data import DataLoader, TensorDataset

from ODNN_functions import generate_complex_weights, generate_fields_ts
from odnn_io import load_complex_modes_from_mat
from odnn_processing import prepare_sample
from odnn_multiwl_model import D2NNModelMultiWL


# ----------------------------
# label/roi + dataset (same as your training)
# ----------------------------
def generate_detector_patterns_multiwl(H, W, num_modes, num_wavelengths, radius, pattern_mode="circle"):
    total_labels = num_modes * num_wavelengths
    num_rows = num_modes
    num_cols = num_wavelengths

    margin = radius + 3
    xs = np.linspace(margin, W - 1 - margin, num_cols)
    ys = np.linspace(margin, H - 1 - margin, num_rows)

    centers = []
    for mode_idx in range(num_rows):
        for wl_idx in range(num_cols):
            cx = int(round(xs[wl_idx]))
            cy = int(round(ys[mode_idx]))
            centers.append((cy, cx))

    if pattern_mode != "circle":
        raise NotImplementedError(pattern_mode)

    patterns = np.zeros((H, W, total_labels), dtype=np.float32)
    yy, xx = np.ogrid[:H, :W]
    for idx, (cy, cx) in enumerate(centers):
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
        patterns[:, :, idx] = mask.astype(np.float32)

    evaluation_regions = []
    for cy, cx in centers:
        x0 = max(0, int(cx - radius))
        x1 = min(W, int(cx + radius))
        y0 = max(0, int(cy - radius))
        y1 = min(H, int(cy + radius))
        evaluation_regions.append((x0, x1, y0, y1))

    return patterns, evaluation_regions


def build_mode_context(base_modes: np.ndarray, num_modes: int, phase_option: int):
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
        "mmf_data_ts": torch.from_numpy(mmf_data),
        "base_amplitudes": base_amplitudes_local,
        "base_phases": base_phases_local,
    }


def build_eigenmode_dataset_multiwl(
    *,
    MMF_data_ts: torch.Tensor,
    MMF_Label_data: torch.Tensor,  # (H,W,M*L)
    field_size: int,
    layer_size: int,
    num_modes: int,
    L: int,
    phase_option: int,
    base_amplitudes: np.ndarray,
    base_phases: np.ndarray,
):
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

    dummy_label = torch.zeros([1, layer_size, layer_size], dtype=torch.float32)
    images_prepared = []
    for i in range(num_samples):
        img_i, _ = prepare_sample(image_data[i], dummy_label, layer_size)
        images_prepared.append(img_i)
    image_tensor = torch.stack(images_prepared, dim=0)

    for wl_idx in range(L):
        label_indices = [k * L + wl_idx for k in range(num_modes)]
        wl_label_patterns = MMF_Label_data[:, :, label_indices]  # (H,W,M)

        amp = torch.from_numpy(amplitudes.astype(np.float32))
        energy = amp ** 2
        label_img = torch.einsum("nm,hwm->nhw", energy, wl_label_patterns)
        label_img = label_img.unsqueeze(1).contiguous()

        amp_tensor = torch.from_numpy(np.asarray(amplitudes, dtype=np.float32))
        ds = TensorDataset(image_tensor, label_img, amp_tensor)
        datasets_per_wl.append(ds)

    return datasets_per_wl


# ----------------------------
# metrics
# ----------------------------
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
def region_energy_fractions(I_bhw: torch.Tensor, evaluation_regions, detect_radius: int) -> torch.Tensor:
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
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = (np.sqrt((a0 * a0).sum() + eps) * np.sqrt((b0 * b0).sum() + eps))
    return float((a0 * b0).sum() / denom)


@torch.no_grad()
def evaluate_spot_metrics_multiwl(model, loader, *, device, evaluation_regions, detect_radius, wl_idx, L, num_modes):
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

        wl_regions = [evaluation_regions[k * L + wl_idx] for k in range(num_modes)]
        pred_energy_frac = region_energy_fractions(I_bhw, wl_regions, detect_radius=detect_radius)
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
def evaluate_target_wl_over_all_wl_roi_ratio(model, loader, *, device, evaluation_regions, detect_radius, L, num_modes):
    model.eval()
    ratio_list = []

    for images, label_img, amp in loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)
        if images.ndim == 3:
            images = images.unsqueeze(1)

        x = images.repeat(1, L, 1, 1).contiguous()
        I_blhw = model(x).to(torch.float32)
        B = I_blhw.shape[0]
        ratios = torch.zeros((B, L), device=device, dtype=torch.float32)

        for s in range(L):
            src = I_blhw[:, s]
            E_in_each_wl_roi = torch.zeros((B, L), device=device, dtype=torch.float32)

            for t in range(L):
                t_regions = [evaluation_regions[m * L + t] for m in range(num_modes)]
                total = torch.zeros((B,), device=device, dtype=torch.float32)
                for (x0, x1, y0, y1) in t_regions:
                    patch = src[:, y0:y1, x0:x1]
                    hh, ww = patch.shape[-2], patch.shape[-1]
                    cmask = _make_circle_mask(hh, ww, float(detect_radius), device=device)
                    total += (patch * cmask.unsqueeze(0)).sum(dim=(-1, -2))
                E_in_each_wl_roi[:, t] = total

            denom = E_in_each_wl_roi.sum(dim=1) + 1e-12
            ratios[:, s] = E_in_each_wl_roi[:, s] / denom

        ratio_list.append(ratios.detach().cpu())

    ratio_all = torch.cat(ratio_list, dim=0).numpy()
    return {"ratio_mean": float(ratio_all.mean()), "ratio_per_wl": ratio_all.mean(axis=0)}


def parse_num_layers_from_name(p: Path) -> Optional[int]:

    m = re.search(r"multiwl_(\d+)layers", p.name)
    if m:
        return int(m.group(1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", type=str, required=True, help="directory containing multiwl_*layers_*.pth")
    ap.add_argument("--out", type=str, required=True, help="RUN_ROOT output dir (writes metrics_analysis under it)")
    ap.add_argument("--gpu", type=int, default=3)

    # must match training
    ap.add_argument("--layer_size", type=int, default=200)
    ap.add_argument("--field_size", type=int, default=25)
    ap.add_argument("--num_modes", type=int, default=3)
    ap.add_argument("--detectsize", type=int, default=10)
    ap.add_argument("--circle_focus_radius", type=int, default=5)
    ap.add_argument("--padding_ratio", type=float, default=0.5)
    ap.add_argument("--z_layers", type=float, default=40e-6)
    ap.add_argument("--z_prop", type=float, default=120e-6)
    ap.add_argument("--z_input_to_first", type=float, default=40e-6)

    ap.add_argument("--phase_option", type=int, default=4)
    ap.add_argument("--base_wavelength_idx", type=int, default=4)

    ap.add_argument("--modes_mat", type=str, default="mmf_103modes_25_PD_1.15.mat")
    ap.add_argument("--modes_key", type=str, default="modes_field")

    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", device)

    wavelengths = np.array(
        [1530e-9, 1535e-9, 1540e-9, 1545e-9, 1550e-9, 1555e-9, 1560e-9, 1565e-9],
        dtype=np.float32,
    )
    L = int(len(wavelengths))
    num_modes = int(args.num_modes)
    layer_size = int(args.layer_size)
    detect_radius_eval = int(args.detectsize // 2)

    RUN_ROOT = Path(args.out)
    RUN_ROOT.mkdir(parents=True, exist_ok=True)

    # build fixed labels/dataset once
    mmf_label_patterns, evaluation_regions = generate_detector_patterns_multiwl(
        H=layer_size,
        W=layer_size,
        num_modes=num_modes,
        num_wavelengths=L,
        radius=int(args.circle_focus_radius),
        pattern_mode="circle",
    )
    MMF_Label_data = torch.from_numpy(mmf_label_patterns).to(torch.float32)

    eigenmodes = load_complex_modes_from_mat(args.modes_mat, key=args.modes_key)
    mode_context = build_mode_context(eigenmodes, num_modes=num_modes, phase_option=int(args.phase_option))
    test_datasets_per_wl = build_eigenmode_dataset_multiwl(
        MMF_data_ts=mode_context["mmf_data_ts"],
        MMF_Label_data=MMF_Label_data,
        field_size=int(args.field_size),
        layer_size=layer_size,
        num_modes=num_modes,
        L=L,
        phase_option=int(args.phase_option),
        base_amplitudes=mode_context["base_amplitudes"],
        base_phases=mode_context["base_phases"],
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpts = sorted(ckpt_dir.glob("*.pth"))
    items = []
    for p in ckpts:
        nl = parse_num_layers_from_name(p)
        if nl is not None:
            items.append((nl, p))
    items.sort(key=lambda x: x[0])

    if not items:
        raise RuntimeError(f"No multiwl_*layers*.pth found in {ckpt_dir}")

    print("Found checkpoints:")
    for nl, p in items:
        print(" ", nl, "->", p)

    metrics_by_wl = {int(li): [] for li in range(L)}
    target_ratio_per_layer = {}  # num_layers -> (L,)

    # evaluate each checkpoint
    for num_layers, ckpt_path in items:
        print(f"\n--- Evaluating num_layers={num_layers} | {ckpt_path.name} ---")
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt

        model = D2NNModelMultiWL(
            num_layers=int(num_layers),
            layer_size=layer_size,
            z_layers=float(args.z_layers),
            z_prop=float(args.z_prop),
            pixel_size=1e-6,
            wavelengths=wavelengths,
            device=device,
            padding_ratio=float(args.padding_ratio),
            z_input_to_first=float(args.z_input_to_first),
            base_wavelength_idx=int(args.base_wavelength_idx),
        ).to(device)

        model.load_state_dict(state_dict, strict=True)
        model.eval()

        for li in range(L):
            test_loader_wl = DataLoader(test_datasets_per_wl[li], batch_size=16, shuffle=False)
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
            metrics_by_wl[int(li)].append({"num_layers": int(num_layers), **metrics})

        test_loader_any = DataLoader(test_datasets_per_wl[0], batch_size=16, shuffle=False)
        wl_ratio = evaluate_target_wl_over_all_wl_roi_ratio(
            model,
            test_loader_any,
            device=device,
            evaluation_regions=evaluation_regions,
            detect_radius=detect_radius_eval,
            L=L,
            num_modes=num_modes,
        )
        target_ratio_per_layer[int(num_layers)] = wl_ratio["ratio_per_wl"]
        print("TargetWL/AllWL ROI mean:", wl_ratio["ratio_mean"])

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # save plots/mats (metrics vs layers)
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

        tgt_curve = np.asarray([target_ratio_per_layer[int(nl)][li] for nl in layer_counts], dtype=np.float64)

        fig, axes = plt.subplots(4, 1, figsize=(7, 11), sharex=True)

        axes[0].plot(layer_counts, amp_err, marker="o")
        axes[0].set_ylabel("Avg Amplitude Error")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(layer_counts, amp_err_rel, marker="o", color="tab:orange")
        axes[1].set_ylabel("Avg Relative Error")
        axes[1].grid(True, alpha=0.3)

        axes[2].errorbar(layer_counts, cc_amp_mean, yerr=cc_amp_std, marker="o", capsize=4, color="tab:green")
        axes[2].set_ylabel("Correlation Coef")
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(0.0, 1.01)

        axes[3].plot(layer_counts, tgt_curve, marker="o", color="tab:purple")
        axes[3].set_ylabel("TargetWL / AllWL (ROI)")
        axes[3].set_xlabel("Number of Layers")
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(0.0, 1.01)

        fig.suptitle(f"Metrics vs Layers | λ={wavelengths[li]*1e9:.1f} nm")
        fig.tight_layout(rect=[0, 0.0, 1, 0.96])

        fig_path = metrics_dir / f"metrics_wl{li}_{tag}.png"
        fig.savefig(fig_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print("✔ Metrics plot saved ->", fig_path)

        mat_path = metrics_dir / f"metrics_wl{li}_{tag}.mat"
        savemat(
            str(mat_path),
            {
                "layer_counts": layer_counts,
                "avg_amp_error": amp_err,
                "avg_relative_amp_error": amp_err_rel,
                "cc_amp_mean": cc_amp_mean,
                "cc_amp_std": cc_amp_std,
                "target_wl_over_all_wl_roi": tgt_curve,
                "wavelength_nm": np.array([wavelengths[li] * 1e9], dtype=np.float32),
            },
        )
        print("✔ Metrics MAT saved ->", mat_path)

    print("✅ Done. metrics saved to:", metrics_dir)


if __name__ == "__main__":
    main()
