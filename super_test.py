import argparse
import os
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import mat73
import scipy.io as sio

from ODNN_functions import (
    create_labels,
    generate_complex_weights,
    generate_fields_ts,
)
from odnn_model import D2NNModel, complex_pad_asymm


def load_complex_modes_from_mat(
    mat_path: Path,
    key: str | None = None,
    key_candidates: Tuple[str, ...] = ("eigenmodes_OM4_176", "modes_field", "modes", "E"),
) -> np.ndarray:
    """Load complex mode cube (H, W, M) from MAT/Mat73 file."""
 

    def _to_complex(payload):
        if isinstance(payload, np.ndarray) and np.iscomplexobj(payload):
            return payload.astype(np.complex64, copy=False)
        if isinstance(payload, dict):
            for re_key, im_key in (("real", "imag"), ("realPart", "imagPart"), ("Re", "Im")):
                if re_key in payload and im_key in payload:
                    return (
                        np.asarray(payload[re_key]) + 1j * np.asarray(payload[im_key])
                    ).astype(np.complex64, copy=False)
        if hasattr(payload, "dtype") and np.iscomplexobj(payload):
            return np.asarray(payload, dtype=np.complex64)
        raise ValueError("Unsupported MAT payload: expected complex array or dict with real/imag parts.")

    mat_path = Path(mat_path).expanduser()
    if not mat_path.exists():
        raise FileNotFoundError(f"Cannot find mode file: {mat_path}")

    try:
        raw = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        keys = [key] if key else [k for k in key_candidates if k in raw]
        if not keys:
            keys = [k for k in raw.keys() if not k.startswith("__")]
        cube = _to_complex(raw[keys[0]])
    except Exception:
        raw = mat73.loadmat(mat_path)
        keys = [key] if key else [k for k in key_candidates if k in raw] or [next(iter(raw.keys()))]
        cube = _to_complex(raw[keys[0]])

    cube = np.asarray(cube)
    if cube.ndim == 2:
        cube = cube[..., None]
    elif cube.ndim == 3 and cube.shape[0] != cube.shape[1] and cube.shape[1] == cube.shape[2]:
        cube = np.transpose(cube, (1, 2, 0))
    elif cube.ndim != 3:
        raise ValueError(f"Unexpected mode cube ndim={cube.ndim}; expected 2 or 3.")

    return cube.astype(np.complex64, copy=False)


def build_superposition_dataset(
    *,
    num_samples: int,
    num_modes: int,
    field_size: int,
    layer_size: int,
    focus_radius: int,
    phase_option: int,
    mat_path: Path,
    mat_key: str,
) -> Dict[str, torch.Tensor]:
    """Generate superposition inputs (phase stripped) and target labels."""
    mmf_cube = load_complex_modes_from_mat(mat_path, key=mat_key)
    if mmf_cube.shape[2] < num_modes:
        raise ValueError(f"Mode cube provides {mmf_cube.shape[2]} modes, need at least {num_modes}.")
    mmf_cube = mmf_cube[:, :, :num_modes].transpose(2, 0, 1)  # (M, H, W)

    # Normalise amplitude 0-1 while keeping phase information
    amp = np.abs(mmf_cube)
    denom = np.ptp(amp) + 1e-12
    mmf_cube = ((amp - amp.min()) / denom) * np.exp(1j * np.angle(mmf_cube))

    amplitudes, phases = generate_complex_weights(num_samples, num_modes, phase_option)
    amplitudes = amplitudes.astype(np.float32, copy=False)
    phases = phases.astype(np.float32, copy=False)

    complex_weights = (amplitudes * np.exp(1j * phases)).astype(np.complex64, copy=False)

    mmf_tensor = torch.from_numpy(mmf_cube.astype(np.complex64))
    weight_tensor = torch.from_numpy(complex_weights)

    fields = generate_fields_ts(weight_tensor, mmf_tensor, num_samples, num_modes, field_size)
    fields = torch.abs(fields).to(torch.complex64)  # strip phase before padding

    # detection masks (num_modes, layer_size, layer_size)
    detector_stack = torch.stack(
        [
            torch.from_numpy(
                create_labels(layer_size, layer_size, num_modes, focus_radius, idx + 1)
            ).to(torch.float32)
            for idx in range(num_modes)
        ],
        dim=0,
    )

    dh = layer_size - field_size
    dw = layer_size - field_size
    pt, pb = dh // 2, dh - dh // 2
    pl, pr = dw // 2, dw - dw // 2

    inputs = []
    labels = []
    for n in range(num_samples):
        field_small = fields[n, 0]  # (field_size, field_size) complex
        padded = complex_pad_asymm(field_small, pt, pb, pl, pr).unsqueeze(0)  # (1, layer_size, layer_size)
        inputs.append(padded)

        weights = torch.from_numpy(amplitudes[n])
        label_map = torch.einsum("k,kij->ij", weights, detector_stack)
        labels.append(label_map.unsqueeze(0))  # (1, layer_size, layer_size)

    inputs = torch.stack(inputs, dim=0)  # (N, 1, layer_size, layer_size) complex
    labels = torch.stack(labels, dim=0)  # (N, 1, layer_size, layer_size) float

    return {
        "inputs": inputs.to(torch.complex64),
        "labels": labels.to(torch.float32),
        "amplitudes": torch.from_numpy(amplitudes),  # (N, num_modes)
        "phases": torch.from_numpy(phases),          # (N, num_modes)
        "detectors": detector_stack,
    }


def energy_weights(
    intensity: np.ndarray,
    detector_masks: torch.Tensor,
) -> np.ndarray:
    """Compute amplitude-like weights from intensity via detector masks."""
    accum = []
    for mask in detector_masks.numpy():
        accum.append(float(np.sum(intensity * mask)))
    accum = np.array(accum, dtype=np.float64)
    total = np.sum(accum)
    if total <= 0:
        return np.zeros_like(accum, dtype=np.float32)
    weights_sq = accum / total
    weights = np.sqrt(np.clip(weights_sq, a_min=0.0, a_max=None))
    return weights.astype(np.float32)


def plot_sample(
    *,
    output_dir: Path,
    sample_idx: int,
    input_field: torch.Tensor,
    target_map: torch.Tensor,
    pred_map: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    amp_map = torch.abs(input_field[0]).cpu().numpy()
    amp_max = np.max(amp_map) if np.max(amp_map) > 0 else 1.0
    amp_norm = amp_map / amp_max

    label_np = target_map[0].cpu().numpy()

    diff = pred_map - label_np
    mse = np.mean(diff**2)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    im0 = axes[0].imshow(amp_norm, cmap="turbo", vmin=0.0, vmax=1.0)
    axes[0].set_title("Input amplitude")
    axes[0].axis("off")

    im1 = axes[1].imshow(label_np, cmap="turbo")
    axes[1].set_title("Target label")
    axes[1].axis("off")

    im2 = axes[2].imshow(pred_map, cmap="turbo")
    axes[2].set_title("Model output")
    axes[2].axis("off")

    clim = np.max(np.abs(diff)) or 1.0
    im3 = axes[3].imshow(diff, cmap="bwr", vmin=-clim, vmax=clim)
    axes[3].set_title(f"Difference\nMSE={mse:.3e}")
    axes[3].axis("off")

    for ax, im in zip(axes, (im0, im1, im2, im3)):
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Sample {sample_idx}", fontsize=12)
    fig.tight_layout()
    fig.savefig(output_dir / f"sample_{sample_idx:03d}.png", dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved ODNN model on random superposition samples.")
    parser.add_argument("--ckpt", type=Path, default=Path("checkpoints/odnn_2layers.pth"), help="Checkpoint path.")
    parser.add_argument("--mat", type=Path, default=Path("mmf_6modes_25_PD_1.15.mat"), help="MMF mode MAT file.")
    parser.add_argument("--mat-key", type=str, default="modes_field", help="Key inside MAT file.")
    parser.add_argument("--num-samples", type=int, default=32, help="Size of evaluation pool.")
    parser.add_argument("--sample-idx", type=int, default=None, help="Specific sample index to inspect (0-based).")
    parser.add_argument("--phase-option", type=int, default=3, help="Phase option for superposition test set.")
    parser.add_argument("--focus-radius", type=int, default=5, help="Detector focus radius.")
    parser.add_argument("--seed", type=int, default=20251102, help="Random seed for reproducibility.")
    parser.add_argument("--outdir", type=Path, default=Path("verify_outputs/super_test"), help="Where to store plots.")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.ckpt, map_location=device)
    meta = ckpt.get("meta", {})
    layer_size = int(meta.get("layer_size", 100))
    z_layers = float(meta.get("z_layers", 40e-6))
    z_prop = float(meta.get("z_prop", 120e-6))
    pixel_size = float(meta.get("pixel_size", 1e-6))
    wavelength = float(meta.get("wavelength", 1568e-9))
    z_input_to_first = float(meta.get("z_input_to_first", 40e-6))
    field_size = int(meta.get("field_size", 25))
    num_modes = int(meta.get("num_modes", 6))
    padding_ratio = float(meta.get("padding_ratio", 0.5))

    model = D2NNModel(
        num_layers=int(meta.get("num_layers", 2)),
        layer_size=layer_size,
        z_layers=z_layers,
        z_prop=z_prop,
        pixel_size=pixel_size,
        wavelength=wavelength,
        device=device,
        padding_ratio=padding_ratio,
        z_input_to_first=z_input_to_first,
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dataset = build_superposition_dataset(
        num_samples=args.num_samples,
        num_modes=num_modes,
        field_size=field_size,
        layer_size=layer_size,
        focus_radius=args.focus_radius,
        phase_option=args.phase_option,
        mat_path=args.mat,
        mat_key=args.mat_key,
    )

    num_available = dataset["inputs"].shape[0]
    sample_idx = args.sample_idx if args.sample_idx is not None else int(rng.integers(num_available))
    if not (0 <= sample_idx < num_available):
        raise ValueError(f"sample_idx {sample_idx} out of range for dataset size {num_available}.")

    input_field = dataset["inputs"][sample_idx]  # (1, layer_size, layer_size) complex
    target_map = dataset["labels"][sample_idx]   # (1, layer_size, layer_size) float
    target_weights = dataset["amplitudes"][sample_idx].numpy()
    detector_masks = dataset["detectors"]

    with torch.no_grad():
        pred = model(input_field.unsqueeze(0).to(device))
    pred_map = pred.squeeze().cpu().numpy()
    label_np = target_map.squeeze().numpy()

    mse = float(np.mean((pred_map - label_np) ** 2))
    mae = float(np.mean(np.abs(pred_map - label_np)))
    rel_l1 = float(np.sum(np.abs(pred_map - label_np)) / (np.sum(np.abs(label_np)) + 1e-12))

    pred_weights = energy_weights(pred_map, detector_masks)
    weight_diff = np.abs(pred_weights - target_weights)

    print(f"Sample index       : {sample_idx}")
    print(f"Checkpoint         : {args.ckpt}")
    print(f"MSE (output vs tgt): {mse:.6e}")
    print(f"MAE (output vs tgt): {mae:.6e}")
    print(f"Rel-L1 error       : {rel_l1:.6e}")
    print("Target amplitudes  :", np.round(target_weights, 4))
    print("Pred amplitudes    :", np.round(pred_weights, 4))
    print("Amplitude |Δ|      :", np.round(weight_diff, 4))

    outdir = Path(args.outdir)
    plot_sample(
        output_dir=outdir,
        sample_idx=sample_idx,
        input_field=input_field,
        target_map=target_map,
        pred_map=pred_map,
    )

    np.save(outdir / f"pred_{sample_idx:03d}.npy", pred_map.astype(np.float32))
    np.save(outdir / f"label_{sample_idx:03d}.npy", label_np.astype(np.float32))
    np.save(outdir / f"input_amp_{sample_idx:03d}.npy", torch.abs(input_field[0]).cpu().numpy().astype(np.float32))
    print(f"Artifacts saved under {outdir.resolve()}")


if __name__ == "__main__":
    main()
