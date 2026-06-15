#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Batch sweep script for ODNN parameter analysis.

Launched as a subprocess by the frontend Analysis page.
Iterates over (num_modes × layer_size × num_layers) combinations,
trains a D2NN model for each, evaluates metrics, and writes
one JSON line per combination to sweep_metrics.jsonl for live
frontend monitoring.

Does NOT import mainfor6.py — only uses shared odnn modules.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset

# Shared odnn modules (same as mainfor6.py)
from ODNN_functions import (
    create_evaluation_regions,
    generate_complex_weights,
    generate_fields_ts,
)
from odnn_generate_label import (
    compute_label_centers,
    compose_labels_from_patterns,
    generate_detector_patterns,
)
from odnn_io import load_complex_modes_from_mat
from odnn_model import D2NNModel
from odnn_processing import prepare_sample
from odnn_training_eval import evaluate_spot_metrics
from odnn_wavelength_analysis import (
    ModelGeometry,
    compute_mode_isolation_wavelength_sweep,
)

SEED = 424242

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def build_mode_context(
    base_modes: np.ndarray, num_modes: int
) -> Dict[str, Any]:
    """Build mode-dependent tensors and one-hot weights for eigenmode training.

    Identical logic to mainfor6.py `build_mode_context`.
    """
    if base_modes.shape[2] < num_modes:
        raise ValueError(
            f"Requested {num_modes} modes, source only has {base_modes.shape[2]}."
        )
    mmf_data = base_modes[:, :, :num_modes].transpose(2, 0, 1).copy()
    amp_min = np.min(np.abs(mmf_data))
    amp_max = np.max(np.abs(mmf_data))
    mmf_data_norm = (np.abs(mmf_data) - amp_min) / (amp_max - amp_min + 1e-12)
    mmf_data = mmf_data_norm * np.exp(1j * np.angle(mmf_data))

    # phase_option == 4: one-hot amplitudes (eigenmode)
    amplitudes = np.eye(num_modes, dtype=np.float32)
    phases = np.eye(num_modes, dtype=np.float32)

    return {
        "mmf_data_np": mmf_data,
        "mmf_data_ts": torch.from_numpy(mmf_data),
        "base_amplitudes": amplitudes,
        "base_phases": phases,
    }


def compute_overflow_ratio(
    model: D2NNModel,
    input_field_2d: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute energy overflow ratio for a single input field.

    overflow = 1 - output_energy / input_energy

    Positive value means energy leaked outside the FOV during propagation.
    """
    model.eval()
    with torch.no_grad():
        field = input_field_2d.to(device=device, dtype=torch.complex64)

        # pad to layer_size (same as prepare_sample does)
        _, h, w = field.shape if field.dim() == 3 else (1, field.shape[0], field.shape[1])
        if field.dim() == 2:
            field = field.unsqueeze(0)
        # field is now (1, field_size, field_size), pad to layer_size
        from odnn_processing import pad_field_to_layer
        layer_size = model.layers[0].units
        padded = pad_field_to_layer(field.squeeze(0), layer_size).unsqueeze(0).unsqueeze(0)
        # (1, 1, layer_size, layer_size)

        input_intensity = torch.abs(padded) ** 2
        input_energy = float(input_intensity.sum().item())

        output = model(padded.to(torch.complex64))
        output_intensity = output  # model.regression already squares abs
        output_energy = float(output_intensity.sum().item())

    if input_energy < 1e-12:
        return 0.0
    return float(1.0 - output_energy / input_energy)


# ---------------------------------------------------------------------------
# main sweep
# ---------------------------------------------------------------------------


def run_sweep(config: Dict[str, Any]) -> None:
    """Execute the full (num_modes × layer_size × num_layers) sweep."""

    # --- unpack config --------------------------------------------------
    mat_file = config["mat_file"]
    num_modes_list = config["num_modes_list"]
    num_layers_list = config["num_layers_list"]
    layer_size_list = config["layer_size_list"]
    field_size = config["field_size"]
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    lr = config["lr"]
    lr_gamma = config["lr_gamma"]
    wavelength = config["wavelength_nm"] * 1e-9
    z_layers = config["z_layers_um"] * 1e-6
    z_prop = config["z_prop_um"] * 1e-6
    z_input_to_first = config["z_input_to_first_um"] * 1e-6
    pixel_size = config["pixel_size_um"] * 1e-6
    label_pattern_mode = config.get("label_pattern_mode", "circle")
    phase_option = config.get("phase_option", 4)
    output_dir = Path(config["output_dir"])
    metrics_log = Path(config["metrics_log"])
    log_file = Path(config["log_file"])

    # --- seed & device --------------------------------------------------
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        device = torch.device("cuda:2" if torch.cuda.device_count() > 2 else "cuda")
        print(f"Using Device: {device}")
    else:
        device = torch.device("cpu")
        print("Using Device: CPU")

    # --- prepare output paths -------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_log.parent.mkdir(parents=True, exist_ok=True)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # clear previous metrics
    metrics_log.write_text("")

    # redirect stdout to log file (keep parent process clean)
    log_fh = open(log_file, "w")
    sys.stdout = log_fh
    sys.stderr = log_fh

    # --- load .mat modes ------------------------------------------------
    print(f"Loading mat file: {mat_file}")
    eigenmodes = load_complex_modes_from_mat(mat_file, key="modes_field")
    print(f"Loaded modes shape: {eigenmodes.shape}")

    total_modes = eigenmodes.shape[2]
    total_combinations = (
        len(num_modes_list) * len(layer_size_list) * len(num_layers_list)
    )
    print(f"Total combinations: {total_combinations}")
    print(f"Config: num_modes={num_modes_list}, layer_sizes={layer_size_list}, "
          f"num_layers={num_layers_list}, epochs={epochs}")

    completed = 0

    # =====================================================================
    # Outer loop: num_modes
    # =====================================================================
    for num_modes in num_modes_list:
        if num_modes > total_modes:
            print(f"SKIP num_modes={num_modes} > total available ({total_modes})")
            continue

        mode_ctx = build_mode_context(eigenmodes, num_modes)
        mmf_data_np = mode_ctx["mmf_data_np"]
        mmf_data_ts = mode_ctx["mmf_data_ts"]
        base_amplitudes = mode_ctx["base_amplitudes"]
        base_phases = mode_ctx["base_phases"]

        # pick a fixed eigenmode field for overflow measurement (mode 0)
        overflow_eigenmode = mmf_data_ts[0]  # (field_size, field_size) complex

        # ===================================================================
        # Middle loop: layer_size
        # ===================================================================
        for layer_size in layer_size_list:
            label_size = layer_size
            num_detector = num_modes

            # --- build label maps (circle pattern) --------------------------
            circle_focus_radius = 5
            circle_detectsize = 10
            circle_radius = circle_focus_radius
            pattern_size = circle_radius * 2
            if pattern_size % 2 == 0:
                pattern_size += 1
            pattern_stack = generate_detector_patterns(
                pattern_size, pattern_size, num_detector, shape="circle"
            )
            layout_radius = circle_radius
            focus_radius = circle_focus_radius
            detectsize = circle_detectsize

            centers, _, _ = compute_label_centers(
                label_size, label_size, num_detector, layout_radius
            )
            mode_label_maps = [
                compose_labels_from_patterns(
                    label_size, label_size, pattern_stack, centers,
                    Index=i + 1, visualize=False,
                )
                for i in range(num_detector)
            ]
            MMF_Label_data = torch.from_numpy(
                np.stack(mode_label_maps, axis=2).astype(np.float32)
            )

            # --- build eigenmode training dataset --------------------------
            num_train_samples = num_modes
            amplitudes = base_amplitudes[:num_train_samples]
            phases = base_phases[:num_train_samples]
            amplitudes_phases = np.hstack(
                (amplitudes, phases[:, 1:] / (2 * np.pi))
            )
            label_data = torch.zeros(
                [num_train_samples, 1, label_size, label_size]
            )
            amp_weights = torch.from_numpy(
                amplitudes_phases[:, :num_modes]
            ).float()
            energy_weights = amp_weights ** 2
            combined_labels = (
                energy_weights[:, None, None, :] * MMF_Label_data.unsqueeze(0)
            ).sum(dim=3)
            label_data[:, 0, :, :] = combined_labels

            complex_weights = amplitudes * np.exp(1j * phases)
            complex_weights_ts = torch.from_numpy(
                complex_weights.astype(np.complex64)
            )
            image_data = generate_fields_ts(
                complex_weights_ts, mmf_data_ts,
                num_train_samples, num_modes, field_size,
            ).to(torch.complex64)

            train_dataset = [
                prepare_sample(image_data[i], label_data[i], label_size)
                for i in range(num_train_samples)
            ]
            train_tensor_data = TensorDataset(
                *[torch.stack(t) for t in zip(*train_dataset)]
            )

            g = torch.Generator()
            g.manual_seed(SEED)
            train_loader = DataLoader(
                train_tensor_data, batch_size=batch_size, shuffle=True,
                generator=g,
            )
            # test = train for eigenmode
            test_loader = DataLoader(
                train_tensor_data, batch_size=batch_size, shuffle=False,
            )

            eval_amplitudes = amplitudes
            eval_amplitudes_phases = amplitudes_phases
            eval_phases = phases
            image_test_data = image_data

            evaluation_regions = create_evaluation_regions(
                label_size, label_size, num_detector, focus_radius, detectsize,
            )

            # =================================================================
            # Inner loop: num_layers
            # =================================================================
            for num_layers in num_layers_list:
                combination_label = (
                    f"modes={num_modes}, ls={layer_size}, L={num_layers}"
                )
                print(f"\n{'='*60}")
                print(f"Training: {combination_label}")
                print(f"{'='*60}")

                t0 = time.time()

                try:
                    # --- build model ----------------------------------------
                    model = D2NNModel(
                        num_layers=num_layers,
                        layer_size=label_size,
                        z_layers=z_layers,
                        z_prop=z_prop,
                        pixel_size=pixel_size,
                        wavelength=wavelength,
                        device=device,
                        padding_ratio=0.5,
                        z_input_to_first=z_input_to_first,
                    ).to(device)

                    # --- train ----------------------------------------------
                    criterion = nn.MSELoss()
                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    scheduler = ExponentialLR(optimizer, gamma=lr_gamma)

                    for epoch in range(1, epochs + 1):
                        model.train()
                        epoch_loss = 0.0
                        for images, labels in train_loader:
                            images = images.to(
                                device, dtype=torch.complex64, non_blocking=True
                            )
                            labels = labels.to(
                                device, dtype=torch.float32, non_blocking=True
                            )
                            optimizer.zero_grad(set_to_none=True)
                            outputs = model(images)
                            loss = criterion(outputs, labels)
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()
                        scheduler.step()
                        avg_loss = epoch_loss / len(train_loader)
                        if epoch % 100 == 0 or epoch == 1 or epoch == epochs:
                            print(
                                f"  Epoch {epoch}/{epochs}  loss={avg_loss:.8f}"
                            )

                    if device.type == "cuda":
                        torch.cuda.synchronize(device)

                    # --- evaluate metrics -----------------------------------
                    metrics = evaluate_spot_metrics(
                        model, test_loader, evaluation_regions,
                        detect_radius=detectsize,
                        device=device, pred_case=1,
                        num_modes=num_modes, phase_option=phase_option,
                        amplitudes=eval_amplitudes,
                        amplitudes_phases=eval_amplitudes_phases,
                        phases=eval_phases,
                        mmf_modes=mmf_data_ts,
                        field_size=field_size,
                        image_test_data=image_test_data,
                    )

                    # --- mode isolation (single wavelength) -----------------
                    phase_masks_np = []
                    for layer in model.layers:
                        p = layer.phase.detach().cpu().numpy()
                        phase_masks_np.append(np.remainder(p, 2 * np.pi))

                    geometry = ModelGeometry(
                        layer_size=layer_size,
                        z_layers=z_layers,
                        z_prop=z_prop,
                        pixel_size=pixel_size,
                        z_input_to_first=z_input_to_first,
                    )
                    iso_result = compute_mode_isolation_wavelength_sweep(
                        phase_masks_np,
                        base_wavelength=wavelength,
                        wavelength_list=[wavelength],
                        geometry=geometry,
                        device=device,
                        test_loader=test_loader,
                        evaluation_regions=evaluation_regions,
                        detect_radius=detectsize // 2,
                    )

                    isolation_db_mean = float(
                        iso_result["isolation_db_mean"][0]
                    )
                    isolation_db_per_mode = (
                        iso_result["isolation_db"][0].tolist()
                    )

                    # --- overflow ratio ------------------------------------
                    overflow = compute_overflow_ratio(
                        model, overflow_eigenmode, device
                    )

                    if device.type == "cuda":
                        torch.cuda.empty_cache()

                    elapsed = time.time() - t0

                    avg_relative_amp_err = float(
                        metrics.get("avg_relative_amp_err", float("nan"))
                    )
                    snr_db_full = float(
                        metrics.get("snr_db_full", float("nan"))
                    )

                    print(
                        f"  Done {combination_label}  "
                        f"rel_amp_err={avg_relative_amp_err:.6f}  "
                        f"SNR={snr_db_full:.2f} dB  "
                        f"iso={isolation_db_mean:.2f} dB  "
                        f"overflow={overflow:.4f}  "
                        f"time={elapsed:.1f}s"
                    )

                    record = {
                        "status": "done",
                        "num_modes": num_modes,
                        "num_layers": num_layers,
                        "layer_size": layer_size,
                        "avg_relative_amp_err": round(avg_relative_amp_err, 6),
                        "snr_db_full": round(snr_db_full, 4),
                        "isolation_db_mean": round(isolation_db_mean, 4),
                        "isolation_db_per_mode": [
                            round(v, 4) for v in isolation_db_per_mode
                        ],
                        "overflow_ratio": round(overflow, 6),
                        "elapsed_s": round(elapsed, 1),
                    }

                except Exception as exc:
                    import traceback
                    elapsed = time.time() - t0
                    print(f"  FAILED {combination_label}: {exc}")
                    traceback.print_exc()
                    record = {
                        "status": "error",
                        "num_modes": num_modes,
                        "num_layers": num_layers,
                        "layer_size": layer_size,
                        "error": str(exc),
                        "elapsed_s": round(elapsed, 1),
                    }

                # write single line to JSONL
                with open(metrics_log, "a") as mf:
                    mf.write(json.dumps(record) + "\n")
                mf.close()

                completed += 1

        # --- end-of-group summary for one (num_modes, layer_size) pair ---
        print(
            f"\n--- Completed group: num_modes={num_modes}, "
            f"layer_size={layer_size}, progress={completed}/{total_combinations} ---"
        )

    # --- final summary ---------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Sweep finished. {completed}/{total_combinations} combinations completed.")
    print(f"Results written to: {metrics_log}")
    log_fh.close()


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True,
                   help="Path to sweep config JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    # fill defaults for optional fields
    cfg.setdefault("field_size", None)
    cfg.setdefault("label_pattern_mode", "circle")
    cfg.setdefault("phase_option", 4)
    cfg.setdefault("output_dir", "results/sweep")
    cfg.setdefault("metrics_log", "frontend/logs/sweep_metrics.jsonl")
    cfg.setdefault("log_file", "frontend/logs/sweep.log")
    run_sweep(cfg)
