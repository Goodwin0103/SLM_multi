import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from adapters.base_adapter import BaseODNNAdapter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FRONTEND_DIR = PROJECT_ROOT / "frontend"
TEMP_DIR     = FRONTEND_DIR / "temp"
LOG_DIR      = FRONTEND_DIR / "logs"
RESULTS_DIR  = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

_TRAINING_LOG = LOG_DIR / "training_wl.log"
_CONFIG_PATH  = TEMP_DIR / "train_config_wl.json"


class Mainfor6WLAdapter(BaseODNNAdapter):
    """Adapter for mainfor6_wl.py (multi-wavelength D2NN).

    Training is launched as a non-blocking subprocess; testing is done
    in-process (synchronous).
    """

    def __init__(self) -> None:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def load_default_config(self) -> Dict[str, Any]:
        """Canonical defaults matching mainfor6_wl.py hard-coded values."""
        return {
            # geometry
            "layer_size": 300,
            "out_size": 600,
            "padding_ratio_out": 0.5,
            "num_layers_list": [1, 2, 3, 4, 5],
            "field_size": 176,
            # physics (SLM parameters)
            "wl_start_nm": 1550,
            "wl_spacing_nm": 0.5,
            "wl_count": 2,
            "base_wavelength_idx": 0,
            "z_layers_um": 45,
            "z_prop_um": 200000,
            "z_input_to_first_um": 0,
            "pixel_size_um": 12.5,
            # modes
            "num_modes": 10,
            "phase_option": 4,
            # label
            "circle_focus_radius": 5,
            "margin_ratio": 0.2,
            # dataset
            "num_data": 1000,
            "batch_size": 16,
            "training_dataset_mode": "eigenmode",
            "label_pattern_mode": "circle",
            # training
            "epochs": 1000,
            "learning_rate": 1.99,
            "lr_gamma": 0.99,
            "padding_ratio": 0.5,
            # evaluation
            "evaluation_mode": "eigenmode",
            "num_superposition_eval_samples": 1000,
        }

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def start_training(self, config: Dict[str, Any], mat_file: str = "") -> int:
        """Write config to disk and launch mainfor6_wl.py as a subprocess.

        Returns:
            PID of the spawned process.

        Raises:
            ValueError: if mat_file is empty.
        """
        if not mat_file:
            raise ValueError("mat_file path is required to start training.")

        with open(_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)

        with open(_TRAINING_LOG, "w") as log_fh:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "mainfor6_wl.py"),
                    "--config",   str(_CONFIG_PATH),
                    "--mat_file", mat_file,
                ],
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )

        return proc.pid

    def stop_training(self, pid: int) -> None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def is_training_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    # ------------------------------------------------------------------
    # Log access
    # ------------------------------------------------------------------

    def read_log_tail(self, n: int = 50) -> List[str]:
        if not _TRAINING_LOG.exists():
            return []
        try:
            lines = _TRAINING_LOG.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            return lines[-n:]
        except OSError:
            return []

    # ------------------------------------------------------------------
    # Checkpoint discovery
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[str]:
        """Return sorted list of absolute .pth paths across all known locations."""
        paths: List[Path] = []

        # fixed checkpoints/ directory
        if CHECKPOINTS_DIR.exists():
            paths.extend(CHECKPOINTS_DIR.glob("*.pth"))
            for sub in CHECKPOINTS_DIR.iterdir():
                if sub.is_dir():
                    paths.extend(sub.glob("*.pth"))

        # results/**/checkpoints/ (dynamic per-run directories)
        for sub in PROJECT_ROOT.glob("results/**/checkpoints"):
            if sub.is_dir():
                paths.extend(sub.glob("*.pth"))

        return sorted(str(p) for p in paths)

    def load_checkpoint_meta(self, pth_path: str) -> Dict[str, Any]:
        import torch
        try:
            ckpt = torch.load(pth_path, map_location="cpu", weights_only=False)
            return dict(ckpt.get("meta", {}))
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Testing (in-process)
    # ------------------------------------------------------------------

    def run_test(
        self,
        config: Dict[str, Any],
        checkpoint_path: str,
        mat_file: str,
    ) -> Dict[str, Any]:
        """Load a MultiWL checkpoint and run comprehensive evaluation.

        Returns frames, player_frames, full metrics dict, and metadata.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        from odnn_multiwl_model import D2NNModelMultiWL
        from odnn_io import load_complex_modes_from_mat
        from odnn_processing import prepare_sample, pad_field_to_layer
        from ODNN_functions import generate_fields_ts
        from odnn_multiwl_metrices import evaluate_multiwl_comprehensive_metrics

        # -- load model --------------------------------------------------
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        meta = ckpt.get("meta", {})

        # read training config JSON as fallback for old checkpoints
        _train_cfg_path = FRONTEND_DIR / "temp" / "train_config_wl.json"
        _train_cfg: Dict[str, Any] = {}
        if _train_cfg_path.exists():
            try:
                _train_cfg = json.loads(_train_cfg_path.read_text())
            except Exception:
                pass

        def _m(key: str, default: Any) -> Any:
            v = meta.get(key)
            if v is not None:
                return v
            v = _train_cfg.get(key)
            if v is not None:
                return v
            return default

        num_layers       = int(_m("num_layers", 3))
        layer_size       = int(_m("layer_size", 300))
        out_size         = int(_m("out_size", layer_size))
        padding_ratio_out = float(_m("padding_ratio_out", 0.5))
        num_modes_train  = int(_m("num_modes", 10))
        wavelengths_meta = meta.get("wavelengths", None)
        if wavelengths_meta is not None:
            wavelengths = np.asarray(wavelengths_meta, dtype=np.float64)
        else:
            # construct from wl_start_nm / wl_spacing_nm / wl_count
            ws = float(_m("wl_start_nm", _train_cfg.get("wl_start_nm", 1550)))
            wd = float(_m("wl_spacing_nm", _train_cfg.get("wl_spacing_nm", 0.5)))
            wc = int(_m("wl_count", _train_cfg.get("wl_count", 2)))
            wavelengths = (ws + np.arange(wc) * wd).astype(np.float64) * 1e-9

        # physics params — meta stores metres, training config stores um with _um suffix
        def _m_um(meta_key: str, cfg_key_um: str, default_m: float) -> float:
            v = meta.get(meta_key)
            if v is not None:
                return float(v)
            v = _train_cfg.get(cfg_key_um)
            if v is not None:
                return float(v) * 1e-6  # config stores um, convert to m
            return float(default_m)

        z_layers         = _m_um("z_layers",         "z_layers_um",         45e-3)
        z_prop           = _m_um("z_prop",           "z_prop_um",           20e-2)
        z_input_to_first = _m_um("z_input_to_first", "z_input_to_first_um", 0.0)
        pixel_size       = _m_um("pixel_size",       "pixel_size_um",       12.5e-6)
        padding_ratio    = float(_m("padding_ratio", 0.5))
        base_wavelength_idx = int(_m("base_wavelength_idx", 0))
        L = int(len(wavelengths))

        # user-configurable overrides
        num_modes_valid = min(
            int(config.get("num_modes", num_modes_train)), num_modes_train
        )
        test_mode           = str(config.get("evaluation_mode", "eigenmode"))
        circle_detectsize   = int(config.get("circle_detectsize", 10))
        circle_focus_radius = int(_m("circle_focus_radius", 5))
        margin_ratio_eval   = float(_m("margin_ratio", 0.2))
        z_step_um           = float(config.get("z_step_um", 20.0))
        mode_index          = int(config.get("mode_index", 0))
        super_samples       = int(config.get("num_superposition_eval_samples", 1000))
        super_seed          = int(config.get("superposition_eval_seed", 20240116))
        phase_option        = int(meta.get("phase_option", 4))

        device = torch.device("cpu")

        model = D2NNModelMultiWL(
            num_layers=num_layers,
            layer_size=layer_size,
            z_layers=z_layers,
            z_prop=z_prop,
            pixel_size=pixel_size,
            wavelengths=wavelengths.astype(np.float32),
            device=device,
            padding_ratio=padding_ratio,
            z_input_to_first=z_input_to_first,
            base_wavelength_idx=base_wavelength_idx,
            out_size=out_size,
            padding_ratio_out=padding_ratio_out,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # -- load & normalise eigenmodes ---------------------------------
        eigenmodes_raw = load_complex_modes_from_mat(mat_file, key="modes_field")
        field_size = int(eigenmodes_raw.shape[0])
        max_modes_mat = int(eigenmodes_raw.shape[2])
        if max_modes_mat < num_modes_valid:
            raise ValueError(
                f"Mat file only has {max_modes_mat} modes, requested {num_modes_valid}."
            )
        mmf_np = eigenmodes_raw[:, :, :num_modes_valid].transpose(2, 0, 1)
        amp_min = float(np.min(np.abs(mmf_np)))
        amp_max = float(np.max(np.abs(mmf_np)))
        mmf_norm = (np.abs(mmf_np) - amp_min) / (amp_max - amp_min + 1e-12)
        mmf_np = mmf_norm * np.exp(1j * np.angle(mmf_np))
        mmf_ts = torch.from_numpy(mmf_np.astype(np.complex64))

        # -- generate evaluation regions (on out_size canvas) ------------
        eval_regions = _build_evaluation_regions(
            H=out_size, W=out_size,
            num_modes=num_modes_valid, num_wavelengths=L,
            radius=circle_focus_radius, margin_ratio=margin_ratio_eval,
        )

        # -- build test dataset ------------------------------------------
        if phase_option == 4:
            amplitudes = np.eye(num_modes_valid, dtype=np.float32)
            phases = np.eye(num_modes_valid, dtype=np.float32)
        else:
            from ODNN_functions import generate_complex_weights
            amplitudes, phases = generate_complex_weights(1000, num_modes_valid, phase_option)

        if test_mode == "eigenmode":
            num_samples = num_modes_valid
            amps_eval = amplitudes[:num_samples]
            phs_eval = phases[:num_samples]
        else:
            rng = np.random.RandomState(super_seed)
            amps_eval = rng.uniform(0.0, 1.0, size=(super_samples, num_modes_valid)).astype(np.float32)
            amps_eval = amps_eval / (np.linalg.norm(amps_eval, axis=1, keepdims=True) + 1e-12)
            phs_eval = np.zeros_like(amps_eval) if phase_option == 4 else rng.uniform(
                0.0, 2 * np.pi, size=(super_samples, num_modes_valid)
            ).astype(np.float32)

        complex_weights = amps_eval * np.exp(1j * phs_eval)
        cw_ts = torch.from_numpy(complex_weights.astype(np.complex64))
        image_data = generate_fields_ts(
            cw_ts, mmf_ts, int(amps_eval.shape[0]), num_modes_valid, field_size,
        ).to(torch.complex64)

        dummy_label = torch.zeros([1, layer_size, layer_size], dtype=torch.float32)
        images_prepared = []
        for i in range(int(amps_eval.shape[0])):
            img_i, _ = prepare_sample(image_data[i], dummy_label, layer_size)
            images_prepared.append(img_i)
        image_tensor = torch.stack(images_prepared, dim=0)

        # Build per-wavelength label patterns for label_field (on out_size canvas)
        label_patterns_np, _ = _build_label_patterns(
            H=out_size, W=out_size,
            num_modes=num_modes_valid, num_wavelengths=L,
            radius=circle_focus_radius, margin_ratio=margin_ratio_eval,
        )

        # -- comprehensive metrics ---------------------------------------
        metrics_result = evaluate_multiwl_comprehensive_metrics(
            model=model,
            evaluation_regions=eval_regions,
            detect_radius=int(circle_detectsize // 2),
            device=device,
            num_modes=num_modes_valid,
            num_wavelengths=L,
            wavelengths_m=wavelengths,
            mmf_modes=mmf_ts,
            layer_size=layer_size,
        )

        # -- pick sample for propagation viz -----------------------------
        if test_mode == "eigenmode":
            vis_idx = max(0, min(mode_index, num_modes_valid - 1))
            eigen_field = mmf_ts[vis_idx]
        else:
            vis_idx = max(0, min(int(config.get("superposition_vis_sample", 0)), int(amps_eval.shape[0]) - 1))
            eigen_field = image_data[vis_idx]
        mode_label = f"Mode {vis_idx + 1}" if test_mode == "eigenmode" else f"Superposition sample {vis_idx}"

        # label field for output overlay (on out_size canvas)
        amp2 = (amps_eval[vis_idx] ** 2).astype(np.float32)
        energy = amp2 / (amp2.sum() + 1e-12)
        label_field = np.zeros((out_size, out_size), dtype=np.float32)
        for m in range(num_modes_valid):
            li_start = m * L
            for l in range(L):
                label_field += energy[m] * label_patterns_np[:, :, li_start + l]

        # -- collect propagation frames ----------------------------------
        z_step_m = z_step_um * 1e-6
        frames = self._collect_multiwl_propagation_frames(
            model=model, eigenmode_field=eigen_field,
            layer_size=layer_size, z_input_to_first=z_input_to_first,
            z_layers=z_layers, z_prop=z_prop, pixel_size=pixel_size,
            wavelengths=wavelengths, base_wavelength_idx=base_wavelength_idx,
            z_step_m=z_step_m, mode_label=mode_label, device=device,
            max_frames=80, out_size=out_size, padding_ratio_out=padding_ratio_out,
        )
        for f in frames:
            if f["type"] == "output":
                f["label"] = label_field

        # player frames (target ~40 field frames)
        total_z_m = z_input_to_first + num_layers * z_layers + z_prop
        player_frames = self._collect_multiwl_propagation_frames(
            model=model, eigenmode_field=eigen_field,
            layer_size=layer_size, z_input_to_first=z_input_to_first,
            z_layers=z_layers, z_prop=z_prop, pixel_size=pixel_size,
            wavelengths=wavelengths, base_wavelength_idx=base_wavelength_idx,
            z_step_m=z_step_um * 1e-6, mode_label=mode_label, device=device,
            max_frames=40, out_size=out_size, padding_ratio_out=padding_ratio_out,
        )
        for f in player_frames:
            if f["type"] == "output":
                f["label"] = label_field

        return {
            "frames": frames,
            "player_frames": player_frames,
            "metrics": metrics_result,
            "model_meta": meta,
            "evaluation_regions": eval_regions,
            "detect_radius": circle_detectsize,
            "mode_index": vis_idx,
            "test_mode": test_mode,
            "label_field": label_field,
            "num_modes": num_modes_valid,
            "num_wavelengths": L,
            "wavelengths_nm": (wavelengths * 1e9).tolist(),
        }

    # ------------------------------------------------------------------
    # Propagation frame collection (multi-wavelength)
    # ------------------------------------------------------------------

    def _collect_multiwl_propagation_frames(
        self,
        model: Any,
        eigenmode_field: Any,
        *,
        layer_size: int,
        z_input_to_first: float,
        z_layers: float,
        z_prop: float,
        pixel_size: float,
        wavelengths: np.ndarray,
        base_wavelength_idx: int,
        z_step_m: float,
        mode_label: str = "input",
        device: Any = None,
        max_frames: int = 100,
        out_size: int | None = None,
        padding_ratio_out: float | None = None,
    ) -> List[Dict[str, Any]]:
        """Walk the multi-wavelength D2NN optical path and collect snapshots.

        Frame dict keys:
            key          str   unique identifier
            description  str   human-readable label
            z_um         float absolute z-position in micrometres
            intensity_wl np.ndarray | None  (L, H, W) float32 — all wavelengths
            intensity    np.ndarray | None  (H, W) float32 — base wavelength
            phase        np.ndarray | None  (H, W) float32  [0, 2pi]
            type         str   'field' | 'mask' | 'output'
            label        np.ndarray | None  label overlay for output frames
        """
        import torch
        from odnn_training_visualization import (
            _make_kz_stack_multiwl,
            _propagate_multiwl_kz,
            _complex_pad_blhw,
            _complex_crop_blhw,
        )
        from odnn_processing import pad_field_to_layer

        if device is None:
            device = next(model.parameters()).device

        L = int(len(wavelengths))
        base_idx = int(np.clip(base_wavelength_idx, 0, L - 1))
        wls = np.asarray(wavelengths, dtype=np.float32)

        # helpers --------------------------------------------------------
        def _z_fractions(z_total: float, z_step: float) -> List[float]:
            if z_total <= 0 or z_step <= 0:
                return []
            n = max(1, int(z_total / z_step))
            return [i / n for i in range(1, n)]

        def _extract_intensity(t: Any) -> np.ndarray:
            arr = t.detach().cpu().numpy()
            return (np.abs(arr[0]) ** 2).astype(np.float32)  # (L, H, W) from (1, L, H, W)

        def _propagate_pad_crop(E_blhw, kz_lnn, pad, H, W, z_dist):
            """Pad, propagate, crop in one call (only when pad>0)."""
            big = _complex_pad_blhw(E_blhw, pad)
            out = _propagate_multiwl_kz(big, kz_lnn, z_dist)
            return _complex_crop_blhw(out, H, W, pad)

        frames: List[Dict[str, Any]] = []
        current_z = 0.0

        def _add_field(key: str, desc: str, tensor, z_m: float, ftype: str = "field"):
            i_wl = _extract_intensity(tensor)  # (L, H, W)
            i_base = i_wl[base_idx]            # (H, W)
            frames.append({
                "key": key, "description": desc, "z_um": z_m * 1e6,
                "intensity_wl": i_wl, "intensity": i_base,
                "phase": None, "type": ftype,
            })

        def _add_mask(layer_idx: int, phase_np: np.ndarray, z_m: float):
            frames.append({
                "key": f"mask_{layer_idx + 1}",
                "description": f"Layer {layer_idx + 1} — Phase Mask",
                "z_um": z_m * 1e6,
                "intensity_wl": None, "intensity": None,
                "phase": phase_np.astype(np.float32),
                "type": "mask",
            })

        # -- input plane -------------------------------------------------
        ef = eigenmode_field.to(device=device, dtype=torch.complex64)
        while ef.ndim > 2:
            ef = ef.squeeze(0)
        field = padded[None, None, ...].repeat(1, L, 1, 1).contiguous()  # (1, L, H, W)
        H, W = int(field.shape[-2]), int(field.shape[-1])

        # clamp z_step to respect max_frames
        total_z = z_input_to_first + len(model.layers) * z_layers + z_prop
        min_z_step = total_z / max(max_frames, 5)
        if z_step_m < min_z_step:
            z_step_m = min_z_step

        _add_field("input", f"Input — {mode_label}", field, current_z)

        # -- pre-propagation fractions -----------------------------------
        pre = model.pre_propagation
        pre_pad = int(pre.pad_px) if hasattr(pre, "pad_px") else 0
        pre_kz = pre.kz_pad if (pre_pad > 0 and hasattr(pre, "kz_pad") and pre.kz_pad is not None) else pre.kz_base

        for frac in _z_fractions(z_input_to_first, z_step_m):
            z_snap = z_input_to_first * frac
            if pre_pad > 0:
                snap = _propagate_pad_crop(field, pre_kz, pre_pad, H, W, z_snap)
            else:
                snap = _propagate_multiwl_kz(field, pre_kz, z_snap)
            _add_field(f"pre_{frac:.2f}", f"z = {(z_snap)*1e6:.0f} um", snap, z_snap)

        field = model.pre_propagation(field)
        current_z += z_input_to_first

        # -- each diffraction layer --------------------------------------
        for li, layer in enumerate(model.layers):
            _add_field(f"L{li + 1}_arr", f"Arrival at layer {li + 1}", field, current_z)

            # phase mask
            phase_np = np.remainder(
                layer.phase.detach().cpu().numpy().astype(np.float32), 2 * np.pi,
            )
            _add_mask(li, phase_np, current_z)

            # apply wavelength-scaled phase mask
            ly_pad = int(layer.pad_px) if hasattr(layer, "pad_px") else 0
            ly_kz = layer.kz_pad if (ly_pad > 0 and layer.kz_pad is not None) else layer.kz_base
            scale = (layer.lam0 / layer.wavelengths).to(device)
            phi = layer.phase[None, :, :] * scale[:, None, None]
            phase_c = torch.exp(1j * phi).to(torch.complex64)

            if ly_pad > 0:
                Ein = _complex_pad_blhw(field, ly_pad)
                phase_big = torch.ones(L, H + 2 * ly_pad, W + 2 * ly_pad,
                                       dtype=torch.complex64, device=device)
                phase_big[:, ly_pad:ly_pad + H, ly_pad:ly_pad + W] = phase_c
                Ein = Ein * phase_big[None]
                # Ein already padded — propagate and crop, don't re-pad
                for frac in _z_fractions(z_layers, z_step_m):
                    z_snap = z_layers * frac
                    snap = _propagate_multiwl_kz(Ein, ly_kz, z_snap)
                    snap = _complex_crop_blhw(snap, H, W, ly_pad)
                    _add_field(f"L{li + 1}_prop_{frac:.2f}",
                               f"z = {(current_z + z_snap)*1e6:.0f} um",
                               snap, current_z + z_snap)
            else:
                Ein = field * phase_c[None]
                for frac in _z_fractions(z_layers, z_step_m):
                    z_snap = z_layers * frac
                    snap = _propagate_multiwl_kz(Ein, ly_kz, z_snap)
                    _add_field(f"L{li + 1}_prop_{frac:.2f}",
                               f"z = {(current_z + z_snap)*1e6:.0f} um",
                               snap, current_z + z_snap)

            field = layer(field)
            current_z += z_layers

        # -- embed to out_size canvas if different -------------------------
        if out_size is not None and out_size != H:
            field = model._embed_to_out_canvas(field)
            H_out, W_out = int(field.shape[-2]), int(field.shape[-1])
        else:
            H_out, W_out = H, W

        # -- last layer -> output fractions -------------------------------
        prop = model.propagation
        prop_pad = int(prop.pad_px) if hasattr(prop, "pad_px") else 0
        prop_kz = prop.kz_pad if (prop_pad > 0 and hasattr(prop, "kz_pad") and prop.kz_pad is not None) else prop.kz_base

        for frac in _z_fractions(z_prop, z_step_m):
            z_snap = z_prop * frac
            if prop_pad > 0:
                snap = _propagate_pad_crop(field, prop_kz, prop_pad, H_out, W_out, z_snap)
            else:
                snap = _propagate_multiwl_kz(field, prop_kz, z_snap)
            _add_field(f"out_prop_{frac:.2f}",
                       f"z = {(current_z + z_snap)*1e6:.0f} um",
                       snap, current_z + z_snap)

        # -- final output (detector) -------------------------------------
        output = model.propagation(field)
        current_z += z_prop
        output_intensity = (output.detach().abs() ** 2).cpu().numpy().astype(np.float32)
        i_output_2d = output_intensity[0, base_idx]  # (H_out, W_out)
        i_output_wl = output_intensity[0]             # (L, H_out, W_out)
        frames.append({
            "key": "output", "description": "Detector Output",
            "z_um": current_z * 1e6,
            "intensity_wl": i_output_wl, "intensity": i_output_2d,
            "phase": None, "type": "output", "label": None,
        })

        return frames


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _build_evaluation_regions(
    *, H: int, W: int, num_modes: int, num_wavelengths: int, radius: int, margin_ratio: float = 0.2,
) -> List[tuple]:
    """Build flat evaluation_regions list: regions[mode * L + wavelength]."""
    mx = max(int(W * margin_ratio), radius + 5)
    my = max(int(H * margin_ratio), radius + 5)
    xs = np.linspace(mx, W - 1 - mx, num_wavelengths)
    ys = np.linspace(my, H - 1 - my, num_modes)

    regions = []
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wavelengths):
            cx = int(round(xs[wl_idx]))
            cy = int(round(ys[mode_idx]))
            x0 = max(0, cx - radius)
            x1 = min(W, cx + radius)
            y0 = max(0, cy - radius)
            y1 = min(H, cy + radius)
            regions.append((x0, x1, y0, y1))
    return regions


def _build_label_patterns(
    *, H: int, W: int, num_modes: int, num_wavelengths: int, radius: int, margin_ratio: float = 0.2,
) -> tuple:
    """Return (patterns_np, regions) matching generate_detector_patterns_multiwl output."""

    mx = max(int(W * margin_ratio), radius + 5)
    my = max(int(H * margin_ratio), radius + 5)
    xs = np.linspace(mx, W - 1 - mx, num_wavelengths)
    ys = np.linspace(my, H - 1 - my, num_modes)

    total = num_modes * num_wavelengths
    patterns = np.zeros((H, W, total), dtype=np.float32)
    regions = []
    idx = 0
    for mode_idx in range(num_modes):
        for wl_idx in range(num_wavelengths):
            cx = int(round(xs[wl_idx]))
            cy = int(round(ys[mode_idx]))
            yy, xx = np.ogrid[:H, :W]
            patterns[:, :, idx] = (
                (yy - cy) ** 2 + (xx - cx) ** 2 <= radius ** 2
            ).astype(np.float32)
            x0 = max(0, cx - radius)
            x1 = min(W, cx + radius)
            y0 = max(0, cy - radius)
            y1 = min(H, cy + radius)
            regions.append((x0, x1, y0, y1))
            idx += 1
    return patterns, regions
