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

# Root of the NC_version project (one level above frontend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

FRONTEND_DIR = PROJECT_ROOT / "frontend"
TEMP_DIR     = FRONTEND_DIR / "temp"
LOG_DIR      = FRONTEND_DIR / "logs"
RESULTS_DIR  = PROJECT_ROOT / "results"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"

_TRAINING_LOG = LOG_DIR / "training.log"
_CONFIG_PATH  = TEMP_DIR / "train_config.json"


def _ensure_backend_on_path() -> None:
    """Make sure PROJECT_ROOT is importable (backend modules live there)."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


class Mainfor6Adapter(BaseODNNAdapter):
    """Adapter for mainfor6.py.

    Training is launched as a non-blocking subprocess; testing is done
    in-process (synchronous) so that field arrays can be returned directly
    to the Streamlit page for inline rendering.
    """

    def __init__(self) -> None:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def load_default_config(self) -> Dict[str, Any]:
        """Canonical defaults matching mainfor6.py hard-coded values.

        This is the single source of truth -- pages must not define their
        own parallel DEFAULTS dict.
        """
        return {
            # geometry
            "layer_size": 110,
            "num_layers": 3,
            "field_size": 25,
            # physics
            "wavelength_nm": 1568,
            "z_layers_um": 40,
            "z_prop_um": 120,
            "z_input_to_first_um": 40,
            "pixel_size_um": 1,
            # modes
            "num_modes": 6,
            "phase_option": 4,
            # dataset
            "num_data": 1000,
            "batch_size": 16,
            "training_dataset_mode": "eigenmode",
            "label_pattern_mode": "circle",
            # training
            "epochs": 1000,
            "learning_rate": 1.99,
            "lr_gamma": 0.99,
            # evaluation
            "evaluation_mode": "eigenmode",
            "num_superposition_eval_samples": 1000,
        }

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def start_training(self, config: Dict[str, Any], mat_file: str = "") -> int:
        """Write config to disk and launch mainfor6.py as a subprocess.

        Returns:
            PID of the spawned process.

        Raises:
            ValueError: if mat_file is empty.
        """
        if not mat_file:
            raise ValueError("mat_file path is required to start training.")

        with open(_CONFIG_PATH, "w") as f:
            json.dump(config, f, indent=2)

        # open log; parent fd closes after Popen, child keeps its copy
        with open(_TRAINING_LOG, "w") as log_fh:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "mainfor6.py"),
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
        """Return the last *n* lines of training.log for error surfacing."""
        if not _TRAINING_LOG.exists():
            return []
        try:
            lines = _TRAINING_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
            return lines[-n:]
        except OSError:
            return []

    # ------------------------------------------------------------------
    # Checkpoint discovery
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[str]:
        """Return sorted list of absolute .pth paths in checkpoints/."""
        if not CHECKPOINTS_DIR.exists():
            return []
        return sorted(str(p) for p in CHECKPOINTS_DIR.glob("*.pth"))

    def load_checkpoint_meta(self, pth_path: str) -> Dict[str, Any]:
        """Read only the 'meta' dict from a checkpoint (weights not loaded).

        Returns an empty dict on any error.
        """
        _ensure_backend_on_path()
        try:
            import torch
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
        """Load checkpoint, run evaluation, collect propagation frames.

        All computation runs in the calling process (synchronous).  The
        caller should wrap this with st.spinner to avoid UI freeze.

        Returns:
            dict with keys:
                'frames'             -- list of frame dicts for the timeline
                'metrics'            -- scalar evaluation metrics
                'model_meta'         -- checkpoint meta dict
                'evaluation_regions' -- list of (x0,x1,y0,y1) tuples
                'detect_radius'      -- int
                'mode_index'         -- int (eigenmode mode only)
                'test_mode'          -- str
                'label_field'        -- np.ndarray label for the visualised sample
        """
        _ensure_backend_on_path()

        import torch
        import math as _math
        from torch.utils.data import DataLoader, TensorDataset

        from odnn_model import D2NNModel, complex_pad, complex_crop
        from odnn_io import load_complex_modes_from_mat
        from ODNN_functions import create_evaluation_regions, generate_complex_weights
        from odnn_training_eval import evaluate_spot_metrics, build_superposition_eval_context
        from odnn_generate_label import (
            compute_label_centers,
            compose_labels_from_patterns,
            generate_detector_patterns,
        )
        from odnn_processing import prepare_sample, pad_field_to_layer

        # -- load model --------------------------------------------------
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        meta = ckpt.get("meta", {})

        num_layers       = int(meta.get("num_layers", config.get("num_layers", 3)))
        layer_size       = int(meta.get("layer_size", config.get("layer_size", 110)))
        field_size       = int(meta.get("field_size", config.get("field_size", 25)))
        num_modes        = int(meta.get("num_modes", config.get("num_modes", 6)))
        wavelength       = float(meta.get("wavelength", config.get("wavelength_nm", 1568) * 1e-9))
        z_layers         = float(meta.get("z_layers", config.get("z_layers_um", 40) * 1e-6))
        z_prop           = float(meta.get("z_prop", config.get("z_prop_um", 120) * 1e-6))
        z_input_to_first = float(meta.get("z_input_to_first", config.get("z_input_to_first_um", 40) * 1e-6))
        pixel_size       = float(meta.get("pixel_size", config.get("pixel_size_um", 1) * 1e-6))
        padding_ratio    = float(meta.get("padding_ratio", 0.5))

        device = torch.device("cpu")
        model = D2NNModel(
            num_layers=num_layers,
            layer_size=layer_size,
            z_layers=z_layers,
            z_prop=z_prop,
            pixel_size=pixel_size,
            wavelength=wavelength,
            device=device,
            padding_ratio=padding_ratio,
            z_input_to_first=z_input_to_first,
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()

        # -- load eigenmodes from .mat ----------------------------------
        eigenmodes_raw = load_complex_modes_from_mat(mat_file, key="modes_field")
        # eigenmodes_raw: (H, W, M) complex
        if eigenmodes_raw.shape[2] < num_modes:
            raise ValueError(
                f"Mat file only has {eigenmodes_raw.shape[2]} modes, "
                f"but model needs {num_modes}."
            )
        mmf_data_np = eigenmodes_raw[:, :, :num_modes].transpose(2, 0, 1)
        amp_min = np.min(np.abs(mmf_data_np))
        amp_max = np.max(np.abs(mmf_data_np))
        mmf_data_norm = (np.abs(mmf_data_np) - amp_min) / (amp_max - amp_min + 1e-12)
        mmf_data_np = mmf_data_norm * np.exp(1j * np.angle(mmf_data_np))
        mmf_data_ts = torch.from_numpy(mmf_data_np.astype(np.complex64))

        # -- test / eval configuration ----------------------------------
        phase_option       = int(config.get("phase_option", 4))
        batch_size         = int(config.get("batch_size", 16))
        test_mode          = str(config.get("test_mode", "eigenmode"))
        mode_index         = int(config.get("mode_index", 0))
        label_pattern_mode = str(config.get("label_pattern_mode", "circle"))

        num_superposition_eval_samples = int(config.get("num_superposition_eval_samples", 1000))
        superposition_seed = int(config.get("superposition_seed", 20240116))
        z_step_m = float(config.get("z_step_um", 5)) * 1e-6

        # -- build amplitudes / phases ----------------------------------
        if phase_option == 4:
            amplitudes = np.eye(num_modes, dtype=np.float32)
            phases     = np.eye(num_modes, dtype=np.float32)
        else:
            amplitudes, phases = generate_complex_weights(num_modes, num_modes, phase_option)
        amplitudes_phases = np.hstack((amplitudes, phases[:, 1:] / (2 * np.pi)))

        # -- build label patterns (detector circles / eigenmode patterns)
        circle_focus_radius = 5
        circle_detectsize   = 10
        num_detector        = num_modes

        if label_pattern_mode == "eigenmode":
            pattern_stack = np.transpose(np.abs(mmf_data_np), (1, 2, 0))
            layout_radius    = _math.ceil(max(pattern_stack.shape[0], pattern_stack.shape[1]) / 2)
            focus_radius     = 12
            detectsize       = 15
        else:
            circle_radius = circle_focus_radius
            ps = circle_radius * 2
            if ps % 2 == 0:
                ps += 1
            pattern_stack = generate_detector_patterns(ps, ps, num_detector, shape="circle")
            layout_radius = circle_radius
            focus_radius  = circle_focus_radius
            detectsize    = circle_detectsize

        centers, _, _ = compute_label_centers(layer_size, layer_size, num_detector, layout_radius)
        mode_label_maps = [
            compose_labels_from_patterns(layer_size, layer_size, pattern_stack, centers, Index=i + 1)
            for i in range(num_detector)
        ]
        MMF_Label_data = torch.from_numpy(
            np.stack(mode_label_maps, axis=2).astype(np.float32)
        )

        # -- build test dataset ------------------------------------------
        if test_mode == "eigenmode":
            num_train = num_modes if phase_option == 4 else amplitudes.shape[0]
            amp_ts = torch.from_numpy(amplitudes[:num_train].astype(np.float32))
            energy_weights = amp_ts ** 2
            combined_labels = (
                energy_weights[:, None, None, :] * MMF_Label_data.unsqueeze(0)
            ).sum(dim=3)
            label_data = torch.zeros([num_train, 1, layer_size, layer_size])
            label_data[:, 0, :, :] = combined_labels

            complex_weights = amplitudes[:num_train] * np.exp(1j * phases[:num_train])
            from ODNN_functions import generate_fields_ts
            cw_ts = torch.from_numpy(complex_weights.astype(np.complex64))
            image_data = generate_fields_ts(cw_ts, mmf_data_ts, num_train, num_modes, field_size).to(torch.complex64)

            dataset_pairs = [prepare_sample(image_data[i], label_data[i], layer_size) for i in range(num_train)]
            images = torch.stack([p[0] for p in dataset_pairs])
            labels = torch.stack([p[1] for p in dataset_pairs])
            tensor_dataset = TensorDataset(images, labels)
            test_loader = DataLoader(tensor_dataset, batch_size=batch_size, shuffle=False)
            eval_amplitudes        = amplitudes[:num_train]
            eval_amplitudes_phases = amplitudes_phases[:num_train]
            eval_phases            = phases[:num_train]
            image_test_data        = image_data

        else:  # superposition
            ctx = build_superposition_eval_context(
                num_superposition_eval_samples,
                num_modes=num_modes,
                field_size=field_size,
                layer_size=layer_size,
                mmf_modes=mmf_data_ts,
                mmf_label_data=MMF_Label_data,
                batch_size=batch_size,
                second_mode_half_range=True,
                rng_seed=superposition_seed,
            )
            test_loader            = ctx["loader"]
            eval_amplitudes        = ctx["amplitudes"]
            eval_amplitudes_phases = ctx["amplitudes_phases"]
            eval_phases            = ctx["phases"]
            image_test_data        = ctx["image_data"]
            label_data             = ctx["tensor_dataset"].tensors[1]

        # -- evaluation regions -------------------------------------------
        evaluation_regions = create_evaluation_regions(
            layer_size, layer_size, num_detector, focus_radius, detectsize
        )

        # -- run evaluation metrics ---------------------------------------
        metrics_raw = evaluate_spot_metrics(
            model,
            test_loader,
            evaluation_regions,
            detect_radius=detectsize,
            device=device,
            pred_case=1,
            num_modes=num_modes,
            phase_option=phase_option,
            amplitudes=eval_amplitudes,
            amplitudes_phases=eval_amplitudes_phases,
            phases=eval_phases,
            mmf_modes=mmf_data_ts,
            field_size=field_size,
            image_test_data=image_test_data,
        )

        metrics = {
            "avg_relative_amp_err": float(metrics_raw.get("avg_relative_amp_err", float("nan"))),
            "avg_amplitudes_diff":  float(metrics_raw.get("avg_amplitudes_diff", float("nan"))),
            "snr_ratio_full":       float(metrics_raw.get("snr_ratio_full", float("nan"))),
            "snr_db_full":          float(metrics_raw.get("snr_db_full", float("nan"))),
            "throughput":           float(metrics_raw.get("throughput", float("nan"))),
        }

        # -- read per-mode isolation matching this checkpoint ----------
        _iso_dir = RESULTS_DIR / "wavelength_analysis"
        _ckpt_nl = int(meta.get("num_layers", 3))
        _ckpt_nm = int(meta.get("num_modes", num_modes))
        _ckpt_ls = int(meta.get("layer_size", layer_size))
        _pattern = f"per_mode_isolation_m{_ckpt_nm}_ls{_ckpt_ls}_L{_ckpt_nl}_*.mat"
        _mat_candidates = sorted(_iso_dir.glob(_pattern))
        if _mat_candidates:
            try:
                from scipy.io import loadmat as _loadmat
                _iso_data  = _loadmat(str(_mat_candidates[-1]))
                _wl_nm_arr = np.array(_iso_data["wavelength_nm"]).flatten()
                _train_nm  = wavelength * 1e9
                _row_idx   = int(np.argmin(np.abs(_wl_nm_arr - _train_nm)))
                _iso_db    = np.array(_iso_data["isolation_db"])[_row_idx].tolist()
                _iso_pct   = np.array(_iso_data["isolation_percent"])[_row_idx].tolist()
                metrics["isolation_db_per_mode"]  = _iso_db
                metrics["isolation_pct_per_mode"] = _iso_pct
                metrics["isolation_db_mean"]      = float(np.mean(_iso_db))
                metrics["isolation_pct_mean"]     = float(np.mean(_iso_pct))
            except Exception:
                pass  # missing scipy or malformed file — skip silently

        # -- pick sample for propagation visualisation --------------------
        if test_mode == "eigenmode":
            idx = max(0, min(mode_index, num_modes - 1))
            eigenmode_field = mmf_data_ts[idx]   # (field_size, field_size) complex
            label_field_np  = label_data[idx, 0].detach().cpu().numpy()
        else:
            vis_idx = int(config.get("superposition_vis_sample", 0))
            vis_idx = max(0, min(vis_idx, num_superposition_eval_samples - 1))
            eigenmode_field = image_test_data[vis_idx]  # (1, field_size, field_size)
            if eigenmode_field.dim() == 3:
                eigenmode_field = eigenmode_field[0]    # -> (field_size, field_size)
            label_field_np  = label_data[vis_idx, 0].detach().cpu().numpy()

        mode_label_str = (
            f"mode {mode_index + 1}" if test_mode == "eigenmode" else "superposition sample"
        )

        # -- collect propagation frames (inline, no disk writes) ---------
        frames = self._collect_propagation_frames(
            model=model,
            eigenmode_field=eigenmode_field,
            layer_size=layer_size,
            z_input_to_first=z_input_to_first,
            z_layers=z_layers,
            z_prop=z_prop,
            z_step_m=z_step_m,
            mode_label=mode_label_str,
        )

        # attach label to the last frame (output frame)
        for frame in frames:
            if frame["type"] == "output":
                frame["label"] = label_field_np

        # -- collect player frames (fixed target frame count) -------------
        # Re-run collection with a z_step computed to hit ~player_target_frames
        # field frames.  Only the z_step differs; model and field are the same.
        player_target = 40
        total_z_m = z_input_to_first + num_layers * z_layers + z_prop
        z_step_play = max(1e-8, total_z_m / max(player_target, 5))
        player_frames = self._collect_propagation_frames(
            model=model,
            eigenmode_field=eigenmode_field,
            layer_size=layer_size,
            z_input_to_first=z_input_to_first,
            z_layers=z_layers,
            z_prop=z_prop,
            z_step_m=z_step_play,
            mode_label=mode_label_str,
        )
        for frame in player_frames:
            if frame["type"] == "output":
                frame["label"] = label_field_np

        return {
            "frames":             frames,
            "player_frames":      player_frames,
            "metrics":            metrics,
            "model_meta":         meta,
            "evaluation_regions": evaluation_regions,
            "detect_radius":      detectsize,
            "mode_index":         mode_index if test_mode == "eigenmode" else -1,
            "test_mode":          test_mode,
            "label_field":        label_field_np,
        }

    # ------------------------------------------------------------------
    # Propagation frame collection
    # ------------------------------------------------------------------

    def _collect_propagation_frames(
        self,
        model: Any,
        eigenmode_field: Any,
        *,
        layer_size: int,
        z_input_to_first: float,
        z_layers: float,
        z_prop: float,
        z_step_m: float,
        mode_label: str = "input",
    ) -> List[Dict[str, Any]]:
        """Walk through the D2NN optical path and collect intensity/phase snapshots.

        Sampling is driven by z_step_m (physical distance).  Mask frames are
        inserted at each diffraction layer.  No files are written.

        Frame dict keys:
            key         str   unique identifier
            description str   human-readable label for the UI
            z_um        float absolute z-position in micrometres
            intensity   np.ndarray | None   shape (H, W), float32
            phase       np.ndarray | None   shape (H, W), float32  [0, 2pi]
            type        str   'field' | 'mask' | 'output'
        """
        import torch
        from odnn_model import complex_pad, complex_crop
        from odnn_processing import pad_field_to_layer

        device = next(model.parameters()).device

        def z_to_fractions(z_total: float, z_step: float):
            if z_total <= 0 or z_step <= 0:
                return []
            n = max(1, int(z_total / z_step))
            return [i / n for i in range(1, n)]

        def field_to_intensity(t: Any) -> np.ndarray:
            arr = t.detach().squeeze().cpu().numpy()
            return (np.abs(arr) ** 2).astype(np.float32)

        def propagate_partial(plane_2d, kz, pad_px, units, z_dist):
            """Propagate a (H, W) complex tensor by z_dist and return (H, W) tensor."""
            if pad_px > 0:
                big = complex_pad(plane_2d, pad_px, pad_px)
                out = model.pre_propagation._propagate(big, kz, z_dist)
                return complex_crop(out, units, units, pad_px, pad_px)
            return model.pre_propagation._propagate(plane_2d, kz, z_dist)

        frames: List[Dict[str, Any]] = []
        current_z = 0.0

        def add_field(key, desc, t, z_m, ftype="field"):
            frames.append({
                "key":         key,
                "description": desc,
                "z_um":        z_m * 1e6,
                "intensity":   field_to_intensity(t),
                "phase":       None,
                "type":        ftype,
            })

        def add_mask(layer_idx, z_m, phase_np):
            frames.append({
                "key":         f"mask_{layer_idx + 1}",
                "description": f"Layer {layer_idx + 1} — Phase Mask",
                "z_um":        z_m * 1e6,
                "intensity":   None,
                "phase":       phase_np.astype(np.float32),
                "type":        "mask",
            })

        # --- input plane ------------------------------------------------
        ef = eigenmode_field.to(device=device, dtype=torch.complex64)
        padded = pad_field_to_layer(ef, layer_size)
        field = padded.unsqueeze(0).unsqueeze(0)   # (1,1,H,W)
        add_field("input", f"Input — {mode_label}", field, current_z)

        # --- input -> layer 1 propagation fractions ---------------------
        pre = model.pre_propagation
        pad = int(pre.pad_px)
        units = int(pre.units)
        plane = field.squeeze(1)   # (1,H,W)

        for frac in z_to_fractions(z_input_to_first, z_step_m):
            z_snap = z_input_to_first * frac
            if pad > 0:
                big = complex_pad(plane, pad, pad)
                snap = pre._propagate(big, pre.kz_pad, z_snap)
                snap = complex_crop(snap, units, units, pad, pad)
            else:
                snap = pre._propagate(plane, pre.kz_base, z_snap)
            add_field(
                f"pre_{frac:.2f}",
                f"z = {(current_z + z_snap)*1e6:.0f} um",
                snap,
                current_z + z_snap,
            )

        # propagate to layer 1
        field = model.pre_propagation(field)
        current_z += z_input_to_first

        # --- each diffraction layer -------------------------------------
        for li, layer in enumerate(model.layers):
            add_field(f"L{li+1}_arr", f"Arrival at layer {li + 1}", field, current_z)

            # phase mask
            phase_np = np.remainder(
                layer.phase.detach().cpu().numpy().astype(np.float32),
                2 * np.pi,
            )
            add_mask(li, current_z, phase_np)

            # apply mask then propagate with fractions
            pad  = int(layer.pad_px)
            lun  = int(layer.units)
            phase_c = torch.exp(
                1j * layer.phase.to(device=device, dtype=torch.float32)
            ).to(torch.complex64)

            plane = field.squeeze(1)   # (1,H,W) or (H,W)
            if pad > 0:
                big = complex_pad(plane, pad, pad)
                mask_big = torch.ones(
                    lun + 2 * pad, lun + 2 * pad,
                    dtype=torch.complex64, device=device,
                )
                mask_big[pad:pad+lun, pad:pad+lun] = phase_c
                f_masked = big * mask_big
                kz = layer.kz_pad
            else:
                f_masked = plane * phase_c
                kz = layer.kz_base

            for frac in z_to_fractions(z_layers, z_step_m):
                z_snap = z_layers * frac
                snap = layer._propagate(f_masked, kz, z_snap)
                if pad > 0:
                    snap = complex_crop(snap, lun, lun, pad, pad)
                add_field(
                    f"L{li+1}_prop_{frac:.2f}",
                    f"z = {(current_z + z_snap)*1e6:.0f} um",
                    snap,
                    current_z + z_snap,
                )

            field = layer(field)
            current_z += z_layers

        # --- last layer -> detector propagation fractions ---------------
        prop = model.propagation
        pad  = int(prop.pad_px)
        pun  = int(prop.units)
        plane = field.squeeze(1)

        if pad > 0:
            big = complex_pad(plane, pad, pad)
            kz  = prop.kz_pad
        else:
            big = plane
            kz  = prop.kz_base

        for frac in z_to_fractions(z_prop, z_step_m):
            z_snap = z_prop * frac
            snap = prop._propagate(big, kz, z_snap)
            if pad > 0:
                snap = complex_crop(snap, pun, pun, pad, pad)
            add_field(
                f"out_prop_{frac:.2f}",
                f"z = {(current_z + z_snap)*1e6:.0f} um",
                snap,
                current_z + z_snap,
            )

        # --- final output (detector) ------------------------------------
        output = model.propagation(field)
        current_z += z_prop
        # intensity via regression (square of abs)
        output_intensity = (
            output.detach().squeeze().cpu().numpy().real ** 2
            + output.detach().squeeze().cpu().numpy().imag ** 2
        ).astype(np.float32)
        frames.append({
            "key":         "output",
            "description": "Detector Output",
            "z_um":        current_z * 1e6,
            "intensity":   output_intensity,
            "phase":       None,
            "type":        "output",
            "label":       None,   # filled in by run_test
        })

        return frames
