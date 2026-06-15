import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
TEMP_DIR = FRONTEND_DIR / "temp"
LOG_DIR = FRONTEND_DIR / "logs"

_SWEEP_CONFIG_PATH = TEMP_DIR / "sweep_config.json"
_SWEEP_LOG = LOG_DIR / "sweep.log"
_SWEEP_METRICS = LOG_DIR / "sweep_metrics.jsonl"


class SweepAdapter:
    """Manages the batch_sweep.py subprocess for the Analysis page.

    Follows the same subprocess / JSONL pattern as Mainfor6Adapter.
    """

    def __init__(self) -> None:
        TEMP_DIR.mkdir(parents=True, exist_ok=True)
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def load_default_sweep_config(self) -> Dict[str, Any]:
        return {
            "num_modes_list": [10, 20, 30],
            "num_layers_from": 1,
            "num_layers_to": 15,
            "num_layers_step": 1,
            "layer_size_list": [200],
            "epochs": 500,
            "batch_size": 16,
            "lr": 1.99,
            "lr_gamma": 0.99,
            "wavelength_nm": 1568,
            "z_layers_um": 40,
            "z_prop_um": 120,
            "z_input_to_first_um": 40,
            "pixel_size_um": 1.0,
        }

    # ------------------------------------------------------------------
    # Subprocess control (mirrors mainfor6_adapter.py pattern)
    # ------------------------------------------------------------------

    def start_sweep(self, config: Dict[str, Any], mat_file: str) -> int:
        """Write sweep config JSON, launch batch_sweep.py, return PID."""
        if not mat_file:
            raise ValueError("mat_file path is required to start sweep.")

        # convert UI config to backend config
        backend_cfg = self._ui_to_backend(config, mat_file)

        with open(_SWEEP_CONFIG_PATH, "w") as f:
            json.dump(backend_cfg, f, indent=2)

        with open(_SWEEP_LOG, "w") as log_fh:
            proc = subprocess.Popen(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "batch_sweep.py"),
                    "--config", str(_SWEEP_CONFIG_PATH),
                ],
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
            )
        return proc.pid

    def stop_sweep(self, pid: int) -> None:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    def is_sweep_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def read_sweep_log_tail(self, n: int = 50) -> List[str]:
        if not _SWEEP_LOG.exists():
            return []
        try:
            lines = _SWEEP_LOG.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            return lines[-n:]
        except OSError:
            return []

    def parse_sweep_metrics(self) -> pd.DataFrame:
        """Read sweep_metrics.jsonl into DataFrame."""
        if not _SWEEP_METRICS.exists():
            return pd.DataFrame()
        records: List[dict] = []
        try:
            with open(_SWEEP_METRICS, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        except OSError:
            return pd.DataFrame()
        if not records:
            return pd.DataFrame()
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Config translation
    # ------------------------------------------------------------------

    def _ui_to_backend(
        self, ui_cfg: Dict[str, Any], mat_file: str
    ) -> Dict[str, Any]:
        """Convert analysis-page config to batch_sweep.py config."""
        from_val = int(ui_cfg.get("num_layers_from", 1))
        to_val = int(ui_cfg.get("num_layers_to", 15))
        step_val = int(ui_cfg.get("num_layers_step", 1))
        num_layers_list = list(range(from_val, to_val + 1, step_val))

        return {
            "mat_file": mat_file,
            "num_modes_list": ui_cfg["num_modes_list"],
            "num_layers_list": num_layers_list,
            "layer_size_list": ui_cfg["layer_size_list"],
            "field_size": ui_cfg.get("field_size", 100),
            "epochs": int(ui_cfg.get("epochs", 500)),
            "batch_size": int(ui_cfg.get("batch_size", 16)),
            "lr": float(ui_cfg.get("lr", 1.99)),
            "lr_gamma": float(ui_cfg.get("lr_gamma", 0.99)),
            "wavelength_nm": float(ui_cfg.get("wavelength_nm", 1568)),
            "z_layers_um": float(ui_cfg.get("z_layers_um", 40)),
            "z_prop_um": float(ui_cfg.get("z_prop_um", 120)),
            "z_input_to_first_um": float(
                ui_cfg.get("z_input_to_first_um", 40)
            ),
            "pixel_size_um": float(ui_cfg.get("pixel_size_um", 1.0)),
            "label_pattern_mode": "circle",
            "phase_option": 4,
            "output_dir": "results/sweep",
            "metrics_log": str(_SWEEP_METRICS),
            "log_file": str(_SWEEP_LOG),
        }
