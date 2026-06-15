"""Analysis page: batch sweep across (num_modes x layer_size x num_layers)."""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

_FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

from adapters.sweep_adapter import SweepAdapter
from services.config_manager import ConfigManager
from utils.time_utils import fmt_seconds

TEMP_DIR = _FRONTEND_DIR / "temp"
LOG_DIR = _FRONTEND_DIR / "logs"
CONFIG_PATH = TEMP_DIR / "sweep_config.json"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

_METRICS_PATH = LOG_DIR / "sweep_metrics.jsonl"


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "sweep_adapter" not in st.session_state:
        st.session_state.sweep_adapter = SweepAdapter()

    if "mat_file_path" not in st.session_state:
        st.session_state.mat_file_path = None
    if "mat_meta" not in st.session_state:
        st.session_state.mat_meta = {}  # {"field_size": 100, "max_modes": 103}

    if "sweep_config" not in st.session_state:
        defaults = st.session_state.sweep_adapter.load_default_sweep_config()
        mgr = ConfigManager(CONFIG_PATH)
        saved = mgr.load_config()
        st.session_state.sweep_config = mgr.merge_with_defaults(saved, defaults)

    if "sweep_pid" not in st.session_state:
        st.session_state.sweep_pid = None
    if "is_sweeping" not in st.session_state:
        st.session_state.is_sweeping = False
    if "sweep_error" not in st.session_state:
        st.session_state.sweep_error = None
    if "sweep_completed_count" not in st.session_state:
        st.session_state.sweep_completed_count = 0


# ---------------------------------------------------------------------------
# Section 1: Load Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _inspect_mat_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Read .mat file shape without loading full complex data.

    Returns {"field_size": H, "max_modes": M} or None on failure.
    """
    try:
        from odnn_io import load_complex_modes_from_mat
        modes = load_complex_modes_from_mat(file_path, key="modes_field")
        return {
            "field_size": int(modes.shape[0]),
            "max_modes": int(modes.shape[2]),
        }
    except Exception:
        return None


def _section_load_data() -> None:
    st.subheader("1. Load Data")

    uploaded = st.file_uploader("Select .mat mode file", type=["mat"])

    if uploaded is not None:
        dest = TEMP_DIR / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        st.session_state.mat_file_path = str(dest)

    if st.session_state.mat_file_path:
        p = Path(st.session_state.mat_file_path)
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        col1, col2 = st.columns(2)
        col1.success(f"Loaded: {p.name}")
        col2.caption(f"Size: {size_kb:.1f} KB")

        # auto-detect field_size and max_modes
        meta = _inspect_mat_file(st.session_state.mat_file_path)
        if meta:
            st.session_state.mat_meta = meta
            st.info(
                f"Detected: **{meta['max_modes']} modes**, "
                f"field size = **{meta['field_size']} x {meta['field_size']}**"
            )
        else:
            st.warning(
                "Could not inspect .mat file metadata. "
                "Ensure scipy/h5py/mat73 is installed in the backend environment."
            )
    else:
        st.info(
            "No file loaded. Upload a .mat file above, "
            "or enter the path manually."
        )
        manual = st.text_input(
            "Or enter path manually",
            placeholder="/path/to/your/mmf_file.mat",
        )
        if manual and Path(manual).exists():
            st.session_state.mat_file_path = manual
            st.rerun()
        elif manual:
            st.warning("File not found at the given path.")


# ---------------------------------------------------------------------------
# Helpers: parse comma-separated input
# ---------------------------------------------------------------------------

def _parse_int_list(text: str) -> List[int]:
    """Parse a comma-separated string of integers."""
    if not text.strip():
        return []
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return [int(p) for p in parts]


def _format_int_list(values: List[int]) -> str:
    return ", ".join(str(v) for v in values)


def _total_combinations(
    num_modes_list: List[int],
    layer_size_list: List[int],
    num_layers_from: int,
    num_layers_to: int,
    num_layers_step: int,
) -> int:
    n_layers = len(
        list(range(num_layers_from, num_layers_to + 1, num_layers_step))
    )
    return len(num_modes_list) * len(layer_size_list) * n_layers


# ---------------------------------------------------------------------------
# Section 2: Sweep Configuration
# ---------------------------------------------------------------------------

def _section_sweep_config() -> None:
    st.subheader("2. Sweep Configuration")

    cfg = st.session_state.sweep_config
    meta = st.session_state.mat_meta
    max_modes = meta.get("max_modes", 999)
    field_size = meta.get("field_size", "?")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Sweep ranges**")

        default_modes = _format_int_list(cfg.get("num_modes_list", [10, 20, 30]))
        modes_input = st.text_input(
            "num_modes",
            value=default_modes,
            placeholder="e.g. 10, 20, 30",
        )
        num_modes_list = _parse_int_list(modes_input)
        invalid_modes = [m for m in num_modes_list if m > max_modes]
        if invalid_modes:
            st.error(
                f"Values exceed max modes ({max_modes}): {invalid_modes}"
            )

        from_val = st.number_input(
            "num_layers: from", min_value=1, max_value=50,
            value=int(cfg.get("num_layers_from", 1)), step=1,
        )
        to_val = st.number_input(
            "num_layers: to", min_value=1, max_value=50,
            value=int(cfg.get("num_layers_to", 15)), step=1,
        )

        default_ls = _format_int_list(cfg.get("layer_size_list", [200]))
        ls_input = st.text_input(
            "layer_size",
            value=default_ls,
            placeholder="e.g. 200, 300",
        )
        layer_size_list = _parse_int_list(ls_input)

    with col2:
        st.markdown("**Training**")
        epochs = st.number_input(
            "Epochs", min_value=10, max_value=5000, step=50,
            value=int(cfg.get("epochs", 500)),
        )
        batch_size = st.number_input(
            "Batch size", min_value=1, max_value=256, step=4,
            value=int(cfg.get("batch_size", 16)),
        )
        lr = st.number_input(
            "Learning rate", min_value=0.001, max_value=10.0,
            format="%.3f", step=0.01,
            value=float(cfg.get("lr", 1.99)),
        )
        lr_gamma = st.number_input(
            "LR gamma", min_value=0.5, max_value=1.0,
            format="%.3f", step=0.005,
            value=float(cfg.get("lr_gamma", 0.99)),
        )

    with col3:
        st.markdown("**Preview**")
        total = _total_combinations(
            num_modes_list, layer_size_list, from_val, to_val, 1,
        )
        st.metric("Total combinations", total)
        st.caption(
            f"= {len(num_modes_list)} modes x {len(layer_size_list)} layer_sizes "
            f"x {to_val - from_val + 1} layers"
        )

    # --- update config ---
    st.session_state.sweep_config = {
        "num_modes_list": num_modes_list,
        "num_layers_from": from_val,
        "num_layers_to": to_val,
        "num_layers_step": 1,
        "layer_size_list": layer_size_list,
        "field_size": meta.get("field_size", 100),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "lr_gamma": lr_gamma,
        "wavelength_nm": cfg.get("wavelength_nm", 1568),
        "z_layers_um": cfg.get("z_layers_um", 40),
        "z_prop_um": cfg.get("z_prop_um", 120),
        "z_input_to_first_um": cfg.get("z_input_to_first_um", 40),
        "pixel_size_um": cfg.get("pixel_size_um", 1.0),
    }

    if st.button("Save Config", type="secondary"):
        ConfigManager(CONFIG_PATH).save_config(
            st.session_state.sweep_config
        )
        st.success(f"Config saved to {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Section 3: Run
# ---------------------------------------------------------------------------

def _do_start_sweep() -> None:
    cfg = st.session_state.sweep_config
    mat_path = st.session_state.mat_file_path
    adapter = st.session_state.sweep_adapter

    assert mat_path, "Start button must be disabled when no .mat file is loaded"

    _METRICS_PATH.write_text("")
    st.session_state.sweep_error = None
    st.session_state.sweep_completed_count = 0

    try:
        pid = adapter.start_sweep(cfg, mat_file=mat_path)
    except Exception as exc:
        st.session_state.sweep_error = str(exc)
        return

    st.session_state.sweep_pid = pid
    st.session_state.is_sweeping = True


def _do_stop_sweep() -> None:
    pid = st.session_state.sweep_pid
    if pid:
        st.session_state.sweep_adapter.stop_sweep(pid)
    st.session_state.is_sweeping = False
    st.session_state.sweep_pid = None


def _section_run() -> None:
    st.subheader("3. Run")

    adapter = st.session_state.sweep_adapter

    # keep sweeping flag in sync with actual process liveness
    if st.session_state.is_sweeping and st.session_state.sweep_pid:
        if not adapter.is_sweep_alive(st.session_state.sweep_pid):
            st.session_state.is_sweeping = False
            st.session_state.sweep_pid = None

    if st.session_state.is_sweeping:
        st_autorefresh(interval=2000, key="sweep_monitor")

    col_start, col_stop = st.columns([1, 1])
    with col_start:
        disabled = (
            st.session_state.is_sweeping
            or st.session_state.mat_file_path is None
        )
        if st.button("Start Sweep", type="primary", disabled=disabled):
            _do_start_sweep()
            st.rerun()
    with col_stop:
        if st.button("Stop", disabled=not st.session_state.is_sweeping):
            _do_stop_sweep()
            st.rerun()

    # --- status banner ---
    if st.session_state.sweep_error:
        st.error(
            f"Failed to start sweep: {st.session_state.sweep_error}"
        )
        return

    if st.session_state.is_sweeping:
        st.info(
            f"Sweep running (PID {st.session_state.sweep_pid})"
        )
        # NOTE: do NOT return here — fall through to show live progress + table

    # --- live progress (shown during AND after sweep) ---
    df = adapter.parse_sweep_metrics()
    if not df.empty:
        done_df = df[df["status"] == "done"]
        cfg = st.session_state.sweep_config
        total = _total_combinations(
            cfg.get("num_modes_list", []),
            cfg.get("layer_size_list", []),
            cfg.get("num_layers_from", 1),
            cfg.get("num_layers_to", 15),
            1,
        )
        if not done_df.empty:
            progress = min(1.0, len(done_df) / max(total, 1))
            st.progress(
                progress, text=f"Completed: {len(done_df)} / {total}"
            )
            # live results table
            display_cols = [
                "num_modes", "layer_size", "num_layers",
                "avg_relative_amp_err", "snr_db_full",
                "isolation_db_mean", "overflow_ratio", "elapsed_s",
            ]
            available = [c for c in display_cols if c in done_df.columns]
            if available:
                st.dataframe(
                    done_df[available].sort_values(
                        ["num_modes", "layer_size", "num_layers"]
                    ),
                    width="stretch",
                )

    # --- post-sweep status (only after process ends) ---
    if not st.session_state.is_sweeping:
        if _METRICS_PATH.exists() and _METRICS_PATH.stat().st_size > 0:
            df2 = adapter.parse_sweep_metrics()
            done_count = len(df2[df2["status"] == "done"]) if not df2.empty else 0
            st.session_state.sweep_completed_count = done_count

            cfg = st.session_state.sweep_config
            total = _total_combinations(
                cfg.get("num_modes_list", []),
                cfg.get("layer_size_list", []),
                cfg.get("num_layers_from", 1),
                cfg.get("num_layers_to", 15),
                1,
            )
            log_lines = adapter.read_sweep_log_tail(50)
            has_error = any(
                kw in line
                for line in log_lines
                for kw in ("Error", "Traceback", "Exception")
            ) and done_count < total
            if has_error:
                st.error("Sweep stopped with errors. See log details below.")
                with st.expander("Sweep log (last 50 lines)"):
                    st.code("\n".join(log_lines), language="text")
            else:
                if done_count >= total:
                    st.success(
                        f"Sweep finished — {done_count}/{total} completed."
                    )
                else:
                    st.warning(
                        f"Sweep stopped — {done_count}/{total} completed."
                    )
        else:
            st.caption("Status: idle")

    # --- CSV download (shown when sweep has completed data) ---
    if not st.session_state.is_sweeping and not df.empty:
        done_final = df[df["status"] == "done"]
        if not done_final.empty:
            csv = done_final.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="sweep_results.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------------------------
# Page entry
# ---------------------------------------------------------------------------

_init_state()

st.title("Analysis")
st.divider()

_section_load_data()
st.divider()

_section_sweep_config()
st.divider()

_section_run()
