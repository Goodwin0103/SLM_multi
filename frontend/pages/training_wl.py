"""Multi-WL Training page: Load Data -> Parameter Config -> Training."""

import io
import json
import sys
from pathlib import Path

import altair as alt
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh

_FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

_PROJECT_ROOT = _FRONTEND_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adapters.mainfor6_wl_adapter import Mainfor6WLAdapter
from adapters.remote_adapter import RemoteAdapter, load_remote_config
from services.config_manager import ConfigManager
from utils.log_parser import latest_metrics, parse_metrics_jsonl
from utils.time_utils import fmt_seconds

# Directories
TEMP_DIR     = _FRONTEND_DIR / "temp"
LOG_DIR      = _FRONTEND_DIR / "logs"
CONFIG_PATH  = TEMP_DIR / "train_config_wl.json"
METRICS_PATH = LOG_DIR / "metrics_wl.jsonl"
REMOTE_METRICS_PATH = Path("/tmp/odnn_remote_metrics_wl.jsonl")

TEMP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

_UI_STATE_PATH = TEMP_DIR / "training_wl_ui_state.json"

_PARAM_FIELDS = [
    "layer_size", "num_modes", "batch_size", "epochs",
    "learning_rate", "lr_gamma", "lr_step_size",
    "z_layers_um", "z_prop_um", "z_input_to_first_um", "pixel_size_um",
    "padding_ratio_out",
]


def _parse_int_list(text: str) -> list[int]:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    result = []
    for p in parts:
        try:
            result.append(int(p))
        except ValueError:
            continue
    return result


def _load_ui_state() -> dict:
    """Load persisted UI state from disk (survives session resets)."""
    if _UI_STATE_PATH.exists():
        try:
            return json.loads(_UI_STATE_PATH.read_text())
        except Exception:
            pass
    return {}

def _save_ui_state(d: dict) -> None:
    """Persist UI state to disk."""
    _UI_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _UI_STATE_PATH.write_text(json.dumps(d))

def _persist_training_state() -> None:
    """Save current training-related state to disk for cross-navigation survival."""
    _save_ui_state({
        "compute_mode": st.session_state.get("wl_compute_mode", "Local"),
        "is_training": st.session_state.get("is_training", False),
        "remote_job_id": st.session_state.get("wl_remote_job_id", None),
    })


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    ui = _load_ui_state()
    if "wl_compute_mode" not in st.session_state:
        st.session_state.wl_compute_mode = ui.get("compute_mode", "Local")
    if "wl_remote_job_id" not in st.session_state:
        st.session_state.wl_remote_job_id = ui.get("remote_job_id", None)
    if "wl_remote_adapter" not in st.session_state:
        st.session_state.wl_remote_adapter = None

    if "wl_adapter" not in st.session_state:
        st.session_state.wl_adapter = Mainfor6WLAdapter()

    if "mat_file_path" not in st.session_state:
        st.session_state.mat_file_path = None
    if "max_modes" not in st.session_state:
        st.session_state.max_modes = 20
    if "training_pid" not in st.session_state:
        st.session_state.training_pid = None
    if "is_training" not in st.session_state:
        st.session_state.is_training = ui.get("is_training", False)
    if "training_error" not in st.session_state:
        st.session_state.training_error = None

    # Reload config from disk whenever the file changed (designer may have saved new values)
    _config_mtime = CONFIG_PATH.stat().st_mtime if CONFIG_PATH.exists() else 0.0
    if "wl_config_mtime" not in st.session_state:
        st.session_state.wl_config_mtime = 0.0
    if "wl_train_config" not in st.session_state or _config_mtime > st.session_state.wl_config_mtime:
        defaults = st.session_state.wl_adapter.load_default_config()
        mgr = ConfigManager(CONFIG_PATH)
        saved = mgr.load_config()
        st.session_state.wl_train_config = mgr.merge_with_defaults(saved, defaults)
        st.session_state.wl_config_mtime = _config_mtime

    cfg = st.session_state.wl_train_config
    for field in _PARAM_FIELDS:
        wkey = f"wl_param_{field}"
        if wkey not in st.session_state:
            st.session_state[wkey] = cfg.get(field, 0)

    if "wl_param_layers_text" not in st.session_state:
        layers_list = cfg.get("num_layers_list", [1, 2, 3, 4, 5])
        st.session_state["wl_param_layers_text"] = ", ".join(str(l) for l in layers_list)


def _get_adapter():
    """Return the active adapter based on compute mode."""
    if st.session_state.wl_compute_mode == "Remote":
        if st.session_state.wl_remote_adapter is None:
            cfg = load_remote_config()
            if cfg:
                st.session_state.wl_remote_adapter = RemoteAdapter(
                    host=cfg["host"], user=cfg["user"],
                    project_dir=cfg.get("project_dir", ""),
                    workspace_dir=cfg.get("workspace_dir", ""),
                    conda_env=cfg.get("conda_env", "odnn"),
                    port=int(cfg.get("port", 22)),
                )
        return st.session_state.wl_remote_adapter
    return st.session_state.wl_adapter


# ---------------------------------------------------------------------------
# Section 1 — Load Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_mat_shape(mat_path: str):
    from odnn_io import load_complex_modes_from_mat
    arr, _ = load_complex_modes_from_mat(mat_path)
    return arr.shape[0], arr.shape[2]


def _section_load_data() -> None:
    st.subheader("1. Load Data")

    # Auto-fill mat_file_path from train_config_wl.json (saved by Designer)
    if not st.session_state.mat_file_path:
        saved_mat = st.session_state.wl_train_config.get("mat_file_path")
        if saved_mat and Path(saved_mat).exists():
            st.session_state.mat_file_path = saved_mat
            st.info(f"Using dataset: **{Path(saved_mat).name}**")

    uploaded = st.file_uploader("Select .mat mode file", type=["mat"])

    if uploaded is not None:
        dest = TEMP_DIR / uploaded.name
        dest.write_bytes(uploaded.getbuffer())
        st.session_state.mat_file_path = str(dest)
        st.session_state.wl_train_config["mat_file_path"] = str(dest)

    if st.session_state.mat_file_path:
        p = Path(st.session_state.mat_file_path)
        size_kb = p.stat().st_size / 1024 if p.exists() else 0
        col1, col2 = st.columns(2)
        col1.success(f"Loaded: {p.name}")
        col2.caption(f"Size: {size_kb:.1f} KB  |  Path: {p}")

        try:
            fs, mm = _load_mat_shape(st.session_state.mat_file_path)
            st.session_state.max_modes = int(mm)
            st.session_state.wl_train_config["field_size"] = int(fs)
            if st.session_state.wl_train_config.get("num_modes", 10) > mm:
                st.session_state.wl_train_config["num_modes"] = int(mm)
                st.session_state["wl_param_num_modes"] = int(mm)
            st.caption(f"Detected: field_size={fs}, modes={mm}")
        except Exception:
            st.session_state.max_modes = st.session_state.get("max_modes", 20)
    else:
        st.info("No file loaded. Upload a .mat file above, or enter the path manually.")
        manual = st.text_input(
            "Or enter path manually",
            placeholder="/path/to/your/mmf_file.mat",
        )
        if manual and Path(manual).exists():
            st.session_state.mat_file_path = manual
            st.session_state.wl_train_config["mat_file_path"] = manual
            st.rerun()
        elif manual:
            st.warning("File not found at the given path.")


# ---------------------------------------------------------------------------
# Section 2 — Parameter Config
# ---------------------------------------------------------------------------

def _section_param_config() -> None:
    st.subheader("2. Parameter Config")

    cfg = st.session_state.wl_train_config

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Model**")
        max_modes = st.session_state.get("max_modes", 20)
        st.number_input("Layer canvas size (px)", 50,  500,  step=10,  key="wl_param_layer_size")
        layers_text = st.text_input(
            "D2NN layers",
            value=st.session_state.get("wl_param_layers_text", "1, 2, 3, 4, 5"),
            placeholder="e.g. 1, 2, 3, 4, 5",
            key="wl_param_layers_text",
        )
        cfg["num_layers_list"] = _parse_int_list(layers_text)
        st.number_input("Number of modes",        1,   max_modes, step=1, key="wl_param_num_modes")

        st.markdown("**Output**")
        st.number_input(
            "Padding ratio out", 0.0, 1.0, value=cfg.get("padding_ratio_out", 0.5),
            format="%.2f", step=0.05, key="wl_param_padding_ratio_out",
        )

    with col2:
        st.markdown("**Physics**")
        st.number_input("Layer separation z (um)",    1,   100000, step=100,  key="wl_param_z_layers_um")
        st.number_input("Output prop. distance (um)", 1,   500000, step=500,  key="wl_param_z_prop_um")
        st.number_input("Input-to-first z (um)",      0,   100000, step=100,  key="wl_param_z_input_to_first_um")

    with col3:
        st.markdown("**Training**")
        st.number_input("Pixel size (um)",     0.1, 100.0, format="%.1f", step=0.5, key="wl_param_pixel_size_um")
        st.number_input("Epochs",       1,    5000, step=10,                       key="wl_param_epochs")
        st.number_input("Batch size",   1,    256,  step=4,                        key="wl_param_batch_size")
        st.number_input("Learning rate", 0.01, 10.0, value=cfg.get("learning_rate", 1.99), format="%.4f", step=0.01, key="wl_param_learning_rate")
        st.number_input("LR gamma",      0.5,  1.0,  value=cfg.get("lr_gamma", 0.99),       format="%.3f", step=0.005, key="wl_param_lr_gamma")
        st.number_input("LR step (epochs)", 1, 500, value=cfg.get("lr_step_size", 1),     step=1,   key="wl_param_lr_step_size")

        raw_phase = int(cfg.get("phase_option", 4))
        phase_idx = max(0, min(4, raw_phase - 1))
        cfg["phase_option"] = st.selectbox("Phase option", [1, 2, 3, 4, 5], index=phase_idx)

    with col4:
        st.markdown("**Dataset**")
        cfg["training_dataset_mode"] = st.selectbox("Training dataset", ["eigenmode", "superposition"],
                                                      index=0 if cfg["training_dataset_mode"] == "eigenmode" else 1)
        cfg["evaluation_mode"]       = st.selectbox("Evaluation mode",  ["eigenmode", "superposition"],
                                                      index=0 if cfg["evaluation_mode"] == "eigenmode" else 1)

    # sync widget states back into train_config
    for field in _PARAM_FIELDS:
        wkey = f"wl_param_{field}"
        if wkey in st.session_state:
            cfg[field] = st.session_state[wkey]

    cfg["num_layers_list"] = _parse_int_list(
        st.session_state.get("wl_param_layers_text", "")
    )
    st.session_state.wl_train_config = cfg

    if st.button("Save Config", type="secondary"):
        # Only persist known keys (strip garbage from old versions / other adapters)
        _ALL_KNOWN_KEYS = {
            "layer_size", "out_size", "num_modes", "batch_size", "epochs",
            "learning_rate", "lr_gamma", "lr_step_size", "base_wavelength_idx", "phase_option",
            "z_layers_um", "z_prop_um", "z_input_to_first_um", "pixel_size_um",
            "circle_focus_radius", "margin_ratio", "circle_detectsize",
            "wl_start_nm", "wl_spacing_nm", "wl_count", "padding_ratio_out",
            "padding_ratio", "field_size", "num_layers_list",
            "training_dataset_mode", "evaluation_mode",
            "num_superposition_eval_samples", "num_data",
            "label_config", "mat_file_path", "mat_file_remote_path",
        }
        clean = {k: v for k, v in cfg.items() if k in _ALL_KNOWN_KEYS}
        ConfigManager(CONFIG_PATH).save_config(clean)
        st.session_state.wl_config_mtime = CONFIG_PATH.stat().st_mtime
        st.success(f"Config saved to {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Section 3 — Training
# ---------------------------------------------------------------------------

def _do_start_training() -> None:
    cfg      = st.session_state.wl_train_config
    mat_path = st.session_state.mat_file_path
    assert mat_path, "Start button must be disabled when no .mat file is loaded"

    # Save config before training so the current mat_file_path is persisted
    cfg["mat_file_path"] = mat_path
    ConfigManager(CONFIG_PATH).save_config(cfg)
    st.session_state.wl_config_mtime = CONFIG_PATH.stat().st_mtime

    st.session_state.training_error = None
    adapter = _get_adapter()

    is_remote = st.session_state.wl_compute_mode == "Remote"

    if is_remote:
        if adapter is None:
            st.session_state.training_error = "Remote server not configured. Go to Settings first."
            return
        REMOTE_METRICS_PATH.write_text("")
    else:
        METRICS_PATH.write_text("")

    try:
        if is_remote:
            gpu_id = st.session_state.get("manual_gpu_id")
            result = adapter.start_training(cfg, mat_file=mat_path, gpu_id=gpu_id)
        else:
            result = adapter.start_training(cfg, mat_file=mat_path)
    except Exception as exc:
        st.session_state.training_error = str(exc)
        return

    if is_remote:
        st.session_state.wl_remote_job_id = result
    else:
        st.session_state.training_pid = result
    st.session_state.is_training = True
    _persist_training_state()


def _do_stop_training() -> None:
    adapter = _get_adapter()
    is_remote = st.session_state.wl_compute_mode == "Remote"

    if is_remote:
        job_id = st.session_state.wl_remote_job_id
        if job_id and adapter:
            try:
                adapter.stop_training(job_id)
            except Exception as exc:
                st.session_state.training_error = f"Failed to stop remote training: {exc}"
        st.session_state.wl_remote_job_id = None
    else:
        pid = st.session_state.training_pid
        if pid:
            try:
                adapter.stop_training(pid)
            except Exception as exc:
                st.session_state.training_error = f"Failed to stop local training: {exc}"
        st.session_state.training_pid = None

    st.session_state.is_training = False
    _persist_training_state()


def _render_training_status() -> None:
    adapter = _get_adapter()
    is_remote = st.session_state.wl_compute_mode == "Remote"

    if st.session_state.training_error:
        st.error(st.session_state.training_error)
        return

    if st.session_state.is_training:
        if is_remote:
            st.info(f"Training in progress  (Job: {st.session_state.wl_remote_job_id})")
        else:
            st.info(f"Training in progress  (PID {st.session_state.training_pid})")
        return

    metrics_path = REMOTE_METRICS_PATH if is_remote else METRICS_PATH
    if metrics_path.exists() and metrics_path.stat().st_size > 0:
        log_lines = adapter.read_log_tail(50)
        has_error = any(
            kw in line for line in log_lines for kw in ("Error", "Traceback", "Exception")
        )
        if has_error:
            st.error("Training stopped with errors. See details below.")
            with st.expander("Training log (last 50 lines)"):
                st.code("\n".join(log_lines), language="text")
        else:
            st.success("Training finished (or stopped).")
    else:
        st.caption("Status: idle")


def _section_training() -> None:
    st.subheader("3. Training")

    adapter     = _get_adapter()
    is_remote   = st.session_state.wl_compute_mode == "Remote"
    job_id      = st.session_state.wl_remote_job_id if is_remote else None

    # -- liveness + metrics polling ----------------------------------------
    if st.session_state.is_training:
        refresh_interval = 3000 if is_remote else 1000
        st_autorefresh(interval=refresh_interval, key="wl_training_monitor")

        # track cycle count for less frequent liveness checks
        cycle = st.session_state.get("wl_poll_cycle", 0) + 1
        st.session_state.wl_poll_cycle = cycle

        # remote: fetch metrics first (it's the primary signal of life)
        metrics_updated = False
        if is_remote and adapter and job_id:
            try:
                raw = adapter.fetch_metrics_jsonl()
                if raw.strip():
                    REMOTE_METRICS_PATH.write_text(raw)
                    metrics_updated = True
            except Exception:
                pass

        # Only check liveness every 5 cycles (15s) or when metrics are stale,
        # to reduce SSH call frequency and avoid connection-refused storms.
        need_liveness = (cycle % 5 == 0) or (not metrics_updated and cycle > 2)
        if need_liveness:
            alive = False
            if is_remote and job_id and adapter:
                try:
                    alive = adapter.is_training_alive(job_id)
                except Exception:
                    # SSH temp failure → assume alive, retry next cycle
                    alive = True
            elif not is_remote and st.session_state.training_pid:
                alive = adapter.is_training_alive(st.session_state.training_pid)

            if not alive:
                st.session_state.is_training = False
                if is_remote:
                    st.session_state.wl_remote_job_id = None
                else:
                    st.session_state.training_pid = None
                _persist_training_state()

    col_start, col_stop = st.columns([1, 1])
    with col_start:
        disabled_start = (
            st.session_state.is_training
            or (st.session_state.mat_file_path is None)
            or (is_remote and adapter is None)
        )
        # debug: show why button is disabled
        reasons = []
        if st.session_state.is_training:
            reasons.append("already training")
        if st.session_state.mat_file_path is None:
            reasons.append("no .mat file loaded")
        if is_remote and adapter is None:
            reasons.append("remote adapter not configured")
        if reasons:
            st.caption("Start disabled: " + ", ".join(reasons))
        if st.button("Start", type="primary", disabled=disabled_start):
            _do_start_training()
            st.rerun()
    with col_stop:
        if st.button("Stop", disabled=not st.session_state.is_training):
            _do_stop_training()
            st.rerun()

    _render_training_status()

    # -- metrics display ----------------------------------------------------
    metrics_path = REMOTE_METRICS_PATH if is_remote else METRICS_PATH
    last = latest_metrics(metrics_path)
    if not last:
        if not st.session_state.is_training:
            st.caption("No metrics yet. Start training to see live data.")
        return

    # Only show live metrics while training is active; avoid stale progress bar
    if not st.session_state.is_training:
        return

    # --- overall progress ---
    layer_now      = int(last.get("layer", 1))
    total_layers   = int(last.get("total_layers", 1))
    overall_epoch  = int(last.get("overall_epoch", 0))
    overall_total  = int(last.get("overall_epochs_total", 1))
    overall_pct    = min(1.0, overall_epoch / max(overall_total, 1))

    if total_layers > 1:
        st.progress(
            overall_pct,
            text=f"Overall: {overall_epoch} / {overall_total} epochs  |  Layer {layer_now} / {total_layers}",
        )
    else:
        st.progress(overall_pct, text=f"Epoch {overall_epoch} / {overall_total}")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    epoch_now      = int(last.get("epoch", 0))
    epochs_total   = int(last.get("epochs_total", 1))
    m1.metric("Epoch (layer)",  f"{epoch_now}/{epochs_total}")
    m2.metric("Loss",           f"{last.get('loss', 0):.6f}")
    m3.metric("Elapsed",        fmt_seconds(last.get("elapsed_time", 0)))
    m4.metric("Learning Rate",  f"{last.get('lr', 0):.6g}")
    m5.metric("ETR (layer)",    fmt_seconds(last.get("etr", 0)))
    m6.metric("ETR (overall)",  fmt_seconds(last.get("overall_etr", 0)))

    # --- charts: one pair per layer ---
    df = parse_metrics_jsonl(metrics_path)
    if not df.empty and len(df) >= 2 and "layer" in df.columns:
        trained_layers = sorted(df["layer"].unique())
        for lyr in trained_layers:
            df_layer = df[df["layer"] == lyr]
            if len(df_layer) < 2:
                continue
            st.markdown(f"**Layer {int(lyr)}**")
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                st.caption("Loss vs Epoch")
                loss_chart = (
                    alt.Chart(df_layer)
                    .mark_line()
                    .encode(
                        alt.X("epoch", type="quantitative", title="Epoch"),
                        alt.Y("loss", type="quantitative", title="Loss"),
                    )
                    .properties(width="container")
                )
                st.altair_chart(loss_chart)
            with chart_col2:
                st.caption("Learning Rate vs Epoch")
                lr_chart = (
                    alt.Chart(df_layer)
                    .mark_line()
                    .encode(
                        alt.X("epoch", type="quantitative", title="Epoch"),
                        alt.Y("lr", type="quantitative", title="Learning Rate"),
                    )
                    .properties(width="container")
                )
                st.altair_chart(lr_chart)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

_init_state()

st.title("Multi-WL Training")

st.segmented_control(
    "Compute", options=["Local", "Remote"],
    default=st.session_state.wl_compute_mode,
    key="wl_compute_mode",
    on_change=_persist_training_state,
)

st.divider()

_section_load_data()
st.divider()

_section_param_config()
st.divider()

_section_training()
