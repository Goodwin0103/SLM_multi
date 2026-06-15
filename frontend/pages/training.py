"""Training page: Load Data -> Parameter Config -> Training."""

import sys
from pathlib import Path

import altair as alt
import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Ensure sibling packages are importable when the page is loaded directly
_FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

# Also ensure project root is on path for backend imports (odnn_io, etc.)
_PROJECT_ROOT = _FRONTEND_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adapters.mainfor6_adapter import Mainfor6Adapter
from services.config_manager import ConfigManager
from utils.log_parser import latest_metrics, parse_metrics_jsonl
from utils.time_utils import fmt_seconds

# Directories
TEMP_DIR     = _FRONTEND_DIR / "temp"
LOG_DIR      = _FRONTEND_DIR / "logs"
CONFIG_PATH  = TEMP_DIR / "train_config.json"
METRICS_PATH = LOG_DIR / "metrics.jsonl"

TEMP_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Widget keys for number_input fields (used to sync session_state -> train_config)
_PARAM_FIELDS = [
    "layer_size", "num_layers", "num_modes", "batch_size", "epochs",
    "learning_rate", "lr_gamma", "wavelength_nm", "z_layers_um",
    "z_prop_um", "z_input_to_first_um",
]


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    # adapter is the single source of defaults -- no parallel DEFAULTS dict here
    if "adapter" not in st.session_state:
        st.session_state.adapter = Mainfor6Adapter()

    if "mat_file_path" not in st.session_state:
        st.session_state.mat_file_path = None
    if "max_modes" not in st.session_state:
        st.session_state.max_modes = 20
    if "training_pid" not in st.session_state:
        st.session_state.training_pid = None
    if "is_training" not in st.session_state:
        st.session_state.is_training = False
    if "training_error" not in st.session_state:
        st.session_state.training_error = None  # persists across reruns until cleared

    if "train_config" not in st.session_state:
        defaults = st.session_state.adapter.load_default_config()
        mgr = ConfigManager(CONFIG_PATH)
        saved = mgr.load_config()
        # merge_with_defaults fills any fields missing from an older saved config
        st.session_state.train_config = mgr.merge_with_defaults(saved, defaults)

    # initialise individual widget keys once so number_input uses key= mode
    cfg = st.session_state.train_config
    for field in _PARAM_FIELDS:
        wkey = f"param_{field}"
        if wkey not in st.session_state:
            st.session_state[wkey] = cfg[field]


# ---------------------------------------------------------------------------
# Section 1 — Load Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_mat_shape(mat_path: str):
    """Return (field_size, num_modes) from a .mat file without keeping the full array."""
    from odnn_io import load_complex_modes_from_mat
    arr = load_complex_modes_from_mat(mat_path)
    return arr.shape[0], arr.shape[2]


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
        col2.caption(f"Size: {size_kb:.1f} KB  |  Path: {p}")

        # auto-detect field_size and max_modes from the .mat file
        try:
            fs, mm = _load_mat_shape(st.session_state.mat_file_path)
            st.session_state.max_modes = int(mm)
            st.session_state.train_config["field_size"] = int(fs)
            # clamp current num_modes selection if it exceeds the new max
            if st.session_state.train_config.get("num_modes", 6) > mm:
                st.session_state.train_config["num_modes"] = int(mm)
                st.session_state["param_num_modes"] = int(mm)
            st.caption(f"Detected: field_size={fs}, modes={mm}")
        except Exception:
            st.session_state.max_modes = st.session_state.get("max_modes", 20)
    else:
        st.info("No file loaded. Upload a .mat file above, or place it in the project root and type the path.")
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
# Section 2 — Parameter Config
# ---------------------------------------------------------------------------

def _section_param_config() -> None:
    st.subheader("2. Parameter Config")

    cfg = st.session_state.train_config

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("**Model**")
        max_modes = st.session_state.get("max_modes", 20)
        st.number_input("Layer canvas size (px)", 50,  500,  step=10, key="param_layer_size")
        st.number_input("D2NN layers",            1,   15,   step=1,  key="param_num_layers")
        st.number_input("Number of modes",        1,   max_modes, step=1,  key="param_num_modes")

    with col2:
        st.markdown("**Physics**")
        st.number_input("Wavelength (nm)",            400, 2000, step=1,  key="param_wavelength_nm")
        st.number_input("Layer separation z (um)",    1,   500,  step=1,  key="param_z_layers_um")
        st.number_input("Output prop. distance (um)", 1,   1000, step=5,  key="param_z_prop_um")
        st.number_input("Input-to-first z (um)",      1,   500,  step=1,  key="param_z_input_to_first_um")

    with col3:
        st.markdown("**Training**")
        st.number_input("Epochs",       1,    5000, step=10,              key="param_epochs")
        st.number_input("Batch size",   1,    256,  step=4,               key="param_batch_size")
        st.number_input("Learning rate", 0.01, 10.0, value=cfg.get("learning_rate", 1.99), format="%.4f", step=0.01, key="param_learning_rate")
        st.number_input("LR gamma",      0.5,  1.0,  value=cfg.get("lr_gamma", 0.99),       format="%.3f", step=0.005, key="param_lr_gamma")

        # clamp to valid range [1, 5] before computing index, guard against bad saved values
        raw_phase = int(cfg.get("phase_option", 4))
        phase_idx = max(0, min(4, raw_phase - 1))
        cfg["phase_option"] = st.selectbox("Phase option", [1, 2, 3, 4, 5], index=phase_idx)

    with col4:
        st.markdown("**Dataset**")
        cfg["label_pattern_mode"]    = st.selectbox("Label pattern",    ["circle", "eigenmode"],
                                                      index=0 if cfg["label_pattern_mode"] == "circle" else 1)
        cfg["training_dataset_mode"] = st.selectbox("Training dataset", ["eigenmode", "superposition"],
                                                      index=0 if cfg["training_dataset_mode"] == "eigenmode" else 1)
        cfg["evaluation_mode"]       = st.selectbox("Evaluation mode",  ["eigenmode", "superposition"],
                                                      index=0 if cfg["evaluation_mode"] == "eigenmode" else 1)

    # sync widget states back into train_config
    for field in _PARAM_FIELDS:
        wkey = f"param_{field}"
        if wkey in st.session_state:
            cfg[field] = st.session_state[wkey]

    st.session_state.train_config = cfg

    if st.button("Save Config", type="secondary"):
        ConfigManager(CONFIG_PATH).save_config(cfg)
        st.success(f"Config saved to {CONFIG_PATH}")


# ---------------------------------------------------------------------------
# Section 3 — Training
# ---------------------------------------------------------------------------

def _do_start_training() -> None:
    """Delegate training launch to the adapter; update session state."""
    cfg      = st.session_state.train_config
    mat_path = st.session_state.mat_file_path

    # button is disabled when mat_path is None, so this should never fire
    assert mat_path, "Start button must be disabled when no .mat file is loaded"

    METRICS_PATH.write_text("")  # clear previous metrics
    st.session_state.training_error = None  # clear any previous launch error

    try:
        pid = st.session_state.adapter.start_training(cfg, mat_file=mat_path)
    except Exception as exc:
        st.session_state.training_error = str(exc)
        return

    st.session_state.training_pid = pid
    st.session_state.is_training  = True


def _do_stop_training() -> None:
    """Delegate stop signal to the adapter; update session state."""
    pid = st.session_state.training_pid
    if pid:
        st.session_state.adapter.stop_training(pid)
    st.session_state.is_training  = False
    st.session_state.training_pid = None


def _render_training_status() -> None:
    """Show persistent status banner; surface launch errors and crash logs."""
    adapter = st.session_state.adapter

    # launch error from previous start attempt (persists across reruns)
    if st.session_state.training_error:
        st.error(f"Failed to start training: {st.session_state.training_error}")
        return

    if st.session_state.is_training:
        st.info(f"Training in progress  (PID {st.session_state.training_pid})")
        return

    if METRICS_PATH.exists() and METRICS_PATH.stat().st_size > 0:
        # process has finished -- scan log tail for crash traces
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

    adapter = st.session_state.adapter

    # keep training flag in sync with actual process liveness
    if st.session_state.is_training and st.session_state.training_pid:
        if not adapter.is_training_alive(st.session_state.training_pid):
            st.session_state.is_training  = False
            st.session_state.training_pid = None

    # auto-refresh every 1 s while training is active
    if st.session_state.is_training:
        st_autorefresh(interval=1000, key="training_monitor")

    col_start, col_stop = st.columns([1, 1])
    with col_start:
        disabled_start = st.session_state.is_training or (st.session_state.mat_file_path is None)
        if st.button("Start", type="primary", disabled=disabled_start):
            _do_start_training()
            st.rerun()
    with col_stop:
        if st.button("Stop", disabled=not st.session_state.is_training):
            _do_stop_training()
            st.rerun()

    _render_training_status()

    # live metrics (re-read every autorefresh cycle)
    last = latest_metrics(METRICS_PATH)
    if not last:
        if not st.session_state.is_training:
            st.caption("No metrics yet. Start training to see live data.")
        return

    epochs_total = int(st.session_state.train_config.get("epochs", 1000))
    epoch_now    = int(last.get("epoch", 0))
    progress     = min(1.0, epoch_now / max(epochs_total, 1))

    st.progress(progress, text=f"Epoch {epoch_now} / {epochs_total}")

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Epoch",         epoch_now)
    m2.metric("Loss",          f"{last.get('loss', 0):.6f}")
    m3.metric("Elapsed",       fmt_seconds(last.get("elapsed_time", 0)))
    m4.metric("ETR",           fmt_seconds(last.get("etr", 0)))
    m5.metric("Learning Rate", f"{last.get('lr', 0):.2e}")

    # training history charts (full metrics log)
    df = parse_metrics_jsonl(METRICS_PATH)
    if not df.empty and len(df) >= 2:
        chart_col1, chart_col2 = st.columns(2)
        with chart_col1:
            st.caption("Loss vs Epoch")
            loss_chart = alt.Chart(df).mark_line().encode(
                alt.X("epoch", type="quantitative"),
                alt.Y("loss", type="quantitative"),
            ).properties(width="container")
            st.altair_chart(loss_chart)
        with chart_col2:
            st.caption("Learning Rate vs Epoch")
            lr_chart = alt.Chart(df).mark_line().encode(
                alt.X("epoch", type="quantitative"),
                alt.Y("lr", type="quantitative"),
            ).properties(width="container")
            st.altair_chart(lr_chart)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

_init_state()

st.title("Training")
st.divider()

_section_load_data()
st.divider()

_section_param_config()
st.divider()

_section_training()
