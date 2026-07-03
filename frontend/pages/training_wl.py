"""Multi-WL Training page: Load Data -> Parameter Config -> Training."""

import io
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

_PARAM_FIELDS = [
    "layer_size", "out_size", "num_modes", "batch_size", "epochs",
    "learning_rate", "lr_gamma", "base_wavelength_idx",
    "z_layers_um", "z_prop_um", "z_input_to_first_um", "pixel_size_um",
    "circle_focus_radius", "margin_ratio",
    "wl_start_nm", "wl_spacing_nm", "wl_count", "padding_ratio_out",
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


# ---------------------------------------------------------------------------
# Mode preview (cached)
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading mode preview...")
def _load_mode_preview(
    mat_path: str,
    layer_size: int,
    num_modes: int,
    wavelengths_nm: list,
    circle_radius: int = 5,
    margin_ratio: float = 0.1,
):
    """Return input amplitudes + label patterns for ALL modes (0..num_modes-1)."""
    from odnn_io import load_complex_modes_from_mat

    modes = load_complex_modes_from_mat(mat_path, key="modes_field")
    field_size = int(modes.shape[0])
    total_modes = int(modes.shape[2])
    M = min(num_modes, total_modes)

    mmf = modes[:, :, :M].transpose(2, 0, 1)  # (M, H, W)
    input_amp = np.abs(mmf)

    # build multi-wavelength label patterns (banded per wavelength)
    L = len(wavelengths_nm)
    H, W = layer_size, layer_size

    inner_margin = circle_radius + 3
    total_per_wl = M
    mx = max(int(W * margin_ratio), circle_radius + 5)
    my = max(int(H * margin_ratio), circle_radius + 5)
    avail_y = H - 2 * my
    band_h = avail_y / max(L, 1)
    band_w = W - 2 * mx
    ncols = max(1, min(total_per_wl, int(np.ceil(np.sqrt(total_per_wl * band_w / max(band_h, 1))))))
    nrows = int(np.ceil(total_per_wl / ncols))

    total = M * L
    patterns = np.zeros((H, W, total), dtype=np.float32)
    for mode_idx in range(M):
        for wl_idx in range(L):
            idx = mode_idx * L + wl_idx
            band_y0 = my + wl_idx * band_h
            band_y1 = my + (wl_idx + 1) * band_h
            row = mode_idx // ncols
            col = mode_idx % ncols
            xs_arr = np.linspace(mx, W - 1 - mx, ncols)
            ys_arr = np.linspace(band_y0 + inner_margin, band_y1 - inner_margin, nrows)
            cx = int(round(xs_arr[col]))
            cy = int(round(ys_arr[row]))
            yy, xx = np.ogrid[:H, :W]
            patterns[:, :, idx] = (
                (yy - cy) ** 2 + (xx - cx) ** 2 <= circle_radius ** 2
            ).astype(np.float32)

    labels = np.zeros((M, H, W), dtype=np.float32)
    for mode_idx in range(M):
        wl_indices = [mode_idx * L + wl for wl in range(L)]
        wl_patterns = patterns[:, :, wl_indices]  # (H, W, L)
        labels[mode_idx] = wl_patterns.sum(axis=2)  # sum over all wavelengths

    return input_amp, labels, field_size, total_modes


# ---------------------------------------------------------------------------
# Session state init
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "wl_compute_mode" not in st.session_state:
        st.session_state.wl_compute_mode = "Local"
    if "wl_remote_job_id" not in st.session_state:
        st.session_state.wl_remote_job_id = None
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
        st.session_state.is_training = False
    if "training_error" not in st.session_state:
        st.session_state.training_error = None

    if "wl_train_config" not in st.session_state:
        defaults = st.session_state.wl_adapter.load_default_config()
        mgr = ConfigManager(CONFIG_PATH)
        saved = mgr.load_config()
        st.session_state.wl_train_config = mgr.merge_with_defaults(saved, defaults)

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
            "Output size (px)", 50, 1000, step=10,
            value=cfg.get("out_size", 600), key="wl_param_out_size",
        )
        st.number_input(
            "Padding ratio out", 0.0, 1.0, value=cfg.get("padding_ratio_out", 0.5),
            format="%.2f", step=0.05, key="wl_param_padding_ratio_out",
        )

    with col2:
        st.markdown("**Physics**")
        st.number_input(
            "Wavelength start (nm)", 400, 2000, step=1,
            value=cfg.get("wl_start_nm", 1550), key="wl_param_wl_start_nm",
        )
        st.number_input(
            "Wavelength spacing (nm)", 0.1, 100.0, format="%.1f", step=0.5,
            value=cfg.get("wl_spacing_nm", 0.5), key="wl_param_wl_spacing_nm",
        )
        st.number_input(
            "Wavelength count", 1, 50, step=1,
            value=cfg.get("wl_count", 2), key="wl_param_wl_count",
        )
        st.caption(f"Total: {int(st.session_state.get('wl_param_wl_count', 2))} wavelengths")
        st.number_input("Base wavelength index", 0, 20, step=1, key="wl_param_base_wavelength_idx")
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

        st.markdown("**Label**")
        st.number_input(
            "Focus radius (px)", 1, 50, step=1,
            value=cfg.get("circle_focus_radius", 5), key="wl_param_circle_focus_radius",
        )
        st.number_input(
            "Margin ratio", 0.05, 0.5, format="%.2f", step=0.05,
            value=cfg.get("margin_ratio", 0.2), key="wl_param_margin_ratio",
        )

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
        ConfigManager(CONFIG_PATH).save_config(cfg)
        st.success(f"Config saved to {CONFIG_PATH}")

    # --- mode preview (shown when .mat is loaded) ---
    if st.session_state.mat_file_path and Path(st.session_state.mat_file_path).exists():
        ls = int(st.session_state.get("wl_param_layer_size", 300))
        os_val = int(st.session_state.get("wl_param_out_size", ls))
        nm = int(st.session_state.get("wl_param_num_modes", 10))
        pix_um = float(st.session_state.get("wl_param_pixel_size_um", 12.5))
        cr = int(st.session_state.get("wl_param_circle_focus_radius", 5))
        mr = float(st.session_state.get("wl_param_margin_ratio", 0.2))
        ws = float(st.session_state.get("wl_param_wl_start_nm", 1550))
        wd = float(st.session_state.get("wl_param_wl_spacing_nm", 0.5))
        wc = int(st.session_state.get("wl_param_wl_count", 2))
        wl_nm = [ws + i * wd for i in range(wc)]
        max_m = int(st.session_state.get("max_modes", nm))
        preview_nm = min(nm, max_m)

        if preview_nm > 0:
            try:
                amp_all, lbls_all, _fs, _tm = _load_mode_preview(
                    st.session_state.mat_file_path, os_val, preview_nm, wl_nm,
                    circle_radius=cr, margin_ratio=mr,
                )
                disp_nm = min(preview_nm, _tm)
                if disp_nm > 0:
                    st.markdown("---")
                    st.markdown("**Dataset Preview**")
                    mode_idx = st.selectbox(
                        "Mode",
                        options=list(range(disp_nm)),
                        format_func=lambda i: f"Mode {i + 1}",
                        key="wl_preview_mode_idx",
                    )
                    fig, (ax_amp, ax_lbl) = plt.subplots(1, 2, figsize=(10, 4.5))

                    im_amp = ax_amp.imshow(amp_all[mode_idx], cmap="inferno",
                                           extent=[0, _fs * pix_um, _fs * pix_um, 0])
                    ax_amp.set_title(f"Mode {mode_idx + 1}  Input Amplitude\n"
                                     f"{_fs}x{_fs} px  |  {pix_um:.1f} um/px"
                                     f"  |  {_fs * pix_um:.0f} x {_fs * pix_um:.0f} um")
                    fig.colorbar(im_amp, ax=ax_amp, fraction=0.046, pad=0.04)

                    im_lbl = ax_lbl.imshow(lbls_all[mode_idx], cmap="inferno",
                                           extent=[0, os_val * pix_um, os_val * pix_um, 0])
                    ax_lbl.set_title(f"Mode {mode_idx + 1}  Label\n"
                                     f"out_size={os_val} px  |  {pix_um:.1f} um/px"
                                     f"  |  {os_val * pix_um:.0f} x {os_val * pix_um:.0f} um")
                    fig.colorbar(im_lbl, ax=ax_lbl, fraction=0.046, pad=0.04)

                    fig.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
            except Exception as exc:
                st.caption(f"(mode preview unavailable: {exc})")


# ---------------------------------------------------------------------------
# Section 3 — Training
# ---------------------------------------------------------------------------

def _do_start_training() -> None:
    cfg      = st.session_state.wl_train_config
    mat_path = st.session_state.mat_file_path
    assert mat_path, "Start button must be disabled when no .mat file is loaded"

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
        result = adapter.start_training(cfg, mat_file=mat_path)
    except Exception as exc:
        st.session_state.training_error = str(exc)
        return

    if is_remote:
        st.session_state.wl_remote_job_id = result
    else:
        st.session_state.training_pid = result
    st.session_state.is_training = True


def _do_stop_training() -> None:
    adapter = _get_adapter()
    is_remote = st.session_state.wl_compute_mode == "Remote"

    if is_remote:
        job_id = st.session_state.wl_remote_job_id
        if job_id and adapter:
            adapter.stop_training(job_id)
        st.session_state.wl_remote_job_id = None
    else:
        pid = st.session_state.training_pid
        if pid:
            adapter.stop_training(pid)
        st.session_state.training_pid = None

    st.session_state.is_training = False


def _render_training_status() -> None:
    adapter = _get_adapter()
    is_remote = st.session_state.wl_compute_mode == "Remote"

    if st.session_state.training_error:
        st.error(f"Failed to start training: {st.session_state.training_error}")
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
    m4.metric("Learning Rate",  f"{last.get('lr', 0):.2e}")
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
    key="wl_compute_mode",
)

st.divider()

_section_load_data()
st.divider()

_section_param_config()
st.divider()

_section_training()
