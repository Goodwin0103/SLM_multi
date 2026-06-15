"""Testing page -- load a checkpoint and run full optical propagation visualisation."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import streamlit as st

_FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

from adapters.mainfor6_adapter import Mainfor6Adapter, CHECKPOINTS_DIR, PROJECT_ROOT
from components.propagation_viewer import (
    render_propagation_timeline,
    render_phase_mask_gallery,
    render_propagation_player,
)
from services.config_manager import ConfigManager

# --- persistent config defaults for the testing page -------------------
_TESTING_DEFAULTS: Dict[str, Any] = {
    "test_mode":                    "eigenmode",
    "mode_index":                   0,
    "num_superposition_eval_samples": 1000,
    "superposition_seed":           20240116,
    "superposition_vis_sample":     0,
    "z_step_um":                    5.0,
    "phase_option":                 4,
    "label_pattern_mode":           "circle",
    "batch_size":                   16,
}

# widget keys that use key= mode to avoid value= bounce bug
_TEST_WIDGET_FIELDS = [
    "z_step_um",
    "mode_index",
    "num_superposition_eval_samples",
    "superposition_seed",
    "superposition_vis_sample",
]

# path for persisting test-page config
_TEST_CONFIG_PATH = _FRONTEND_DIR / "temp" / "test_config.json"


# -----------------------------------------------------------------------
# Session-state initialisation
# -----------------------------------------------------------------------

def _init_session_state(adapter: Mainfor6Adapter) -> None:
    if "test_cfg" not in st.session_state:
        mgr = ConfigManager(_TEST_CONFIG_PATH)
        saved = mgr.load_config()
        st.session_state.test_cfg = mgr.merge_with_defaults(saved, _TESTING_DEFAULTS)

    # initialise widget keys once so number_input/slider uses key= mode
    cfg = st.session_state.test_cfg
    for field in _TEST_WIDGET_FIELDS:
        wkey = f"tc_{field}"
        if wkey not in st.session_state:
            st.session_state[wkey] = cfg[field]

    if "test_result" not in st.session_state:
        st.session_state.test_result = None

    if "selected_ckpt" not in st.session_state:
        st.session_state.selected_ckpt = None

    if "ckpt_meta" not in st.session_state:
        st.session_state.ckpt_meta = {}

    if "mat_file" not in st.session_state:
        st.session_state.mat_file = ""

    if "test_error" not in st.session_state:
        st.session_state.test_error = None


# -----------------------------------------------------------------------
# Section 1: Model & Data
# -----------------------------------------------------------------------

def _render_model_section(adapter: Mainfor6Adapter) -> None:
    st.subheader("1. Model & Data")

    ckpt_paths = adapter.list_checkpoints()
    if not ckpt_paths:
        st.warning(
            f"No .pth files found in `{CHECKPOINTS_DIR}`.  "
            "Train a model first, or add a checkpoint manually."
        )
        st.session_state.selected_ckpt = None
        st.session_state.ckpt_meta = {}
        return

    ckpt_names = [Path(p).name for p in ckpt_paths]
    current_name = (
        Path(st.session_state.selected_ckpt).name
        if st.session_state.selected_ckpt
        else None
    )
    default_idx = ckpt_names.index(current_name) if current_name in ckpt_names else 0

    chosen_name = st.selectbox("Checkpoint", ckpt_names, index=default_idx)
    chosen_path = ckpt_paths[ckpt_names.index(chosen_name)]

    if chosen_path != st.session_state.selected_ckpt:
        st.session_state.selected_ckpt = chosen_path
        st.session_state.ckpt_meta     = adapter.load_checkpoint_meta(chosen_path)
        st.session_state.test_result   = None   # clear stale results on model change

    meta = st.session_state.ckpt_meta
    if meta:
        with st.expander("Model info", expanded=False):
            info_rows = [
                ("Layers",       str(meta.get("num_layers", "?"))),
                ("Canvas (px)",  str(meta.get("layer_size", "?"))),
                ("Field size",   str(meta.get("field_size", "?"))),
                ("Modes",        str(meta.get("num_modes", "?"))),
                ("z_layers (m)", f"{meta.get('z_layers', '?'):.2e}" if isinstance(meta.get("z_layers"), float) else "?"),
                ("z_prop (m)",   f"{meta.get('z_prop', '?'):.2e}"   if isinstance(meta.get("z_prop"),   float) else "?"),
                ("Wavelength (m)", f"{meta.get('wavelength', '?'):.2e}" if isinstance(meta.get("wavelength"), float) else "?"),
            ]
            st.table(pd.DataFrame(info_rows, columns=["Parameter", "Value"]))

    # mat file path input — prefer the .mat used during training when available
    st.markdown("**Mode file (.mat)**")
    _fallback_mat = str(PROJECT_ROOT / "mmf_6modes_25_PD_1.15.mat")
    default_mat = st.session_state.get("mat_file_path") or _fallback_mat
    mat_path = st.text_input(
        "Path to .mat file",
        value=st.session_state.mat_file or default_mat,
    )
    st.session_state.mat_file = mat_path.strip()

    if st.session_state.mat_file and not Path(st.session_state.mat_file).exists():
        st.warning("File not found — check the path above.")

    st.divider()


# -----------------------------------------------------------------------
# Path geometry diagram
# -----------------------------------------------------------------------

def _render_path_geometry(
    z_inp: float, z_lyr: float, z_prp: float, n_lyr: int
) -> None:
    """Draw a horizontal diagram showing the optical path with distance labels.

    Nodes (input, L1..Ln, detector) are drawn as vertical bars. Annotated
    arrows between adjacent nodes show the propagation distance in um.
    """
    nodes = ["Input"] + [f"L{i+1}" for i in range(n_lyr)] + ["Detector"]
    # gaps[i] = distance from nodes[i] to nodes[i+1]
    # Input→L1: z_inp, L1→L2...L(n-1)→Ln: z_lyr each, Ln→Detector: z_prp
    gaps = [z_inp] + [z_lyr] * (n_lyr - 1) + [z_prp]

    # build cumulative x positions
    xs = [0.0]
    for g in gaps:
        xs.append(xs[-1] + g)
    total_z = xs[-1]

    fig_w = max(6.0, 1.2 * len(nodes))
    fig, ax = plt.subplots(figsize=(fig_w, 1.6))
    ax.set_xlim(-total_z * 0.05, total_z * 1.05)
    ax.set_ylim(-0.5, 1.2)
    ax.axis("off")
    fig.patch.set_facecolor("none")

    bar_h   = 0.55   # half-height of node bars
    label_y = 0.85   # y for node name labels
    dist_y  = -0.25  # y for distance annotation

    # draw z-axis arrow
    ax.annotate(
        "", xy=(total_z * 1.04, 0), xytext=(-total_z * 0.02, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
    )
    ax.text(total_z * 1.05, -0.08, "z", ha="left", va="top", fontsize=8)

    for i, (x, name) in enumerate(zip(xs, nodes)):
        is_layer = name.startswith("L")
        color    = "#c0392b" if is_layer else "#2c3e50"
        lw       = 2.0 if is_layer else 1.5
        ls       = "-"
        ax.plot([x, x], [-bar_h, bar_h], color=color, lw=lw, ls=ls, solid_capstyle="round")
        ax.text(x, label_y, name, ha="center", va="bottom", fontsize=7.5,
                color=color, fontweight="bold" if is_layer else "normal")

        # distance arrow between node i and i+1
        if i < len(nodes) - 1:
            x_next = xs[i + 1]
            dist   = gaps[i]
            mid    = (x + x_next) / 2
            # double-headed arrow
            ax.annotate(
                "", xy=(x_next, dist_y), xytext=(x, dist_y),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.9),
            )
            ax.text(mid, dist_y - 0.18, f"{dist:.0f} um",
                    ha="center", va="top", fontsize=7, color="#333333")

    plt.tight_layout(pad=0.2)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# -----------------------------------------------------------------------
# Section 2: Test Configuration
# -----------------------------------------------------------------------

def _render_config_section() -> None:
    st.subheader("2. Test Configuration")
    cfg = st.session_state.test_cfg

    col1, col2 = st.columns(2)
    with col1:
        test_mode = st.selectbox(
            "Test mode",
            ["eigenmode", "superposition"],
            index=0 if cfg["test_mode"] == "eigenmode" else 1,
        )
        cfg["test_mode"] = test_mode

        if test_mode == "eigenmode":
            meta_modes = st.session_state.ckpt_meta.get("num_modes", 6)
            max_idx = max(0, int(meta_modes) - 1)
            # clamp before render to avoid out-of-range error when checkpoint changes
            if st.session_state.get("tc_mode_index", 0) > max_idx:
                st.session_state["tc_mode_index"] = max_idx
            st.slider(
                "Mode index (0-based)",
                min_value=0,
                max_value=max_idx,
                key="tc_mode_index",
            )
        else:
            st.number_input(
                "Eval samples",
                min_value=10,
                max_value=5000,
                step=100,
                key="tc_num_superposition_eval_samples",
            )
            st.number_input(
                "RNG seed",
                min_value=0,
                step=1,
                key="tc_superposition_seed",
            )
            max_vis = max(0, int(st.session_state.get("tc_num_superposition_eval_samples", 1000)) - 1)
            if st.session_state.get("tc_superposition_vis_sample", 0) > max_vis:
                st.session_state["tc_superposition_vis_sample"] = max_vis
            st.number_input(
                "Visualise sample index",
                min_value=0,
                max_value=max_vis,
                step=1,
                key="tc_superposition_vis_sample",
            )

    with col2:
        # derive z distances from checkpoint meta for preview calculation
        meta = st.session_state.ckpt_meta
        z_inp = float(meta.get("z_input_to_first", 40e-6)) * 1e6
        z_lyr = float(meta.get("z_layers", 40e-6)) * 1e6
        z_prp = float(meta.get("z_prop",   120e-6)) * 1e6
        n_lyr = int(meta.get("num_layers", 3))

        st.number_input(
            "z_step (um)",
            min_value=1.0,
            max_value=500.0,
            step=1.0,
            format="%.1f",
            key="tc_z_step_um",
        )
        # preview: how many frames total
        def count_frames(z_seg, z_step):
            import math
            return max(0, int(z_seg / z_step) - 1)

        z_step_val = float(st.session_state.get("tc_z_step_um", 5.0))
        n_field = (
            1                           # input
            + count_frames(z_inp, z_step_val)
            + n_lyr * (1 + count_frames(z_lyr, z_step_val))  # arrival + fracs
            + n_lyr                     # masks
            + count_frames(z_prp, z_step_val)
            + 1                         # output
        )
        st.caption(f"Estimated frames: ~{n_field}")

    # path geometry diagram -- rendered below both columns so it uses full width
    if meta:
        _render_path_geometry(z_inp, z_lyr, z_prp, n_lyr)

    # sync widget keys back into test_cfg
    for field in _TEST_WIDGET_FIELDS:
        wkey = f"tc_{field}"
        if wkey in st.session_state:
            cfg[field] = st.session_state[wkey]

    st.session_state.test_cfg = cfg
    st.divider()


# -----------------------------------------------------------------------
# Section 3: Run
# -----------------------------------------------------------------------

def _render_run_section(adapter: Mainfor6Adapter) -> None:
    st.subheader("3. Results")

    ready = (
        st.session_state.selected_ckpt is not None
        and st.session_state.mat_file
        and Path(st.session_state.mat_file).exists()
    )

    col_run, col_clr = st.columns([1, 1])
    with col_run:
        if st.button("Run Test", type="primary", disabled=not ready):
            _do_run_test(adapter)

    with col_clr:
        if st.button("Clear Results", disabled=st.session_state.test_result is None):
            st.session_state.test_result = None
            st.session_state.test_error  = None
            st.rerun()

    if not ready:
        st.caption(
            "Select a checkpoint and a valid .mat file to enable the Run button."
        )

    if st.session_state.test_error:
        st.error("Test failed — see details below.")
        with st.expander("Error details"):
            st.code(st.session_state.test_error, language="text")


def _do_run_test(adapter: Mainfor6Adapter) -> None:
    """Call adapter.run_test, store result in session_state."""
    st.session_state.test_error  = None
    st.session_state.test_result = None

    cfg          = st.session_state.test_cfg
    ckpt_path    = st.session_state.selected_ckpt
    mat_file     = st.session_state.mat_file

    try:
        with st.spinner("Running test — loading model and propagating field..."):
            result = adapter.run_test(
                config=cfg,
                checkpoint_path=ckpt_path,
                mat_file=mat_file,
            )
        st.session_state.test_result = result
        # persist config so it survives page reload
        ConfigManager(_TEST_CONFIG_PATH).save_config(cfg)
    except Exception as exc:
        import traceback
        st.session_state.test_error = traceback.format_exc()
    st.rerun()


# -----------------------------------------------------------------------
# Section 4: Results
# -----------------------------------------------------------------------

def _render_results_section() -> None:
    result = st.session_state.test_result
    if result is None:
        return

    tab_prop, tab_player, tab_metrics = st.tabs(
        ["Propagation Timeline", "Propagation Player", "Evaluation Metrics"]
    )

    with tab_prop:
        _render_propagation_tab(result)

    with tab_player:
        _render_player_tab(result)

    with tab_metrics:
        _render_metrics_tab(result)


def _render_propagation_tab(result: Dict[str, Any]) -> None:
    frames = result.get("frames", [])
    regions = result.get("evaluation_regions")

    st.caption(
        f"Total frames: {len(frames)}  "
        f"(fields: {sum(1 for f in frames if f['type'] == 'field')}, "
        f"masks: {sum(1 for f in frames if f['type'] == 'mask')}, "
        f"output: {sum(1 for f in frames if f['type'] == 'output')})"
    )

    # optionally hide intermediate field frames to reduce clutter
    show_all = st.checkbox("Show all intermediate field frames", value=True)
    if show_all:
        display_frames = frames
    else:
        # only show: input, layer arrivals, masks, output
        keep_types = {"mask", "output"}
        keep_keys  = {"input"} | {f"L{i+1}_arr" for i in range(20)}
        display_frames = [
            f for f in frames
            if f["type"] in keep_types or f["key"] in keep_keys
        ]

    # Cache key includes result identity and filter state so each combination
    # renders once and is served from session_state on all subsequent reruns
    # (including the reruns triggered by the player's st_autorefresh), keeping
    # the per-rerun cost well below the 300ms autorefresh interval.
    cache_key = f"_tl_{id(result)}_{int(show_all)}"
    render_propagation_timeline(display_frames, evaluation_regions=regions,
                                cache_key=cache_key)


def _render_player_tab(result: Dict[str, Any]) -> None:
    player_frames = result.get("player_frames", [])
    regions       = result.get("evaluation_regions")
    render_propagation_player(
        player_frames=player_frames,
        evaluation_regions=regions,
        result_id=id(result),
    )


def _render_metrics_tab(result: Dict[str, Any]) -> None:
    metrics = result.get("metrics", {})
    if not metrics:
        st.info("No metrics available.")
        return

    # scalar metrics shown in the summary table
    scalar_keys = {
        "avg_relative_amp_err": "Avg Amplitude Relative Error",
        "avg_amplitudes_diff":  "Avg Amplitude Difference",
        "snr_ratio_full":       "SNR Ratio (linear)",
        "snr_db_full":          "SNR (dB)",
        "throughput":           "Throughput",
        "isolation_db_mean":    "Avg Mode Isolation (dB)",
        "isolation_pct_mean":   "Avg Mode Isolation (%)",
    }
    rows = []
    for key, label in scalar_keys.items():
        val = metrics.get(key)
        if isinstance(val, float) and not (val != val):  # not NaN
            rows.append({"Metric": label, "Value": f"{val:.4f}"})

    st.table(pd.DataFrame(rows))

    # per-mode isolation breakdown (only when .mat data is available)
    iso_db  = metrics.get("isolation_db_per_mode")
    iso_pct = metrics.get("isolation_pct_per_mode")
    if iso_db and iso_pct:
        st.markdown("**Per-mode isolation at training wavelength**")
        iso_rows = [
            {
                "Mode": f"Mode {i + 1}",
                "Isolation (%)": f"{iso_pct[i]:.1f}",
                "Isolation (dB)": f"{iso_db[i]:.2f}",
            }
            for i in range(len(iso_db))
        ]
        st.table(pd.DataFrame(iso_rows))


# -----------------------------------------------------------------------
# Page entry point
# -----------------------------------------------------------------------

def render() -> None:
    st.title("Testing")
    st.divider()

    adapter = Mainfor6Adapter()
    _init_session_state(adapter)

    _render_model_section(adapter)
    _render_config_section()
    _render_run_section(adapter)
    _render_results_section()


render()
