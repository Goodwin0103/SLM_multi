"""Multi-WL Testing page: Model & Data -> Test Configuration -> Results."""

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

_FRONTEND_DIR = Path(__file__).resolve().parent.parent
if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))

_PROJECT_ROOT = _FRONTEND_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from adapters.mainfor6_wl_adapter import Mainfor6WLAdapter
from components.propagation_viewer import (
    render_propagation_timeline,
    render_propagation_player,
)
from services.config_manager import ConfigManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WL_TEST_CONFIG_PATH = _FRONTEND_DIR / "temp" / "test_config_wl.json"
_WL_TEST_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)

_TESTING_DEFAULTS: Dict[str, Any] = {
    "evaluation_mode":               "eigenmode",
    "mode_index":                    0,
    "num_modes":                     6,
    "circle_detectsize":             10,
    "num_superposition_eval_samples": 1000,
    "superposition_eval_seed":       20240116,
    "z_step_um":                     20.0,
}

_WL_TEST_WIDGET_FIELDS = [
    "mode_index", "num_modes", "circle_detectsize", "z_step_um",
    "num_superposition_eval_samples", "superposition_eval_seed",
]


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_session_state(adapter: Mainfor6WLAdapter) -> None:
    if "wl_test_cfg" not in st.session_state:
        mgr = ConfigManager(_WL_TEST_CONFIG_PATH)
        saved = mgr.load_config()
        st.session_state.wl_test_cfg = mgr.merge_with_defaults(saved, _TESTING_DEFAULTS)

    cfg = st.session_state.wl_test_cfg
    for field in _WL_TEST_WIDGET_FIELDS:
        wkey = f"wltc_{field}"
        if wkey not in st.session_state:
            st.session_state[wkey] = cfg.get(field, _TESTING_DEFAULTS.get(field, 0))

    if "wl_test_result" not in st.session_state:
        st.session_state.wl_test_result = None
    if "wl_selected_ckpt" not in st.session_state:
        st.session_state.wl_selected_ckpt = None
    if "wl_ckpt_meta" not in st.session_state:
        st.session_state.wl_ckpt_meta = {}
    if "wl_mat_file" not in st.session_state:
        st.session_state.wl_mat_file = ""
    if "wl_test_error" not in st.session_state:
        st.session_state.wl_test_error = None


# ---------------------------------------------------------------------------
# Section 1: Model & Data
# ---------------------------------------------------------------------------

def _render_model_section(adapter: Mainfor6WLAdapter) -> None:
    st.subheader("1. Model & Data")

    # -- checkpoint selector -----------------------------------------------
    ckpt_paths = adapter.list_checkpoints()
    if not ckpt_paths:
        st.info("No .pth files auto-detected. Paste a path below or train a model first.")
        st.session_state.wl_selected_ckpt = None
        st.session_state.wl_ckpt_meta = {}
    else:
        ckpt_names = [str(Path(p).relative_to(_PROJECT_ROOT)) for p in ckpt_paths]
        current_rel = (
            str(Path(st.session_state.wl_selected_ckpt).relative_to(_PROJECT_ROOT))
            if st.session_state.wl_selected_ckpt
            else None
        )
        default_idx = ckpt_names.index(current_rel) if current_rel in ckpt_names else 0

        chosen_name = st.selectbox("Checkpoint", ckpt_names, index=default_idx)
        chosen_path = ckpt_paths[ckpt_names.index(chosen_name)]

        if chosen_path != st.session_state.wl_selected_ckpt:
            st.session_state.wl_selected_ckpt = chosen_path
            st.session_state.wl_ckpt_meta = adapter.load_checkpoint_meta(chosen_path)
            st.session_state.wl_test_result = None

    # manual path fallback
    manual_ckpt = st.text_input(
        "Or enter checkpoint path manually",
        placeholder="/path/to/checkpoint.pth",
    )
    if manual_ckpt:
        mp = Path(manual_ckpt.strip())
        if mp.exists() and mp.suffix == ".pth":
            if str(mp) != st.session_state.wl_selected_ckpt:
                st.session_state.wl_selected_ckpt = str(mp)
                st.session_state.wl_ckpt_meta = adapter.load_checkpoint_meta(str(mp))
                st.session_state.wl_test_result = None
                st.rerun()
        elif mp.suffix != ".pth":
            st.caption("Must be a .pth file.")
        else:
            st.caption("File not found.")

    # -- model info --------------------------------------------------------
    meta = st.session_state.wl_ckpt_meta
    if meta:
        # fallback to training config for fields missing from old checkpoints
        _tcfg: Dict[str, Any] = {}
        _tcfg_path = _FRONTEND_DIR / "temp" / "train_config_wl.json"
        if _tcfg_path.exists():
            import json
            try:
                _tcfg = json.loads(_tcfg_path.read_text())
            except Exception:
                pass

        def _show_um(meta_key: str, cfg_key: str) -> str:
            v = meta.get(meta_key)
            if v is not None:
                return f"{float(v) * 1e6:.1f}"
            v = _tcfg.get(cfg_key)
            if v is not None:
                return f"{float(v):.1f}"
            return "?"

        with st.expander("Model info", expanded=False):
            wl_meta = meta.get("wavelengths", None)
            if wl_meta is not None:
                wl_nm_str = ", ".join(f"{w * 1e9:.1f}" for w in np.asarray(wl_meta))
            else:
                ws = _tcfg.get("wl_start_nm", 1550)
                wd = _tcfg.get("wl_spacing_nm", 0.5)
                wc = _tcfg.get("wl_count", 2)
                wl_nm_str = ", ".join(f"{ws + i * wd:.1f}" for i in range(int(wc)))
            info_rows = [
                ("Layers",              str(meta.get("num_layers", "?"))),
                ("Canvas (px)",         str(meta.get("layer_size", "?"))),
                ("Output size (px)",    str(meta.get("out_size", "?"))),
                ("Modes (training)",    str(meta.get("num_modes", "?"))),
                ("Wavelengths",         str(meta.get("num_wavelengths", "?"))),
                ("Wavelengths (nm)",    wl_nm_str),
                ("z layers (um)",       _show_um("z_layers", "z_layers_um")),
                ("z prop (um)",         _show_um("z_prop", "z_prop_um")),
                ("z input->first (um)", _show_um("z_input_to_first", "z_input_to_first_um")),
                ("Pixel size (um)",     _show_um("pixel_size", "pixel_size_um")),
                ("Padding ratio",       f"{float(meta.get('padding_ratio', _tcfg.get('padding_ratio', 0.5))):.2f}"),
                ("Padding ratio out",   f"{float(meta.get('padding_ratio_out', _tcfg.get('padding_ratio_out', 0.5))):.2f}"),
            ]
            st.table(pd.DataFrame(info_rows, columns=["Parameter", "Value"]))

    # -- .mat file ---------------------------------------------------------
    st.markdown("**Mode file (.mat)**")
    default_mat = st.session_state.get("mat_file_path") or ""
    mat_path = st.text_input(
        "Path to .mat file",
        value=st.session_state.wl_mat_file or default_mat,
        placeholder="/path/to/your/mmf_file.mat",
    )
    st.session_state.wl_mat_file = mat_path.strip()

    if st.session_state.wl_mat_file and not Path(st.session_state.wl_mat_file).exists():
        st.warning("File not found at the given path.")

    st.divider()


# ---------------------------------------------------------------------------
# Path geometry diagram
# ---------------------------------------------------------------------------

def _render_path_geometry(
    z_inp: float, z_lyr: float, z_prp: float, n_lyr: int,
) -> None:
    """Draw a horizontal optical-path diagram with distance labels (all in um)."""
    nodes = ["Input"] + [f"L{i + 1}" for i in range(n_lyr)] + ["Detector"]
    gaps = [z_inp] + [z_lyr] * (n_lyr - 1) + [z_prp]

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

    bar_h, label_y, dist_y = 0.55, 0.85, -0.25

    ax.annotate(
        "", xy=(total_z * 1.04, 0), xytext=(-total_z * 0.02, 0),
        arrowprops=dict(arrowstyle="-|>", color="black", lw=1.0),
    )
    ax.text(total_z * 1.05, -0.08, "z", ha="left", va="top", fontsize=8)

    for i, (x, name) in enumerate(zip(xs, nodes)):
        is_layer = name.startswith("L")
        color = "#c0392b" if is_layer else "#2c3e50"
        lw = 2.0 if is_layer else 1.5
        ax.plot([x, x], [-bar_h, bar_h], color=color, lw=lw, ls="-", solid_capstyle="round")
        ax.text(x, label_y, name, ha="center", va="bottom", fontsize=7.5,
                color=color, fontweight="bold" if is_layer else "normal")

        if i < len(nodes) - 1:
            x_next = xs[i + 1]
            dist = gaps[i]
            mid = (x + x_next) / 2
            ax.annotate(
                "", xy=(x_next, dist_y), xytext=(x, dist_y),
                arrowprops=dict(arrowstyle="<->", color="#555555", lw=0.9),
            )
            ax.text(mid, dist_y - 0.18, f"{dist:.0f} um",
                    ha="center", va="top", fontsize=7, color="#333333")

    plt.tight_layout(pad=0.2)
    st.pyplot(fig, width="stretch")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section 2: Test Configuration
# ---------------------------------------------------------------------------

def _render_config_section() -> None:
    st.subheader("2. Test Configuration")
    cfg = st.session_state.wl_test_cfg
    meta = st.session_state.wl_ckpt_meta

    training_modes = int(meta.get("num_modes", 6)) if meta else 6

    col1, col2 = st.columns(2)

    with col1:
        # num_modes (capped by training)
        cur_num = int(st.session_state.get("wltc_num_modes", min(training_modes, 6)))
        if cur_num > training_modes:
            cur_num = training_modes
            st.session_state["wltc_num_modes"] = cur_num
        st.number_input(
            "Number of modes (<= training)", min_value=1, max_value=training_modes,
            value=cur_num, step=1, key="wltc_num_modes",
        )
        cfg["num_modes"] = int(st.session_state["wltc_num_modes"])

        eval_mode = st.selectbox(
            "Evaluation mode", ["eigenmode", "superposition"],
            index=0 if cfg.get("evaluation_mode") == "eigenmode" else 1,
            key="wltc_evaluation_mode",
        )
        cfg["evaluation_mode"] = eval_mode

        if eval_mode == "eigenmode":
            num_modes_val = int(st.session_state.get("wltc_num_modes", 6))
            max_idx = max(0, num_modes_val - 1)
            cur_idx = int(st.session_state.get("wltc_mode_index", 0))
            if cur_idx > max_idx:
                cur_idx = max_idx
                st.session_state["wltc_mode_index"] = cur_idx
            st.slider("Mode index (0-based)", 0, max_idx, cur_idx, key="wltc_mode_index")
            cfg["mode_index"] = int(st.session_state["wltc_mode_index"])
        else:
            st.number_input("Eval samples", 10, 5000, step=100,
                            key="wltc_num_superposition_eval_samples")
            st.number_input("RNG seed", 0, step=1,
                            key="wltc_superposition_eval_seed")

    with col2:
        st.number_input("Detection radius (px)", 2, 50, step=1,
                        key="wltc_circle_detectsize")
        st.number_input("z_step (um)", 1.0, 500.0, format="%.1f", step=1.0,
                        key="wltc_z_step_um")

        # estimated frames preview
        if meta:
            z_inp_um = float(meta.get("z_input_to_first", 0.0)) * 1e6
            z_lyr_um = float(meta.get("z_layers", 0.0)) * 1e6
            z_prp_um = float(meta.get("z_prop", 0.0)) * 1e6
            n_lyr    = int(meta.get("num_layers", 1))
            # fallback to training config if meta values are zero (old checkpoint)
            if z_inp_um == 0 and z_lyr_um == 0 and z_prp_um == 0:
                _tcfg_path = _FRONTEND_DIR / "temp" / "train_config_wl.json"
                if _tcfg_path.exists():
                    import json
                    try:
                        _tcfg = json.loads(_tcfg_path.read_text())
                        z_inp_um = float(_tcfg.get("z_input_to_first_um", 0))
                        z_lyr_um = float(_tcfg.get("z_layers_um", 0))
                        z_prp_um = float(_tcfg.get("z_prop_um", 0))
                    except Exception:
                        pass

            def _count_frames(z_seg, z_step):
                import math
                return max(0, int(z_seg / z_step) - 1)

            z_step_val = float(st.session_state.get("wltc_z_step_um", 20.0))
            n_field = (
                1
                + _count_frames(z_inp_um, z_step_val)
                + n_lyr * (1 + _count_frames(z_lyr_um, z_step_val))
                + n_lyr
                + _count_frames(z_prp_um, z_step_val)
                + 1
            )
            st.caption(f"Estimated frames: ~{n_field}")

    # sync widget keys into config
    for field in _WL_TEST_WIDGET_FIELDS:
        wkey = f"wltc_{field}"
        if wkey in st.session_state:
            cfg[field] = st.session_state[wkey]
    st.session_state.wl_test_cfg = cfg

    # path geometry diagram (full width, below columns)
    if meta:
        z_inp_um = float(meta.get("z_input_to_first", 0.0)) * 1e6
        z_lyr_um = float(meta.get("z_layers", 0.0)) * 1e6
        z_prp_um = float(meta.get("z_prop", 0.0)) * 1e6
        n_lyr    = int(meta.get("num_layers", 1))
        # fallback to training config if meta values are zero (old checkpoint)
        if z_inp_um == 0 and z_lyr_um == 0 and z_prp_um == 0:
            _train_cfg_path = _FRONTEND_DIR / "temp" / "train_config_wl.json"
            if _train_cfg_path.exists():
                import json
                try:
                    _cfg = json.loads(_train_cfg_path.read_text())
                    z_inp_um = float(_cfg.get("z_input_to_first_um", 0))
                    z_lyr_um = float(_cfg.get("z_layers_um", 0))
                    z_prp_um = float(_cfg.get("z_prop_um", 0))
                except Exception:
                    pass
        # only render if we have meaningful distances
        total_z = z_inp_um + z_lyr_um * n_lyr + z_prp_um
        if total_z > 0:
            _render_path_geometry(z_inp_um, z_lyr_um, z_prp_um, n_lyr)

    st.divider()


# ---------------------------------------------------------------------------
# Section 3: Run + Results
# ---------------------------------------------------------------------------

def _render_run_section(adapter: Mainfor6WLAdapter) -> None:
    st.subheader("3. Results")

    ready = (
        st.session_state.wl_selected_ckpt is not None
        and st.session_state.wl_mat_file
        and Path(st.session_state.wl_mat_file).exists()
    )

    col_run, col_clr = st.columns([1, 1])
    with col_run:
        if st.button("Run Test", type="primary", disabled=not ready):
            _do_run_test(adapter)
    with col_clr:
        if st.button("Clear Results", disabled=st.session_state.wl_test_result is None):
            st.session_state.wl_test_result = None
            st.session_state.wl_test_error = None
            st.rerun()

    if not ready:
        st.caption("Select a checkpoint and a valid .mat file to enable Run.")

    if st.session_state.wl_test_error:
        st.error("Test failed — see details below.")
        with st.expander("Error details"):
            st.code(st.session_state.wl_test_error, language="text")

    _render_results_section()


def _do_run_test(adapter: Mainfor6WLAdapter) -> None:
    st.session_state.wl_test_error = None
    st.session_state.wl_test_result = None

    cfg = st.session_state.wl_test_cfg
    ckpt_path = st.session_state.wl_selected_ckpt
    mat_file = st.session_state.wl_mat_file

    try:
        with st.spinner("Running evaluation..."):
            result = adapter.run_test(
                config=cfg,
                checkpoint_path=ckpt_path,
                mat_file=mat_file,
            )
        st.session_state.wl_test_result = result
        ConfigManager(_WL_TEST_CONFIG_PATH).save_config(cfg)
    except Exception:
        import traceback
        st.session_state.wl_test_error = traceback.format_exc()
    st.rerun()


# ---------------------------------------------------------------------------
# Results tabs
# ---------------------------------------------------------------------------

def _render_results_section() -> None:
    result = st.session_state.wl_test_result
    if result is None:
        return

    tab_prop, tab_player, tab_metrics = st.tabs([
        "Propagation Timeline",
        "Propagation Player",
        "Evaluation Metrics",
    ])

    with tab_prop:
        _render_propagation_tab(result)
    with tab_player:
        _render_player_tab(result)
    with tab_metrics:
        _render_metrics_tab(result)


# ---------------------------------------------------------------------------
# Tab: Propagation Timeline
# ---------------------------------------------------------------------------

def _select_wavelength_frames(
    all_frames: List[Dict[str, Any]], wl_idx: int,
) -> List[Dict[str, Any]]:
    """Extract single-wavelength 2D intensity from multi-wavelength frames."""
    out: List[Dict[str, Any]] = []
    for f in all_frames:
        new_f: Dict[str, Any] = {}
        # copy all keys except intensity_wl; replace intensity with selected wavelength
        for k, v in f.items():
            if k == "intensity_wl":
                if v is not None and hasattr(v, "shape") and len(v.shape) == 3:
                    new_f["intensity"] = v[wl_idx]
                # else: mask frames have intensity_wl=None, keep their phase but leave intensity as is
            elif k == "intensity":
                # handled by intensity_wl above; skip the original multi-wl 2D intensity
                if "intensity" not in new_f:
                    new_f[k] = v
            else:
                new_f[k] = v
        out.append(new_f)
    return out


def _render_propagation_tab(result: Dict[str, Any]) -> None:
    frames = result.get("frames", [])
    regions = result.get("evaluation_regions")
    L_val = int(result.get("num_wavelengths", 8))
    wls_nm = result.get("wavelengths_nm", [])
    M_val = int(result.get("num_modes", 6))

    if not frames:
        st.info("No propagation frames available.")
        return

    # wavelength selector
    if L_val > 1:
        wl_idx = st.selectbox(
            "Wavelength channel",
            options=list(range(L_val)),
            format_func=lambda i: (
                f"{wls_nm[i]:.1f} nm" if i < len(wls_nm) else f"Channel {i}"
            ),
            key="wl_timeline_wl",
        )
    else:
        wl_idx = 0

    # mode label
    test_mode = result.get("test_mode", "eigenmode")
    mode_idx = result.get("mode_index", 0)
    if test_mode == "eigenmode":
        st.caption(f"Mode {mode_idx + 1}  |  {L_val} wavelength(s)")
    else:
        st.caption(f"Superposition sample {mode_idx}  |  {L_val} wavelength(s)")

    display_frames = _select_wavelength_frames(frames, wl_idx)

    show_all = st.checkbox("Show all intermediate field frames", value=True)
    if not show_all:
        keep_types = {"mask", "output"}
        keep_keys = {"input"} | {f"L{i + 1}_arr" for i in range(20)}
        display_frames = [
            f for f in display_frames
            if f["type"] in keep_types or f["key"] in keep_keys
        ]

    # subset evaluation regions for selected wavelength
    wl_regions = None
    if regions:
        wl_regions = [regions[m * L_val + wl_idx] for m in range(M_val)]

    cache_key = f"_wltl_{id(result)}_{int(show_all)}_{wl_idx}"
    render_propagation_timeline(display_frames, evaluation_regions=wl_regions, cache_key=cache_key)


# ---------------------------------------------------------------------------
# Tab: Propagation Player
# ---------------------------------------------------------------------------

def _render_player_tab(result: Dict[str, Any]) -> None:
    player_frames = result.get("player_frames", [])
    regions = result.get("evaluation_regions")
    L_val = int(result.get("num_wavelengths", 8))
    wls_nm = result.get("wavelengths_nm", [])
    M_val = int(result.get("num_modes", 6))

    if not player_frames:
        st.info("No player frames available.")
        return

    if L_val > 1:
        wl_idx = st.selectbox(
            "Wavelength for playback",
            options=list(range(L_val)),
            format_func=lambda i: (
                f"{wls_nm[i]:.1f} nm" if i < len(wls_nm) else f"Channel {i}"
            ),
            key="wl_player_wl",
        )
    else:
        wl_idx = 0

    disp_frames = _select_wavelength_frames(player_frames, wl_idx)

    wl_regions = None
    if regions:
        wl_regions = [regions[m * L_val + wl_idx] for m in range(M_val)]

    render_propagation_player(
        player_frames=disp_frames,
        evaluation_regions=wl_regions,
        result_id=id(result),
    )


# ---------------------------------------------------------------------------
# Tab: Evaluation Metrics
# ---------------------------------------------------------------------------

def _render_metrics_tab(result: Dict[str, Any]) -> None:
    metrics = result.get("metrics", {})
    if not metrics:
        st.info("No metrics available.")
        return

    M_val = int(result.get("num_modes", 6))
    L_val = int(result.get("num_wavelengths", 8))
    wls_nm = result.get("wavelengths_nm", [])

    mode_labels = [f"Mode {m + 1}" for m in range(M_val)]
    wl_labels = [f"{w:.1f} nm" for w in wls_nm] if len(wls_nm) == L_val else [
        f"Ch {l}" for l in range(L_val)
    ]

    st.subheader("SNR")
    _plot_heatmap(
        metrics.get("snr_db"),
        title=f"SNR (dB) — mean = {metrics.get('snr_db_mean', 0):.2f} dB",
        x_labels=wl_labels, y_labels=mode_labels,
        xlabel="Wavelength", ylabel="Input Mode", cmap="YlOrRd",
    )

    st.subheader("Mode Isolation")
    _plot_heatmap(
        metrics.get("mode_isolation_db"),
        title=f"Mode Isolation (dB) — mean = {metrics.get('mode_isolation_db_mean', 0):.2f} dB",
        x_labels=wl_labels, y_labels=mode_labels,
        xlabel="Wavelength", ylabel="Input Mode", cmap="RdYlGn",
    )

    st.subheader("Wavelength Isolation")
    wl_iso = metrics.get("wavelength_isolation_db")
    if wl_iso is not None:
        wl_iso_clip = np.clip(np.asarray(wl_iso), -30, 30)
        _plot_heatmap(
            wl_iso_clip,
            title=f"Wavelength Isolation (dB) — mean = {metrics.get('wavelength_isolation_db_mean', 0):.2f} dB",
            x_labels=wl_labels, y_labels=mode_labels,
            xlabel="Target Wavelength", ylabel="Input Mode", cmap="RdYlGn",
        )

    st.subheader("Target / All ROI")
    _plot_bar(
        metrics.get("target_all_roi_ratio"),
        labels=mode_labels,
        title=f"Target / All ROI — mean = {metrics.get('target_all_roi_ratio_mean', 0):.4f}",
        ylabel="Ratio",
    )

    st.subheader("Insertion Loss")
    _plot_bar(
        metrics.get("insertion_loss_db"),
        labels=mode_labels,
        title=f"Insertion Loss (dB) — mean = {metrics.get('insertion_loss_db_mean', 0):.2f} dB",
        ylabel="dB",
    )

    st.subheader("Throughput")
    _plot_bar(
        metrics.get("throughput_per_mode"),
        labels=mode_labels,
        title=f"Throughput (E_out / E_in) — mean = {float(np.mean(metrics.get('throughput_per_mode', [0]))):.4f}",
        ylabel="Ratio",
    )

    # -- mode crosstalk per wavelength ------------------------------------
    st.subheader("Mode Crosstalk per Wavelength")
    ct_per_wl = metrics.get("crosstalk_matrix_per_wl")  # (L, M, M)
    if ct_per_wl is not None:
        ct_arr = np.asarray(ct_per_wl)
        wl_sel = st.selectbox(
            "Select wavelength", options=list(range(L_val)),
            format_func=lambda i: wl_labels[i] if i < len(wl_labels) else f"Ch {i}",
            key="wl_metrics_ct_wl",
        )
        _plot_crosstalk_pair(
            ct_arr[wl_sel],
            title=f"Mode Crosstalk  ({wl_labels[wl_sel] if wl_sel < len(wl_labels) else f'Ch {wl_sel}'})",
            labels=mode_labels,
        )

    # -- wavelength crosstalk per mode ------------------------------------
    st.subheader("Wavelength Crosstalk per Mode")
    ct_wl = metrics.get("crosstalk_matrix_wl")  # (M, L, L)
    if ct_wl is not None:
        ct_wl_arr = np.asarray(ct_wl)
        mode_sel = st.selectbox(
            "Select mode", options=list(range(M_val)),
            format_func=lambda i: mode_labels[i] if i < len(mode_labels) else f"Mode {i + 1}",
            key="wl_metrics_ct_mode",
        )
        _plot_crosstalk_pair(
            ct_wl_arr[mode_sel],
            title=f"Wavelength Crosstalk  ({mode_labels[mode_sel] if mode_sel < len(mode_labels) else f'Mode {mode_sel + 1}'})",
            labels=wl_labels,
        )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_heatmap(
    data, *, title: str, x_labels: List[str], y_labels: List[str],
    xlabel: str, ylabel: str, cmap: str = "RdYlGn",
) -> None:
    if data is None:
        st.info("No data.")
        return
    arr = np.asarray(data)
    fig, ax = plt.subplots(figsize=(max(6, len(x_labels) * 1.2), max(4, len(y_labels) * 0.8)))
    im = ax.imshow(arr, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
    ax.set_yticks(range(len(y_labels)))
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            v = arr[r, c]
            if np.isfinite(v):
                txt = f"{v:.1f}"
                color = "white" if abs(v) > 10 else "black"
                ax.text(c, r, txt, ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_bar(data, *, labels: List[str], title: str, ylabel: str) -> None:
    if data is None:
        st.info("No data.")
        return
    arr = np.asarray(data).flatten()
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(arr))
    ax.bar(x, arr, color="tab:blue", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    if len(arr) > 0 and np.all(np.isfinite(arr)):
        mean_val = float(np.mean(arr))
        ax.axhline(mean_val, color="red", linestyle="--", label=f"mean = {mean_val:.4f}")
        ax.legend(fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def _plot_crosstalk_pair(data, *, title: str, labels: List[str]) -> None:
    if data is None:
        st.info("No data.")
        return
    arr = np.asarray(data)
    arr_db = 10.0 * np.log10(np.clip(arr, 1e-6, None))

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(11, 4.5))

    im0 = ax0.imshow(arr, cmap="viridis", vmin=0, vmax=1)
    ax0.set_title(f"{title} — Linear")
    ax0.set_xlabel("Target"); ax0.set_ylabel("Source")
    ax0.set_xticks(range(len(labels)))
    ax0.set_xticklabels(labels, rotation=45, fontsize=7)
    ax0.set_yticks(range(len(labels)))
    ax0.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)
    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            v = arr[r, c]
            if np.isfinite(v):
                ax0.text(c, r, f"{v:.2f}", ha="center", va="center",
                         color="white" if v < 0.5 else "black", fontsize=7)

    im1 = ax1.imshow(arr_db, cmap="magma", vmin=-30, vmax=0)
    ax1.set_title(f"{title} — dB")
    ax1.set_xlabel("Target"); ax1.set_ylabel("Source")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, fontsize=7)
    ax1.set_yticks(range(len(labels)))
    ax1.set_yticklabels(labels, fontsize=8)
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    for r in range(arr_db.shape[0]):
        for c in range(arr_db.shape[1]):
            v = arr_db[r, c]
            if np.isfinite(v):
                ax1.text(c, r, f"{v:.0f}", ha="center", va="center",
                         color="white" if v < -15 else "black", fontsize=7)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Page entry point
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("Multi-WL Testing")
    st.divider()

    adapter = Mainfor6WLAdapter()
    _init_session_state(adapter)

    _render_model_section(adapter)
    _render_config_section()
    _render_run_section(adapter)


render()
