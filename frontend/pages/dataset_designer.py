"""Dataset & Label Designer.

Section 1 — Dataset Generation (MATLAB on remote server)
Section 2 — Label Customization (params affecting preview)
Section 3 — Dataset & Label Preview + export to train_config_wl.json
"""

import base64
import json
import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

_FRONTEND_DIR = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _FRONTEND_DIR.parent

if str(_FRONTEND_DIR) not in sys.path:
    sys.path.insert(0, str(_FRONTEND_DIR))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from odnn_io import load_complex_modes_from_mat
from adapters.remote_adapter import RemoteAdapter, load_remote_config

TEMP_DIR = _FRONTEND_DIR / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CONFIG_PATH = TEMP_DIR / "train_config_wl.json"

SHAPE_OPTIONS = ["circle", "square", "diamond", "plus", "ring", "larger_circle", "small_circle"]


# =============================================================================
# Session state
# =============================================================================

def _init_state() -> None:
    defaults = {
        # dataset generation
        "dd_mat_file_path": None,
        "dd_mat_remote_path": None,
        "dd_matlab_running": False,
        "dd_matlab_pid": None,
        "dd_matlab_log_path": None,
        "dd_matlab_remote_dir": None,
        "dd_matlab_log_lines": [],
        "dd_loaded_field_size": 0,
        "dd_loaded_total_modes": 0,
        "dd_mode_info": None,
        # label customization
        "dd_out_size": 600,
        "dd_wl_start_nm": 1550,
        "dd_wl_spacing_nm": 0.5,
        "dd_wl_count": 2,
        "dd_base_wavelength_idx": 0,
        "dd_num_modes": 0,
        "dd_label_type": "modes",
        "dd_mode_groups": None,
        "dd_label_shape": "circle",
        "dd_shapes_list": ["circle", "square", "diamond", "plus", "ring"],
        "dd_margin_ratio": 0.20,
        "dd_focus_size": 5,
        "dd_circle_detectsize": 25,
        "dd_radius_per_wl": None,
        # MATLAB params
        "dd_na": 0.2,
        "dd_r_core": 25,
        "dd_wavelength_um": 1.550,
        "dd_n_core": 1.45,
        "dd_pd_factor": 1.05,
        "dd_basis": "LP",
        "dd_matlab_save_folder": "~/eigenmodes_generation_grin/mmf_data",
        "dd_matlab_toolbox_path": "~/eigenmodes_generation_grin",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# =============================================================================
# MATLAB generation helpers
# =============================================================================

def _get_remote_adapter():
    cfg = load_remote_config()
    if not cfg:
        return None
    return RemoteAdapter(
        host=cfg["host"], user=cfg["user"],
        project_dir=cfg.get("project_dir", f"/home/{cfg['user']}/odnn_project"),
        workspace_dir=cfg.get("workspace_dir", f"/home/{cfg['user']}/odnn_workspace"),
        conda_env=cfg.get("conda_env", "odnn"),
        port=cfg.get("port", 22),
    )


def _build_matlab_script() -> str:
    field_sz = st.session_state.get("dd_field_size_override", None)
    grid_size = int(field_sz) if field_sz else 128
    return f"""grid_size = {grid_size};
show_Plots = 0;
PD_factor = {float(st.session_state.dd_pd_factor)};

F.NA = {float(st.session_state.dd_na)};
F.r_core = {float(st.session_state.dd_r_core)};
F.wavelength = {float(st.session_state.dd_wavelength_um)};
F.n_core = {float(st.session_state.dd_n_core)};

basis = '{st.session_state.dd_basis}';
save_folder = '{st.session_state.dd_matlab_save_folder}';

if ~exist(save_folder, 'dir')
    mkdir(save_folder);
end

addpath('{st.session_state.dd_matlab_toolbox_path}');

[modes_field, mode_info, n_modes] = calc_GRIN_modes( ...
    F.r_core, F.n_core, F.NA, F.wavelength, ...
    grid_size, PD_factor * F.r_core, basis);

filename = sprintf('mmf_%dmodes_GRIN_%s_%d_PD%.2f_r%d.mat', ...
                   n_modes, basis, grid_size, PD_factor, F.r_core);
full_path = fullfile(save_folder, filename);
save(full_path, 'modes_field', 'mode_info', 'n_modes', 'basis', 'F');
fprintf('SAVED: %s\\n', full_path);
fprintf('N_MODES: %d\\n', n_modes);
exit;
"""


def _do_generate_matlab() -> None:
    adapter = _get_remote_adapter()
    if adapter is None:
        st.error("Remote server not configured. Please go to Settings page first.")
        return
    script = _build_matlab_script()
    remote_dir = "~/matlab_gen"
    try:
        ssh_target = adapter.ssh_target
        port = adapter.port
        subprocess.run(
            ["ssh", "-p", str(port), "-o", "ConnectTimeout=10",
             "-o", "StrictHostKeyChecking=accept-new",
             ssh_target, f"mkdir -p {remote_dir}"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        encoded = base64.b64encode(script.encode()).decode()
        subprocess.run(
            ["ssh", "-p", str(port), "-o", "ConnectTimeout=10",
             "-o", "StrictHostKeyChecking=accept-new",
             ssh_target, f"echo '{encoded}' | base64 -d > {remote_dir}/gen_script.m"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        log_path = f"{remote_dir}/gen.log"
        result = subprocess.run(
            ["ssh", "-p", str(port), "-o", "ConnectTimeout=10",
             "-o", "StrictHostKeyChecking=accept-new",
             ssh_target,
             f"cd {remote_dir} && nohup matlab -nodisplay -nosplash -nodesktop "
             f"-r \"gen_script; exit\" > {log_path} 2>&1 & echo PID:$!"],
            check=True, capture_output=True, text=True, timeout=30,
        )
        pid_line = [l for l in result.stdout.splitlines() if "PID:" in l]
        if pid_line:
            pid = pid_line[0].split("PID:")[-1].strip()
            st.session_state.dd_matlab_pid = pid
            st.session_state.dd_matlab_log_path = log_path
            st.session_state.dd_matlab_remote_dir = remote_dir
            st.session_state.dd_matlab_running = True
            st.session_state.dd_matlab_log_lines = []
            st.success(f"MATLAB started (PID={pid}).")
        else:
            st.error("Failed to capture MATLAB PID.")
    except subprocess.CalledProcessError as e:
        st.error(f"SSH failed: {e.stderr if hasattr(e, 'stderr') else str(e)}")


def _poll_matlab_log(adapter) -> None:
    if not st.session_state.dd_matlab_running:
        return
    log_path = st.session_state.dd_matlab_log_path
    try:
        result = subprocess.run(
            ["ssh", "-p", str(adapter.port), "-o", "ConnectTimeout=10",
             "-o", "StrictHostKeyChecking=accept-new",
             adapter.ssh_target, f"tail -n 50 {log_path}"],
            check=True, capture_output=True, text=True, timeout=15,
        )
        lines = result.stdout.splitlines()
        st.session_state.dd_matlab_log_lines = lines
        full = "\n".join(lines)
        if "SAVED:" in full:
            saved_line = [l for l in lines if "SAVED:" in l][0]
            mat_path = saved_line.split("SAVED:")[-1].strip()
            st.session_state.dd_mat_remote_path = mat_path
            st.session_state.dd_matlab_running = False
            st.success(f"MATLAB finished. File: {mat_path}")
            # Auto-download
            local_path = TEMP_DIR / Path(mat_path).name
            try:
                subprocess.run(
                    ["scp", "-P", str(adapter.port), "-o", "ConnectTimeout=30",
                     "-o", "StrictHostKeyChecking=accept-new",
                     f"{adapter.ssh_target}:{mat_path}", str(local_path)],
                    check=True, capture_output=True, text=True, timeout=120,
                )
                st.session_state.dd_mat_file_path = str(local_path)
                st.success(f"Downloaded to {local_path}")
            except subprocess.CalledProcessError as e:
                st.warning(f"Auto-download failed. Server path: {mat_path}")
        if any(kw in full.lower() for kw in ["error", "undefined", "unrecognized"]):
            st.warning("MATLAB log shows potential errors.")
    except Exception:
        pass


# =============================================================================
# Auto-load .mat metadata
# =============================================================================

def _auto_load_mat_meta(mat_path: str) -> None:
    """Load modes + mode_info from a .mat file, populate session state."""
    modes, mode_info = load_complex_modes_from_mat(mat_path, key="modes_field")
    st.session_state.dd_loaded_field_size = int(modes.shape[0])
    st.session_state.dd_loaded_total_modes = int(modes.shape[2])
    st.session_state.dd_mode_info = mode_info
    if st.session_state.dd_num_modes == 0:
        st.session_state.dd_num_modes = st.session_state.dd_loaded_total_modes


# =============================================================================
# Preview (cached)
# =============================================================================

@st.cache_data(show_spinner="Rendering preview...")
def _render_preview(
    mat_path: str,
    num_modes: int,
    wl_count: int,
    wl_start_nm: float,
    wl_spacing_nm: float,
    out_size: int,
    label_config_json: str,
    _cache_buster: float = 0.0,
):
    from label_designer import generate_labels

    modes, mode_info = load_complex_modes_from_mat(mat_path, key="modes_field")
    field_size = int(modes.shape[0])
    total_modes = int(modes.shape[2])
    M = min(num_modes, total_modes)

    label_config = json.loads(label_config_json)
    patterns, eval_regions, radii = generate_labels(
        out_size=out_size, num_modes=M, num_wavelengths=wl_count,
        label_config=label_config,
        modes_field=modes if label_config.get("label_shape") == "eigenmode" else None,
        mode_info=mode_info,
        show_debug=False,
    )

    mmf = np.abs(modes[:, :, :M].transpose(2, 0, 1))
    wavelengths_nm = [wl_start_nm + i * wl_spacing_nm for i in range(wl_count)]

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14, 7))

    # Left: dataset modes
    ncols_modes = min(5, M)
    nrows_modes = int(np.ceil(M / ncols_modes))
    mode_grid = np.zeros((nrows_modes * field_size, ncols_modes * field_size))
    for m in range(M):
        r_i = m // ncols_modes; c_i = m % ncols_modes
        y0 = r_i * field_size; y1 = y0 + field_size
        x0 = c_i * field_size; x1 = x0 + field_size
        mode_grid[y0:y1, x0:x1] = mmf[m]
    ax_left.imshow(mode_grid, cmap="inferno")
    for m in range(M):
        r_i = m // ncols_modes; c_i = m % ncols_modes
        g_label = ""
        if label_config.get("label_type") == "mode_groups" and mode_info is not None:
            try:
                from label_designer import _groups_from_mode_info
                grps = _groups_from_mode_info(mode_info, M)
                for gi, gms in enumerate(grps):
                    if m in gms:
                        g_label = f" G{gi}"
                        break
            except Exception:
                pass
        ax_left.text(c_i * field_size + 3, r_i * field_size + 3, f"M{m}{g_label}",
                    color="white", fontsize=5, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.1", facecolor="black", alpha=0.55))
    ax_left.set_title(f"Dataset: {M} modes ({field_size}x{field_size})", fontsize=11)
    ax_left.axis("off")

    # Right: label overlay
    composite = np.maximum(patterns.sum(axis=2), 0)
    if composite.max() > 0:
        composite = composite / composite.max()
    ax_right.imshow(composite, cmap="gray")

    detect_half = label_config.get("circle_detectsize", 25) // 2
    # Build item→group mapping for consistent coloring
    mg_list = label_config.get("mode_groups")
    if label_config.get("label_type") == "mode_groups" and mg_list:
        group_of_mode = {}
        for gi, gms in enumerate(mg_list):
            for m in gms:
                group_of_mode[m] = gi
        items_per_band = len(mg_list)
    else:
        group_of_mode = {m: m for m in range(M)}
        items_per_band = M

    drawn_items: set = set()
    for mode_idx in range(M):
        item_idx = group_of_mode.get(mode_idx, mode_idx)
        for wl_idx in range(wl_count):
            key = (item_idx, wl_idx)
            if key in drawn_items:
                continue
            drawn_items.add(key)
            idx = mode_idx * wl_count + wl_idx
            if idx < len(eval_regions):
                x0, x1, y0, y1 = eval_regions[idx]
                cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
                r = label_config.get("focus_size", 5)
                color = plt.cm.tab10(item_idx % 10)
                # detection window
                rect = plt.Rectangle(
                    (cx - detect_half, cy - detect_half),
                    detect_half * 2, detect_half * 2,
                    fill=False, edgecolor=color, linestyle='--', linewidth=0.6, alpha=0.4,
                )
                ax_right.add_patch(rect)
                # focus circle
                ax_right.add_patch(plt.Circle((cx, cy), r, fill=False, color=color, linewidth=1.5, alpha=0.85))
                if mode_idx == 0:
                    ax_right.text(cx + max(r, detect_half) + 3, cy,
                                f"W{wl_idx}\n{wavelengths_nm[wl_idx]:.1f}nm",
                                color="white", fontsize=6, fontweight="bold",
                                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.7))

    ax_right.set_title(
        f"Label: {label_config.get('label_type')} | {label_config.get('label_shape')} | "
        f"{items_per_band} items/band x {wl_count} bands | "
        f"detect={label_config.get('circle_detectsize', 25)}px",
        fontsize=10,
    )
    ax_right.axis("off")
    plt.tight_layout()
    return fig, field_size, total_modes, mode_info


# =============================================================================
# Section 1 — Dataset Generation
# =============================================================================

def _section_dataset_gen() -> None:
    st.header("1. Dataset Generation")

    remote_cfg = load_remote_config()
    if not remote_cfg:
        st.warning("Remote server not configured. Please go to Settings page first.")
        return
    st.info(f"Server: **{remote_cfg.get('user')}@{remote_cfg.get('host')}**")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.session_state.dd_na = st.number_input(
            "NA", value=st.session_state.dd_na, min_value=0.01, max_value=1.0, step=0.01,
            key="dd_w_na",
        )
        st.session_state.dd_r_core = st.number_input(
            "Core radius (um)", value=st.session_state.dd_r_core, min_value=1, max_value=100, step=1,
            key="dd_w_r_core",
        )
        st.session_state.dd_wavelength_um = st.number_input(
            "Design wavelength (um)", value=st.session_state.dd_wavelength_um,
            min_value=0.3, max_value=3.0, step=0.001, key="dd_w_mat_wl",
        )
    with col2:
        st.session_state.dd_n_core = st.number_input(
            "Core index", value=st.session_state.dd_n_core, min_value=1.0, max_value=5.0, step=0.01,
            key="dd_w_n_core",
        )
        st.session_state.dd_pd_factor = st.number_input(
            "PD factor", value=st.session_state.dd_pd_factor, min_value=1.0, max_value=3.0, step=0.01,
            key="dd_w_pd",
        )
        st.session_state.dd_basis = st.selectbox(
            "Mode basis", options=["LP", "LP_complex", "OAM"], index=0, key="dd_w_basis",
        )
    with col3:
        st.session_state.dd_matlab_save_folder = st.text_input(
            "Save folder on server", value=st.session_state.dd_matlab_save_folder,
            key="dd_w_save_folder",
        )
        st.session_state.dd_matlab_toolbox_path = st.text_input(
            "MATLAB toolbox path", value=st.session_state.dd_matlab_toolbox_path,
            key="dd_w_toolbox_path",
        )
        st.session_state.dd_field_size_override = st.number_input(
            "Grid size", value=st.session_state.get("dd_field_size_override", 128),
            min_value=32, max_value=512, step=16, key="dd_w_grid_size",
        )

    col_a, col_b = st.columns([1, 3])
    with col_a:
        if st.button("Generate on Server", type="primary", use_container_width=True,
                     disabled=st.session_state.dd_matlab_running):
            _do_generate_matlab()

    # MATLAB log
    if st.session_state.dd_matlab_running or st.session_state.dd_matlab_log_lines:
        adapter = _get_remote_adapter()
        if adapter and st.session_state.dd_matlab_running:
            _poll_matlab_log(adapter)
            time.sleep(0.2)
            st.rerun()
        with st.expander("MATLAB Output Log", expanded=st.session_state.dd_matlab_running):
            log_text = "\n".join(st.session_state.dd_matlab_log_lines) if st.session_state.dd_matlab_log_lines else "..."
            st.code(log_text, language="text")
            if st.session_state.dd_matlab_running:
                st.caption("Auto-refreshing every 3s...")

    # Auto-load after generation
    if st.session_state.dd_mat_file_path and st.session_state.dd_loaded_total_modes == 0:
        try:
            _auto_load_mat_meta(st.session_state.dd_mat_file_path)
        except Exception as e:
            st.error(f"Failed to load generated .mat: {e}")

    if st.session_state.dd_mat_file_path:
        st.success(f"Dataset: **{Path(st.session_state.dd_mat_file_path).name}** "
                   f"({st.session_state.dd_loaded_total_modes} modes, "
                   f"{st.session_state.dd_loaded_field_size}x{st.session_state.dd_loaded_field_size})")


# =============================================================================
# Section 2 — Label Customization
# =============================================================================

def _section_label_customization() -> None:
    st.header("2. Label Customization")

    if not st.session_state.dd_mat_file_path:
        st.info("Generate a dataset in Section 1 first.")
        return

    # Ensure .mat metadata is loaded
    if st.session_state.dd_loaded_total_modes == 0:
        try:
            _auto_load_mat_meta(st.session_state.dd_mat_file_path)
        except Exception as e:
            st.error(f"Failed to load .mat: {e}")
            return

    total_modes = st.session_state.dd_loaded_total_modes
    field_size = st.session_state.dd_loaded_field_size
    st.caption(f"Loaded: {total_modes} modes, {field_size}x{field_size} px")

    # --- Preview-affecting parameters ---
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown("**Canvas**")
        st.session_state.dd_out_size = st.number_input(
            "out_size (px)", value=st.session_state.dd_out_size,
            min_value=64, max_value=1024, step=16, key="dd_w_out_size",
        )

    with col_b:
        st.markdown("**Wavelengths**")
        st.session_state.dd_wl_start_nm = st.number_input(
            "wl_start_nm", value=int(st.session_state.dd_wl_start_nm),
            min_value=400, max_value=3000, step=1, key="dd_w_wl_start",
        )
        st.session_state.dd_wl_spacing_nm = st.number_input(
            "wl_spacing_nm", value=st.session_state.dd_wl_spacing_nm,
            min_value=0.1, max_value=1000.0, step=0.5, key="dd_w_wl_spacing",
        )
        st.session_state.dd_wl_count = st.number_input(
            "wl_count", value=int(st.session_state.dd_wl_count),
            min_value=1, max_value=20, step=1, key="dd_w_wl_count",
        )
        st.session_state.dd_base_wavelength_idx = st.number_input(
            "base_wavelength_idx", value=int(st.session_state.dd_base_wavelength_idx),
            min_value=0, max_value=st.session_state.dd_wl_count - 1, step=1,
            key="dd_w_base_wl_idx",
        )

    with col_c:
        st.markdown("**Modes**")
        st.session_state.dd_num_modes = st.number_input(
            "num_modes", value=min(st.session_state.dd_num_modes or total_modes, total_modes),
            min_value=1, max_value=total_modes, step=1, key="dd_w_num_modes",
        )

    # --- Label type ---
    st.divider()
    st.markdown("**Label Type**")
    lt = st.segmented_control("Label type", options=["modes", "mode_groups"], key="dd_w_label_type")
    if lt is not None:
        st.session_state.dd_label_type = lt

    if st.session_state.dd_label_type == "mode_groups":
        mode_info = st.session_state.dd_mode_info
        M = st.session_state.dd_num_modes
        if mode_info is not None:
            try:
                from label_designer import _groups_from_mode_info
                groups = _groups_from_mode_info(mode_info, M)
                st.session_state.dd_mode_groups = groups
                st.info("Groups detected from dataset: " +
                        ", ".join(f"G{g}: {ms}" for g, ms in enumerate(groups)))
            except Exception:
                st.warning("Could not parse mode_info groups.")
        else:
            st.warning("No mode_info in .mat file. Define groups manually (one per line, comma-separated):")
            groups_text = st.text_area(
                "Mode groups", value=st.session_state.get("dd_mode_groups_text", ""), height=120,
                key="dd_w_mode_groups_text",
                help="Example:\n0\n1,2\n3,4,5",
            )
            if groups_text.strip():
                try:
                    parsed = []
                    for line in groups_text.strip().splitlines():
                        indices = [int(x.strip()) for x in line.split(",") if x.strip()]
                        if indices:
                            parsed.append(indices)
                    st.session_state.dd_mode_groups = parsed
                except ValueError:
                    st.error("Invalid format.")

    # --- Label shape ---
    st.markdown("**Label Shape**")
    ls = st.segmented_control("Label shape", options=["circle", "eigenmode", "distinct"], key="dd_w_label_shape")
    if ls is not None:
        st.session_state.dd_label_shape = ls

    if st.session_state.dd_label_shape == "distinct":
        selected = st.multiselect(
            "Shapes (cycled in order)", options=SHAPE_OPTIONS,
            default=st.session_state.dd_shapes_list, key="dd_w_shapes_list",
        )
        if selected:
            st.session_state.dd_shapes_list = selected

    # --- Spacing & Size ---
    st.markdown("**Spacing & Size**")
    col_s1, col_s2, col_s3 = st.columns(3)
    with col_s1:
        st.session_state.dd_margin_ratio = st.slider(
            "margin_ratio", min_value=0.05, max_value=0.50,
            value=st.session_state.dd_margin_ratio, step=0.05, key="dd_w_margin",
        )
    with col_s2:
        st.session_state.dd_focus_size = st.slider(
            "focus_size (px)", min_value=2, max_value=50,
            value=st.session_state.dd_focus_size, step=1, key="dd_w_focus",
        )
    with col_s3:
        st.session_state.dd_circle_detectsize = st.slider(
            "detection window (px)", min_value=4, max_value=100,
            value=st.session_state.dd_circle_detectsize, step=1, key="dd_w_detect",
        )

    # Per-wavelength radius
    wl_count = st.session_state.dd_wl_count
    if wl_count > 1:
        with st.expander("Per-wavelength radius multipliers"):
            radius_mult = []
            cols = st.columns(min(wl_count, 8))
            for wl_idx in range(wl_count):
                wl_nm = st.session_state.dd_wl_start_nm + wl_idx * st.session_state.dd_wl_spacing_nm
                with cols[wl_idx % len(cols)]:
                    mult = st.slider(
                        f"W{wl_idx} ({wl_nm:.1f}nm)", min_value=0.5, max_value=3.0,
                        value=st.session_state.get(f"dd_w_rm_{wl_idx}", 1.0),
                        step=0.1, key=f"dd_w_rm_{wl_idx}",
                    )
                    radius_mult.append(mult)
            st.session_state.dd_radius_per_wl = [
                int(st.session_state.dd_focus_size * m) for m in radius_mult
            ]
    else:
        st.session_state.dd_radius_per_wl = [st.session_state.dd_focus_size]


# =============================================================================
# Section 3 — Preview & Export
# =============================================================================

def _section_preview_and_export() -> None:
    st.header("3. Dataset & Label Preview")

    if not st.session_state.dd_mat_file_path:
        st.info("Generate a dataset first (Section 1).")
        return

    label_config = {
        "label_type": st.session_state.dd_label_type,
        "mode_groups": st.session_state.dd_mode_groups,
        "label_shape": st.session_state.dd_label_shape,
        "shapes_list": st.session_state.dd_shapes_list,
        "margin_ratio": st.session_state.dd_margin_ratio,
        "focus_size": st.session_state.dd_focus_size,
        "radius_per_wl": st.session_state.dd_radius_per_wl,
        "circle_detectsize": st.session_state.dd_circle_detectsize,
    }

    try:
        fig, field_size, total_modes, mode_info = _render_preview(
            mat_path=st.session_state.dd_mat_file_path,
            num_modes=st.session_state.dd_num_modes,
            wl_count=st.session_state.dd_wl_count,
            wl_start_nm=st.session_state.dd_wl_start_nm,
            wl_spacing_nm=st.session_state.dd_wl_spacing_nm,
            out_size=st.session_state.dd_out_size,
            label_config_json=json.dumps(label_config),
            _cache_buster=time.time(),
        )
        st.pyplot(fig)
        plt.close(fig)
    except Exception as e:
        st.error(f"Preview failed: {e}")
        import traceback
        st.code(traceback.format_exc())

    # Info
    items = (len(st.session_state.dd_mode_groups)
             if st.session_state.dd_label_type == "mode_groups" and st.session_state.dd_mode_groups
             else st.session_state.dd_num_modes)
    st.caption(
        f"Layout: {items} items/band × {st.session_state.dd_wl_count} bands | "
        f"Total labels: {st.session_state.dd_num_modes * st.session_state.dd_wl_count} | "
        f"out_size={st.session_state.dd_out_size} | "
        f"focus={st.session_state.dd_focus_size}px | "
        f"detect={st.session_state.dd_circle_detectsize}px"
    )

    # --- Export to train_config_wl.json ---
    st.divider()
    col_btn, col_msg = st.columns([1, 2])
    with col_btn:
        if st.button("Save to train_config_wl.json", type="primary", use_container_width=True):
            # Preserve only known training-only keys from existing config
            _TRAINING_KEYS = {
                "layer_size", "num_layers_list", "batch_size", "epochs",
                "learning_rate", "lr_gamma", "phase_option",
                "z_layers_um", "z_prop_um", "z_input_to_first_um", "pixel_size_um",
                "padding_ratio", "padding_ratio_out",
                "training_dataset_mode", "evaluation_mode",
                "num_superposition_eval_samples", "num_data",
            }
            existing = {}
            if TRAIN_CONFIG_PATH.exists():
                try:
                    with open(TRAIN_CONFIG_PATH) as f:
                        old = json.load(f)
                    existing = {k: v for k, v in old.items() if k in _TRAINING_KEYS}
                except Exception:
                    pass

            # Build clean config: training keys + designer keys (no garbage)
            out = dict(existing)
            out.update({
                "out_size": st.session_state.dd_out_size,
                "wl_start_nm": st.session_state.dd_wl_start_nm,
                "wl_spacing_nm": st.session_state.dd_wl_spacing_nm,
                "wl_count": st.session_state.dd_wl_count,
                "base_wavelength_idx": st.session_state.dd_base_wavelength_idx,
                "num_modes": st.session_state.dd_num_modes,
                "field_size": st.session_state.dd_loaded_field_size,
                "circle_focus_radius": st.session_state.dd_focus_size,
                "margin_ratio": st.session_state.dd_margin_ratio,
                "circle_detectsize": st.session_state.dd_circle_detectsize,
                "label_config": {
                    "label_type": st.session_state.dd_label_type,
                    "mode_groups": st.session_state.dd_mode_groups,
                    "label_shape": st.session_state.dd_label_shape,
                    "shapes_list": st.session_state.dd_shapes_list,
                    "margin_ratio": st.session_state.dd_margin_ratio,
                    "focus_size": st.session_state.dd_focus_size,
                    "radius_per_wl": st.session_state.dd_radius_per_wl,
                    "circle_detectsize": st.session_state.dd_circle_detectsize,
                },
                "mat_file_path": st.session_state.dd_mat_file_path,
                "mat_file_remote_path": st.session_state.dd_mat_remote_path,
            })
            with open(TRAIN_CONFIG_PATH, "w") as f:
                json.dump(out, f, indent=2)
            st.success("Saved to train_config_wl.json. Training page will use these parameters.")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    st.title("Dataset & Label Designer")
    st.caption("Generate .mat datasets on the server, then customize label patterns for training.")

    _init_state()

    _section_dataset_gen()
    st.divider()
    _section_label_customization()
    st.divider()
    _section_preview_and_export()


if __name__ == "__main__":
    main()
