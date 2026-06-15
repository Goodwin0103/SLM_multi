"""propagation_viewer.py -- render the D2NN propagation timeline in Streamlit.

Each "frame" produced by Mainfor6Adapter._collect_propagation_frames is either:
    - type='field'  : intensity only (amplitude squared)
    - type='mask'   : phase only (0 to 2pi, rendered as twilight colormap)
    - type='output' : intensity + optional label for comparison

Output frames are expanded into up to 3 individual grid cells
(Output / Label / Abs Error) so every cell has the same pixel width.
"""

import io
import re
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from streamlit_autorefresh import st_autorefresh


# Maximum columns to show in one row of the timeline before wrapping
_MAX_COLS_PER_ROW = 6
# Figure thumbnail size in inches per column
_THUMB_W = 2.8
_THUMB_H = 2.6


# -----------------------------------------------------------------------
# Single-frame figure builders
# -----------------------------------------------------------------------

def _intensity_image(
    arr: np.ndarray, title: str, z_label: str,
    cmap: str = "inferno",
    evaluation_regions: Optional[List[Tuple[int, int, int, int]]] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(_THUMB_W, _THUMB_H), tight_layout=True)
    vmax = np.percentile(arr, 99.5) if arr.max() > 0 else 1.0
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xlabel(z_label, fontsize=6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    if evaluation_regions:
        for reg in evaluation_regions:
            x0, x1, y0, y1 = reg
            rect = plt.Rectangle(
                (x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                linewidth=0.8, edgecolor="cyan", facecolor="none",
            )
            ax.add_patch(rect)
    return fig


def _error_image(arr: np.ndarray, title: str, z_label: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(_THUMB_W, _THUMB_H), tight_layout=True)
    vmax = np.percentile(arr, 99.5) if arr.max() > 0 else 1.0
    im = ax.imshow(arr, cmap="seismic", vmin=0, vmax=vmax)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xlabel(z_label, fontsize=6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    return fig


def _phase_image(arr: np.ndarray, title: str, z_label: str) -> plt.Figure:
    """Render phase map in [0, 2pi] using the twilight colormap."""
    fig, ax = plt.subplots(figsize=(_THUMB_W, _THUMB_H), tight_layout=True)
    im = ax.imshow(arr, cmap="twilight", vmin=0, vmax=2 * np.pi)
    ax.set_title(title, fontsize=7, pad=2)
    ax.set_xlabel(z_label, fontsize=6)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.set_ticks([0, np.pi, 2 * np.pi])
    cb.set_ticklabels(["0", "pi", "2pi"])
    return fig


# -----------------------------------------------------------------------
# Output frame expansion
# -----------------------------------------------------------------------

def _expand_output_frame(
    frame: Dict[str, Any],
    evaluation_regions: Optional[List[Tuple[int, int, int, int]]],
) -> List[Dict[str, Any]]:
    """Convert one output frame into 1-3 individual renderable frames.

    Always emits the Output cell. If a label array is present, also emits
    Label and Abs Error cells so they sit inline with other grid cells.
    """
    z_um = frame["z_um"]
    intensity = frame["intensity"]
    label = frame.get("label")

    expanded = [
        {
            "type": "_out_intensity",
            "intensity": intensity,
            "description": "Output",
            "z_um": z_um,
            "evaluation_regions": evaluation_regions,
        }
    ]
    if label is not None:
        expanded.append({
            "type": "_out_label",
            "intensity": label,
            "description": "Label",
            "z_um": z_um,
        })
        diff = np.abs(intensity - label)
        expanded.append({
            "type": "_out_error",
            "intensity": diff,
            "description": "Abs Error",
            "z_um": z_um,
        })
    return expanded


# -----------------------------------------------------------------------
# Per-frame PNG bytes helper (used for session_state caching)
# -----------------------------------------------------------------------

def _frame_to_png_bytes(frame: Dict[str, Any]) -> bytes:
    """Render one timeline thumbnail to PNG bytes at low DPI.

    Storing bytes in session_state means subsequent reruns (e.g. those
    triggered by the player's st_autorefresh) just call st.image() with
    pre-built bytes instead of re-running matplotlib, bringing per-rerun
    timeline cost from ~800ms down to ~50ms so it fits inside the 300ms
    autorefresh interval without conflict.
    """
    z_label = f"z = {frame['z_um']:.0f} um"
    ftype = frame["type"]

    if ftype == "mask":
        fig = _phase_image(frame["phase"], frame["description"], z_label)
    elif ftype == "_out_intensity":
        fig = _intensity_image(
            frame["intensity"], frame["description"], z_label,
            cmap="inferno",
            evaluation_regions=frame.get("evaluation_regions"),
        )
    elif ftype == "_out_error":
        fig = _error_image(frame["intensity"], frame["description"], z_label)
    else:
        fig = _intensity_image(frame["intensity"], frame["description"], z_label)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=72, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# -----------------------------------------------------------------------
# Row renderer (uses pre-built PNG bytes)
# -----------------------------------------------------------------------

def _display_image_row(png_row: List[bytes]) -> None:
    """Display a row of pre-rendered PNG bytes in a fixed _MAX_COLS_PER_ROW grid."""
    cols = st.columns(_MAX_COLS_PER_ROW)
    for i, png in enumerate(png_row):
        with cols[i]:
            st.image(png, width="stretch")


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

def render_propagation_timeline(
    frames: List[Dict[str, Any]],
    evaluation_regions: Optional[List[Tuple[int, int, int, int]]] = None,
    cache_key: Optional[str] = None,
) -> None:
    """Render all propagation frames in rows, wrapping at _MAX_COLS_PER_ROW.

    Output frames are expanded inline into Output / Label / Abs Error cells
    so every cell in the grid has the same fixed pixel width.

    Pass cache_key to enable session_state caching: the first call renders all
    frames to PNG bytes (~800ms); every subsequent call just calls st.image()
    with those bytes (~50ms total), safely fitting inside the player's 300ms
    autorefresh interval and preventing gray-page conflicts.
    """
    if not frames:
        st.info("No propagation frames available.")
        return

    # Build or retrieve the PNG-bytes cache
    if cache_key and cache_key in st.session_state:
        image_bytes: List[bytes] = st.session_state[cache_key]
    else:
        # Expand output frames into individual grid cells
        all_frames: List[Dict[str, Any]] = []
        for frame in frames:
            if frame["type"] == "output":
                all_frames.extend(_expand_output_frame(frame, evaluation_regions))
            else:
                all_frames.append(frame)

        image_bytes = [_frame_to_png_bytes(f) for f in all_frames]

        if cache_key:
            st.session_state[cache_key] = image_bytes

    # Display from PNG bytes (fast on every rerun)
    row_buf: List[bytes] = []
    for png in image_bytes:
        row_buf.append(png)
        if len(row_buf) == _MAX_COLS_PER_ROW:
            _display_image_row(row_buf)
            row_buf = []
            st.divider()

    if row_buf:
        _display_image_row(row_buf)


def _player_frame_image(
    frame: Dict[str, Any],
    evaluation_regions: Optional[List[Tuple[int, int, int, int]]] = None,
) -> plt.Figure:
    """Render a single frame at a larger size suitable for the player view."""
    z_label = f"z = {frame['z_um']:.0f} um"
    ftype = frame["type"]
    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    # fixed subplot bounds → consistent PNG dimensions regardless of title/tick text
    fig.subplots_adjust(left=0.06, right=0.88, top=0.92, bottom=0.07)

    if ftype == "mask":
        arr = frame["phase"]
        im = ax.imshow(arr, cmap="twilight", vmin=0, vmax=2 * np.pi)
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.set_ticks([0, np.pi, 2 * np.pi])
        cb.set_ticklabels(["0", "pi", "2pi"])
    elif ftype == "_out_error":
        arr = frame["intensity"]
        vmax = np.percentile(arr, 99.5) if arr.max() > 0 else 1.0
        im = ax.imshow(arr, cmap="seismic", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    else:
        arr = frame["intensity"]
        vmax = np.percentile(arr, 99.5) if arr.max() > 0 else 1.0
        im = ax.imshow(arr, cmap="inferno", vmin=0, vmax=vmax)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        if evaluation_regions and ftype in ("output", "_out_intensity"):
            for reg in evaluation_regions:
                x0, x1, y0, y1 = reg
                rect = plt.Rectangle(
                    (x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0,
                    linewidth=1.0, edgecolor="cyan", facecolor="none",
                )
                ax.add_patch(rect)

    ax.set_title(frame.get("description", ""), fontsize=9, pad=3)
    ax.set_xlabel(z_label, fontsize=8)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    return fig


def _get_frame_event(frame: Dict[str, Any]) -> str:
    """Return a short event label for key frames, empty string otherwise.

    Labels are intentionally short so the caption in col_info (4/7 page width,
    ~61 chars/line) never wraps - wrapping would cause the controls row to grow
    and produce visible layout jitter during playback.
    """
    key = frame.get("key", "")
    if key == "input":
        return "Input"
    if key == "output":
        return "Output"
    m = re.match(r"L(\d+)_arr", key)
    if m:
        return f"L{m.group(1)}"
    return ""


def render_propagation_player(
    player_frames: List[Dict[str, Any]],
    evaluation_regions: Optional[List[Tuple[int, int, int, int]]] = None,
    result_id: int = 0,
) -> None:
    """Interactive single-frame player for the propagation sequence.

    Plays through field/output frames one at a time with Play/Pause/Reset
    controls and a scrubber slider.  Mask frames are skipped in the playback
    but their z-positions are annotated as event labels in the timeline.
    """
    if not player_frames:
        st.info("No player frames available.  Run a test first.")
        return

    # playable = field + output frames; masks are event markers only
    playable = [f for f in player_frames if f["type"] != "mask"]
    if not playable:
        st.info("No playable frames.")
        return

    n = len(playable)

    # -- session state reset when a new result is loaded ------------------
    if st.session_state.get("_player_result_id") != result_id:
        st.session_state["_player_result_id"] = result_id
        st.session_state["player_scrubber"]   = 0
        st.session_state["player_playing"]    = False

    # clamp scrubber after a result with fewer frames
    if st.session_state.get("player_scrubber", 0) >= n:
        st.session_state["player_scrubber"] = 0

    playing = bool(st.session_state.get("player_playing", False))

    # -- auto-advance BEFORE rendering so the rendered frame is current --
    # st_autorefresh is always rendered (even when not playing) so its 26px
    # container is always in the DOM and never causes a layout shift on
    # play/stop. When idle the interval is set to 1 hour so it never fires.
    _FPS = 12
    _interval = max(300, 1000 // _FPS) if playing else 3_600_000
    st_autorefresh(interval=_interval, key="player_ar")

    if playing:
        cur = int(st.session_state.get("player_scrubber", 0))
        if cur < n - 1:
            st.session_state["player_scrubber"] = cur + 1
        else:
            st.session_state["player_playing"] = False
            playing = False

    idx = int(st.session_state.get("player_scrubber", 0))

    # -- controls ---------------------------------------------------------
    col_play, col_info = st.columns([1, 6])
    with col_play:
        if st.button(
            "Pause" if playing else "Play",
            type="primary",
            key="player_playpause",
        ):
            st.session_state["player_playing"] = not playing
            st.rerun()
    with col_info:
        event = _get_frame_event(playable[idx])
        event_suffix = f"  ·  {event}" if event else ""
        st.caption(
            f"Frame {idx + 1} / {n}   |   "
            f"z = {playable[idx]['z_um']:.0f} um"
            + event_suffix
        )

    # -- scrubber (key= mode, no value=, to avoid bounce) -----------------
    st.slider(
        "Frame position",
        min_value=0,
        max_value=n - 1,
        step=1,
        key="player_scrubber",
        label_visibility="collapsed",
    )
    idx = int(st.session_state["player_scrubber"])   # re-read after slider

    # -- current frame image (centered, ~half page wide) ------------------
    # Save with bbox_inches=None so the PNG is always exactly figsize*dpi pixels,
    # bypassing Streamlit's internal bbox_inches='tight' that causes height jitter.
    _, col_img, _ = st.columns([1, 3, 1])
    with col_img:
        fig = _player_frame_image(playable[idx], evaluation_regions)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches=None)
        plt.close(fig)
        buf.seek(0)
        st.image(buf, width="stretch")


def render_phase_mask_gallery(frames: List[Dict[str, Any]]) -> None:
    """Render full-size phase mask plots for each diffraction layer.

    Uses a 3-column grid so several masks can be compared side by side.
    """
    mask_frames = [f for f in frames if f["type"] == "mask"]
    if not mask_frames:
        st.info("No phase masks found in the propagation data.")
        return

    n_cols = min(3, len(mask_frames))
    cols = st.columns(n_cols)
    for i, frame in enumerate(mask_frames):
        with cols[i % n_cols]:
            phase = frame["phase"]
            fig, ax = plt.subplots(figsize=(4.5, 4.0), tight_layout=True)
            im = ax.imshow(phase, cmap="twilight", vmin=0, vmax=2 * np.pi)
            ax.set_title(frame["description"], fontsize=9)
            ax.axis("off")
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_ticks([0, np.pi, 2 * np.pi])
            cb.set_ticklabels(["0", "pi", "2pi"])
            st.pyplot(fig, width="stretch")
            plt.close(fig)
