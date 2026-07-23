"""
Centralized label generation for multi-wavelength D2NN.

Uses band-per-wavelength layout:
  - Canvas is split into num_wavelengths horizontal bands.
  - Within each band, targets (modes or groups) are arranged in an optimal grid.
  - ncols/nrows are auto-computed from the band aspect ratio.

Supports:
  - Label types: "modes" (one position per mode), "mode_groups" (one position per group)
  - Label shapes: "circle", "eigenmode", "distinct" (equal-area multi-shape)
  - Per-wavelength radius
"""

from __future__ import annotations

import numpy as np

from odnn_generate_label import (
    compose_labels_from_patterns,
    generate_detector_patterns,
)


def generate_labels(
    *,
    out_size: int,
    num_modes: int,
    num_wavelengths: int,
    label_config: dict,
    modes_field: np.ndarray | None = None,
    mode_info: dict | None = None,
    show_debug: bool = False,
    debug_save_path: str = "debug_multiwl_labels.png",
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[int]]:
    """Unified label generation for multi-wavelength training and evaluation.

    Returns
    -------
    patterns : (out_size, out_size, num_modes * num_wavelengths) float32
    evaluation_regions : list[(x0, x1, y0, y1)]
    per_label_radii : list[int]
    """
    H = W = out_size

    # --- resolve label type and groups ---
    label_type = label_config.get("label_type", "modes")
    if label_type == "mode_groups":
        mode_groups = label_config.get("mode_groups", None)
        if mode_groups is None and mode_info is not None:
            mode_groups = _groups_from_mode_info(mode_info, num_modes)
        if mode_groups is None:
            raise ValueError("label_type='mode_groups' requires mode_groups or mode_info")
    else:
        mode_groups = [[m] for m in range(num_modes)]

    # mode -> group index map
    mode_to_group: dict[int, int] = {}
    for g_idx, modes_in_group in enumerate(mode_groups):
        for m in modes_in_group:
            mode_to_group[m] = g_idx
    num_groups = len(mode_groups)
    # items to place in each band: groups (for mode_groups) or individual modes
    items_per_band = num_groups if label_type == "mode_groups" else num_modes

    # --- resolve radii ---
    radius_per_wl = label_config.get("radius_per_wl", None)
    focus_size = label_config.get("focus_size", 10)
    margin_ratio = label_config.get("margin_ratio", 0.2)
    circle_detectsize = label_config.get("circle_detectsize", focus_size * 2 + 5)

    if radius_per_wl is None:
        radius_per_wl = [focus_size] * num_wavelengths
    else:
        radius_per_wl = [int(r) for r in radius_per_wl]
        if len(radius_per_wl) != num_wavelengths:
            radius_per_wl = [radius_per_wl[0]] * num_wavelengths

    max_r = max(radius_per_wl)
    margin_x = max(int(W * margin_ratio), max_r + 5)
    margin_y = max(int(H * margin_ratio), max_r + 5)

    # --- band-per-wavelength layout ---
    # Canvas is split into num_wavelengths horizontal bands with gaps between them.
    # Within each band, items_per_band targets are grid-arranged.
    inner_margin = max_r + 3
    band_gap = inner_margin * 3  # vertical space between adjacent bands
    total_gap = band_gap * max(0, num_wavelengths - 1)
    avail_y = H - 2 * margin_y - total_gap
    band_h = avail_y / max(num_wavelengths, 1)
    band_w = W - 2 * margin_x
    ncols, nrows = _pick_grid(items_per_band, band_w, band_h)

    # Pre-compute center positions: dict[(item_idx, wl_idx)] -> (cy, cx)
    centers: dict[tuple[int, int], tuple[int, int]] = {}
    for wl_idx in range(num_wavelengths):
        band_y0 = margin_y + wl_idx * (band_h + band_gap)
        band_y1 = band_y0 + band_h
        if ncols > 1:
            xs_arr = np.linspace(margin_x, W - 1 - margin_x, ncols)
        else:
            xs_arr = np.array([W / 2.0])
        xs_arr = xs_arr[::-1]  # mirror for visualization consistency
        if nrows > 1:
            ys_arr = np.linspace(band_y0 + inner_margin, band_y1 - inner_margin, nrows)
        else:
            ys_arr = np.array([(band_y0 + band_y1) / 2.0])
        # Items per row, with last row centred if short
        _item_idx = 0
        for row in range(nrows):
            row_items = min(ncols, items_per_band - row * ncols)
            offset = (ncols - row_items) / 2.0  # centre the shorter last row
            for col_local in range(row_items):
                cx = int(round(xs_arr[int(col_local + offset)]))
                cy = int(round(ys_arr[row]))
                centers[(_item_idx, wl_idx)] = (cy, cx)
                _item_idx += 1

    # --- generate patterns ---
    label_shape = label_config.get("label_shape", "circle")
    total_labels = num_modes * num_wavelengths
    patterns = np.zeros((H, W, total_labels), dtype=np.float32)
    per_label_radii: list[int] = []

    if label_shape == "circle":
        yy, xx = np.ogrid[:H, :W]
        for mode_idx in range(num_modes):
            g_idx = mode_to_group[mode_idx]
            item_idx = g_idx if label_type == "mode_groups" else mode_idx
            for wl_idx in range(num_wavelengths):
                idx = mode_idx * num_wavelengths + wl_idx
                cy, cx = centers[(item_idx, wl_idx)]
                r = radius_per_wl[wl_idx]
                mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
                patterns[:, :, idx] = mask.astype(np.float32)
        per_label_radii = [
            radius_per_wl[wl_idx]
            for g_idx in range(num_groups)
            for wl_idx in range(num_wavelengths)
        ]

    elif label_shape == "eigenmode":
        if modes_field is None:
            raise ValueError("modes_field is required for eigenmode label_shape")
        mode_amps = np.abs(modes_field)
        for m in range(mode_amps.shape[2]):
            vmin = mode_amps[:, :, m].min()
            vmax = mode_amps[:, :, m].max()
            if vmax > vmin:
                mode_amps[:, :, m] = (mode_amps[:, :, m] - vmin) / (vmax - vmin)

        for mode_idx in range(num_modes):
            g_idx = mode_to_group[mode_idx]
            item_idx = g_idx if label_type == "mode_groups" else mode_idx
            repr_mode = mode_groups[g_idx][0]
            pattern_src = mode_amps[:, :, repr_mode]
            for wl_idx in range(num_wavelengths):
                idx = mode_idx * num_wavelengths + wl_idx
                cy, cx = centers[(item_idx, wl_idx)]
                r = radius_per_wl[wl_idx]
                target_sz = max(r * 2, 4)
                _place_pattern(patterns, idx, H, W, cy, cx, pattern_src, target_sz)
        per_label_radii = [
            radius_per_wl[wl_idx]
            for g_idx in range(num_groups)
            for wl_idx in range(num_wavelengths)
        ]

    elif label_shape == "distinct":
        shapes_list = label_config.get("shapes_list",
                                       ["circle", "square", "diamond", "plus", "ring"])
        for mode_idx in range(num_modes):
            g_idx = mode_to_group[mode_idx]
            item_idx = g_idx if label_type == "mode_groups" else mode_idx
            shape_name = shapes_list[item_idx % len(shapes_list)]
            for wl_idx in range(num_wavelengths):
                idx = mode_idx * num_wavelengths + wl_idx
                cy, cx = centers[(item_idx, wl_idx)]
                r = radius_per_wl[wl_idx]
                sz = max(r * 2, 8)
                target_px = int(np.pi * r * r)
                single_pattern = generate_detector_patterns(
                    sz, sz, N=1, shape=shape_name,
                    equal_area=True, target_area=target_px,
                )[:, :, 0]
                y0 = cy - sz // 2
                y1 = y0 + sz
                x0 = cx - sz // 2
                x1 = x0 + sz
                sp = single_pattern
                if y0 < 0:
                    sp = sp[-y0:, :]; y0 = 0
                if y1 > H:
                    sp = sp[:H - y0, :]; y1 = H
                if x0 < 0:
                    sp = sp[:, -x0:]; x0 = 0
                if x1 > W:
                    sp = sp[:, :W - x0]; x1 = W
                patterns[y0:y1, x0:x1, idx] = np.maximum(
                    patterns[y0:y1, x0:x1, idx], sp
                )
        per_label_radii = [
            radius_per_wl[wl_idx]
            for g_idx in range(num_groups)
            for wl_idx in range(num_wavelengths)
        ]

    else:
        raise ValueError(f"Unknown label_shape: {label_shape}")

    # --- evaluation regions ---
    evaluation_regions: list[tuple[int, int, int, int]] = []
    detect_half = max(max_r, circle_detectsize // 2)
    for mode_idx in range(num_modes):
        g_idx = mode_to_group[mode_idx]
        item_idx = g_idx if label_type == "mode_groups" else mode_idx
        for wl_idx in range(num_wavelengths):
            cy, cx = centers[(item_idx, wl_idx)]
            x0 = max(0, cx - detect_half)
            x1 = min(W, cx + detect_half)
            y0 = max(0, cy - detect_half)
            y1 = min(H, cy + detect_half)
            evaluation_regions.append((x0, x1, y0, y1))

    # --- debug ---
    if show_debug:
        _debug_plot(H, W, patterns, centers, evaluation_regions,
                    num_modes, num_wavelengths, mode_to_group,
                    label_type, label_shape, num_groups,
                    radius_per_wl, margin_x, margin_y,
                    ncols, nrows, band_h, inner_margin,
                    debug_save_path)

    return patterns, evaluation_regions, per_label_radii


def _pick_grid(items: int, band_w: float, band_h: float) -> tuple[int, int]:
    """Pick (ncols, nrows) for *items* targets inside a band of
    given width and height.

    Scores candidate column counts by (a) evenness — how close the
    last row is to a full row — and (b) aspect-ratio match so cells
    stay roughly square.  Single-row layouts are penalised when
    *items* ≥ 9 to avoid long horizontal chains.
    """
    best_ncols, best_score = 1, -1.0
    ideal_aspect = max(band_w, 1.0) / max(band_h, 1.0)

    for nc in range(1, items + 1):
        nr = int(np.ceil(items / nc))
        last_row = items - (nr - 1) * nc
        evenness = last_row / nc  # 1.0 when items % nc == 0
        aspect = nc / max(nr, 1)
        aspect_score = 1.0 / (1.0 + abs(aspect - ideal_aspect))
        # penalise single-row layouts for many items
        single_row_penalty = 0.7 if nr == 1 and items >= 9 else 1.0
        score = (evenness * 0.6 + aspect_score * 0.4) * single_row_penalty
        if score > best_score:
            best_score = score
            best_ncols = nc

    best_nrows = int(np.ceil(items / best_ncols))
    return best_ncols, best_nrows


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _place_pattern(patterns, idx, H, W, cy, cx, src, target_sz):
    """Place a source pattern (any H×W) into patterns[:,:,idx] centered at (cy,cx)."""
    from scipy.ndimage import zoom
    sh, sw = src.shape
    scale = target_sz / max(sh, sw)
    if scale != 1.0:
        src = zoom(src.astype(np.float64), scale, order=1).astype(np.float32)
        sh, sw = src.shape
    y0 = cy - sh // 2
    y1 = y0 + sh
    x0 = cx - sw // 2
    x1 = x0 + sw
    sp = src
    if y0 < 0:
        sp = sp[-y0:, :]; y0 = 0
    if y1 > H:
        sp = sp[:H - y0, :]; y1 = H
    if x0 < 0:
        sp = sp[:, -x0:]; x0 = 0
    if x1 > W:
        sp = sp[:, :W - x0]; x1 = W
    patterns[y0:y1, x0:x1, idx] = np.maximum(patterns[y0:y1, x0:x1, idx], sp)


def _debug_plot(H, W, patterns, centers, eval_regions,
                num_modes, num_wavelengths, mode_to_group,
                label_type, label_shape, num_groups,
                radius_per_wl, margin_x, margin_y,
                ncols, nrows, band_h, inner_margin,
                save_path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 9))
    composite = patterns.sum(axis=2)
    if composite.max() > 0:
        composite = composite / composite.max()
    ax.imshow(composite, cmap='gray')

    drawn_items: set = set()
    for mode_idx in range(num_modes):
        g_idx = mode_to_group[mode_idx]
        item_idx = g_idx if label_type == "mode_groups" else mode_idx
        for wl_idx in range(num_wavelengths):
            key = (item_idx, wl_idx)
            if key in drawn_items:
                continue
            drawn_items.add(key)
            cy, cx = centers[(item_idx, wl_idx)]
            r = radius_per_wl[wl_idx]
            color = plt.cm.tab10(item_idx % 10)
            ax.add_patch(plt.Circle((cx, cy), r, fill=False, color=color, linewidth=1.5))
            if label_type == "mode_groups":
                group_modes = [m for m, g in mode_to_group.items() if g == item_idx]
                label = f"G{item_idx}\n{group_modes}"
            else:
                label = f"M{item_idx}"
            ax.text(cx, cy + r + 2, label, ha='center', va='top',
                    color='white', fontsize=5, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.75))
            if wl_idx == 0:
                ax.text(cx - r - 4, cy, f"W{wl_idx}", ha='right', va='center',
                        color='cyan', fontsize=6)

    # band separators
    for wl_idx in range(1, num_wavelengths):
        y = margin_y + wl_idx * band_h
        ax.axhline(y, color='cyan', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.axhline(margin_y, color='yellow', linestyle=':', linewidth=0.5)
    ax.axhline(H - margin_y, color='yellow', linestyle=':', linewidth=0.5)
    ax.axvline(margin_x, color='yellow', linestyle=':', linewidth=0.5)
    ax.axvline(W - margin_x, color='yellow', linestyle=':', linewidth=0.5)

    ax.set_title(
        f"Labels: {label_type} | {label_shape} | "
        f"{num_modes}M x {num_wavelengths}WL | "
        f"{num_groups if label_type == 'mode_groups' else num_modes} items/band "
        f"({nrows}x{ncols})",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Debug label layout saved -> {save_path}")


def _groups_from_mode_info(mode_info, num_modes: int) -> list[list[int]]:
    """Convert MATLAB mode_info struct array to mode_groups list.

    Handles three formats produced by different .mat loaders:
      - scipy loadmat (struct_as_record=False): ndarray of struct objects
      - mat73 loadmat: dict of lists, e.g. {'group': [1, 1, 2, ...]}
      - list of dicts (mat73 older versions)
    """
    groups: dict[int, list[int]] = {}

    # mat73 format: dict of lists (most common fallback path)
    if isinstance(mode_info, dict) and 'group' in mode_info and not hasattr(mode_info, 'group'):
        group_vals = mode_info['group']
        if isinstance(group_vals, (list, np.ndarray)):
            for m in range(num_modes):
                g = int(group_vals[m]) if m < len(group_vals) else 0
                groups.setdefault(g, []).append(m)
            return [groups[g] for g in sorted(groups.keys())]

    # scipy / list-of-dicts format
    for m in range(num_modes):
        try:
            if hasattr(mode_info, 'group'):
                g = int(mode_info[m].group) if hasattr(mode_info[m], 'group') else int(mode_info['group'][m])
            elif isinstance(mode_info, (list, np.ndarray)):
                item = mode_info[m]
                if hasattr(item, 'group'):
                    g = int(item.group)
                elif isinstance(item, dict):
                    g = int(item.get('group', 0))
                else:
                    g = 0
            else:
                g = 0
        except (IndexError, KeyError, TypeError, ValueError):
            g = 0
        groups.setdefault(g, []).append(m)
    return [groups[g] for g in sorted(groups.keys())]
