"""GPU Monitor page: real-time GPU status on the remote server.

Queries ``nvidia-smi`` over SSH every 5 seconds and displays utilisation,
memory, and temperature for each GPU.  The GPU with the most free memory
is highlighted as the recommended target for new training jobs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from adapters.remote_adapter import RemoteAdapter, load_remote_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_remote_adapter() -> RemoteAdapter | None:
    """Build a RemoteAdapter from saved settings, or return None."""
    cfg = load_remote_config()
    if not cfg:
        return None
    try:
        return RemoteAdapter(
            host=cfg["host"],
            user=cfg["user"],
            project_dir=cfg.get("project_dir", ""),
            workspace_dir=cfg.get("workspace_dir", ""),
            conda_env=cfg.get("conda_env", "odnn"),
            port=int(cfg.get("port", 22)),
        )
    except Exception:
        return None


def _best_gpu(gpus: List[Dict[str, Any]]) -> int:
    """Return the index of the GPU with the most free memory."""
    if not gpus:
        return -1
    best = max(gpus, key=lambda g: g.get("memory_free_mib", 0))
    return best["index"]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("GPU Monitor")
    st.divider()

    adapter = _get_remote_adapter()
    if adapter is None:
        st.warning("Remote server not configured.  Go to **Settings** first.")
        return

    # auto-refresh every 5 seconds
    st_autorefresh(interval=5000, key="gpu_monitor_refresh")

    gpus = adapter.get_gpu_status()
    if not gpus:
        st.info("No GPU data.  Is nvidia-smi available on the server?")
        return

    best_idx = _best_gpu(gpus)

    # summary row
    total_gpus = len(gpus)
    total_mem_used = sum(g["memory_used_mib"] for g in gpus)
    total_mem_all = sum(g["memory_total_mib"] for g in gpus)
    avg_util = sum(g["utilization_gpu"] for g in gpus) / max(total_gpus, 1)

    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    col_s1.metric("GPUs", str(total_gpus))
    col_s2.metric("Avg Utilisation", f"{avg_util:.0f} %")
    col_s3.metric("Total VRAM Used", f"{total_mem_used / 1024:.1f} / {total_mem_all / 1024:.1f} GiB")
    col_s4.metric("Best GPU", f"#{best_idx}" if best_idx >= 0 else "N/A")

    st.divider()

    # per-GPU cards
    for gpu in gpus:
        idx = gpu["index"]
        is_best = idx == best_idx

        border_color = "#27ae60" if is_best else "#cccccc"
        bg_hint = "#f0fff0" if is_best else "transparent"

        with st.container(border=True):
            col_name, col_util, col_mem, col_temp = st.columns([2, 2, 2, 1])

            with col_name:
                label = f"GPU {idx}  {'(Recommended)' if is_best else ''}"
                st.markdown(f"**{gpu['name']}**")
                st.caption(label)

            with col_util:
                util = gpu["utilization_gpu"]
                st.metric("Utilisation", f"{util} %")
                st.progress(util / 100.0)

            with col_mem:
                mem_used_gb = gpu["memory_used_mib"] / 1024
                mem_total_gb = gpu["memory_total_mib"] / 1024
                mem_pct = mem_used_gb / max(mem_total_gb, 1)
                st.metric("VRAM", f"{mem_used_gb:.1f} / {mem_total_gb:.1f} GiB")
                st.progress(mem_pct)

            with col_temp:
                temp = gpu.get("temperature_gpu", 0)
                temp_emoji = "cool" if temp < 50 else ("warm" if temp < 75 else "hot")
                st.metric("Temp", f"{temp} C")


render()
