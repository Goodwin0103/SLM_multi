"""Settings page: configure remote server connection.

Settings are saved to ``~/.odnn/remote_config.json`` (outside the repo)
so they never get committed to git and are unique to each user's Mac.
"""

from __future__ import annotations

import getpass
import subprocess
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from adapters.remote_adapter import load_remote_config, save_remote_config


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def _init_state() -> None:
    if "remote_config" not in st.session_state:
        st.session_state.remote_config = load_remote_config()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _test_ssh(host: str, user: str, port: int) -> tuple[bool, str]:
    """Test whether we can SSH to the server."""
    cmd = [
        "ssh",
        "-p", str(port),
        "-o", "ConnectTimeout=10",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "BatchMode=yes",
        f"{user}@{host}",
        "echo OK",
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15,
        )
        if result.returncode == 0 and "OK" in result.stdout:
            return True, "Connection successful."
        else:
            return False, result.stderr.strip() or "SSH returned non-zero exit code."
    except subprocess.TimeoutExpired:
        return False, "Connection timed out (10 seconds)."
    except FileNotFoundError:
        return False, "The 'ssh' command was not found on this system."
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

def render() -> None:
    st.title("Settings")
    st.divider()

    _init_state()
    cfg: Dict[str, Any] = dict(st.session_state.remote_config)

    st.subheader("Remote Server Connection")

    col1, col2 = st.columns(2)

    with col1:
        cfg["host"] = st.text_input(
            "Server hostname / IP",
            value=cfg.get("host", "141.30.127.3"),
        )
        cfg["port"] = st.number_input(
            "SSH port", min_value=1, max_value=65535,
            value=int(cfg.get("port", 22)), step=1,
        )

    with col2:
        cfg["user"] = st.text_input(
            "SSH username",
            value=cfg.get("user", getpass.getuser()),
        )
        cfg["conda_env"] = st.text_input(
            "Conda environment name",
            value=cfg.get("conda_env", "odnn"),
        )

    st.divider()

    st.subheader("Remote Paths")

    col_a, col_b = st.columns(2)
    with col_a:
        cfg["project_dir"] = st.text_input(
            "Project directory (code)",
            value=cfg.get("project_dir", f"/home/{cfg.get('user', getpass.getuser())}/odnn_project"),
            help="Absolute path to the cloned ODNN repo on the server.",
        )
    with col_b:
        cfg["workspace_dir"] = st.text_input(
            "Workspace directory (outputs)",
            value=cfg.get("workspace_dir", f"/home/{cfg.get('user', getpass.getuser())}/odnn_workspace"),
            help="Absolute path for training outputs: uploads, runs, logs, checkpoints.",
        )

    st.divider()

    col_save, col_test, _ = st.columns([1, 1, 2])

    with col_save:
        if st.button("Save Settings", type="primary"):
            save_remote_config(cfg)
            st.session_state.remote_config = cfg
            st.success("Settings saved to ~/.odnn/remote_config.json")

    with col_test:
        if st.button("Test Connection"):
            with st.spinner("Testing SSH connection..."):
                ok, msg = _test_ssh(
                    host=cfg.get("host", "141.30.127.3"),
                    user=cfg.get("user", getpass.getuser()),
                    port=int(cfg.get("port", 22)),
                )
            if ok:
                st.success(msg)
            else:
                st.error(f"Connection failed: {msg}")

    # show SSH setup hint on first visit
    if not cfg:
        st.info(
            "Before using remote training, configure your SSH key:\n\n"
            "```bash\n"
            "# On your Mac:\n"
            "ssh-keygen -t ed25519   # if you don't have a key yet\n"
            "ssh-copy-id <user>@<server>\n"
            "```\n\n"
            "This only needs to be done once per machine."
        )


render()
