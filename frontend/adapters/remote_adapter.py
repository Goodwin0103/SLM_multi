"""Remote adapter for ODNN training on GPU servers via SSH.

All operations use system ``ssh`` / ``scp`` commands via ``subprocess``.
No extra Python dependencies (no paramiko).  Authentication relies on the
user's existing ``~/.ssh/id_*`` keys and ``~/.ssh/config``.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from adapters.base_adapter import BaseODNNAdapter


# ---------------------------------------------------------------------------
# User-local config (stored outside the repo so it never gets committed)
# ---------------------------------------------------------------------------
_ODNN_CONFIG_DIR = Path.home() / ".odnn"
_REMOTE_CONFIG_PATH = _ODNN_CONFIG_DIR / "remote_config.json"


def load_remote_config() -> Dict[str, Any]:
    """Read the remote connection config from ``~/.odnn/remote_config.json``."""
    if not _REMOTE_CONFIG_PATH.exists():
        return {}
    try:
        return json.loads(_REMOTE_CONFIG_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def save_remote_config(cfg: Dict[str, Any]) -> None:
    """Persist remote connection config to ``~/.odnn/remote_config.json``."""
    _ODNN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _REMOTE_CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# ---------------------------------------------------------------------------
# RemoteAdapter
# ---------------------------------------------------------------------------

class RemoteAdapter(BaseODNNAdapter):
    """Adapter that runs training on a remote GPU server over SSH.

    Each training run gets a unique *run_id* (timestamp-based) so that
    configs, logs, and checkpoints are isolated and never overwrite each
    other.

    Parameters
    ----------
    host:
        Server hostname or IP (e.g. ``"141.30.127.3"``).
    user:
        SSH username.
    project_dir:
        Absolute path to the cloned ``ODNN`` repo on the server
        (e.g. ``"/home/jslai/ODNN"``).
    workspace_dir:
        Absolute path to the per-user workspace on the server
        (e.g. ``"/home/jslai/odnn_workspace"``).
    conda_env:
        Name of the conda environment on the server (e.g. ``"odnn"``).
    port:
        SSH port (default 22).
    """

    def __init__(
        self,
        host: str,
        user: str,
        project_dir: str,
        workspace_dir: str,
        conda_env: str,
        port: int = 22,
    ) -> None:
        self.host = host
        self.user = user
        self.port = port
        self.ssh_target = f"{user}@{host}"
        self.project_dir = project_dir.rstrip("/")
        self.workspace_dir = workspace_dir.rstrip("/")
        self.conda_env = conda_env

        self._current_run_id: Optional[str] = None
        self._remote_upload_dir: str = f"{self.workspace_dir}/uploads"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ssh(
        self, remote_cmd: str, check: bool = True, timeout: int = 30
    ) -> subprocess.CompletedProcess:
        """Run a command on the remote server via SSH."""
        cmd = [
            "ssh",
            "-p", str(self.port),
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            self.ssh_target,
            remote_cmd,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            raise RuntimeError(
                f"SSH command failed (exit {result.returncode}): "
                + (result.stderr or result.stdout or "unknown").strip()
            )
        return result

    def _ssh_no_check(self, remote_cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
        """Run a remote command without raising on non-zero exit."""
        return self._ssh(remote_cmd, check=False, timeout=timeout)

    def _scp_upload(self, local_path: str, remote_path: str) -> None:
        """Copy a local file to the remote server."""
        cmd = [
            "scp",
            "-P", str(self.port),
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            local_path,
            f"{self.ssh_target}:{remote_path}",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"scp upload failed (exit {result.returncode}): "
                + (result.stderr or result.stdout or "unknown").strip()
            )

    def _scp_download(self, remote_path: str, local_path: str) -> None:
        """Copy a file from the remote server to the local machine."""
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "scp",
            "-P", str(self.port),
            "-o", "ConnectTimeout=10",
            "-o", "StrictHostKeyChecking=accept-new",
            "-o", "BatchMode=yes",
            f"{self.ssh_target}:{remote_path}",
            local_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(
                f"scp download failed (exit {result.returncode}): "
                + (result.stderr or result.stdout or "unknown").strip()
            )

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def load_default_config(self) -> Dict[str, Any]:
        """Return the same training defaults as Mainfor6WLAdapter."""
        # Avoid hard import at module level — this adapter file may be loaded
        # even when the local training deps are not installed.
        try:
            from adapters.mainfor6_wl_adapter import Mainfor6WLAdapter
            return Mainfor6WLAdapter().load_default_config()
        except ImportError:
            return {}

    # ------------------------------------------------------------------
    # Training control
    # ------------------------------------------------------------------

    def start_training(self, config: Dict[str, Any], mat_file: str = "") -> str:
        """Upload config & data, auto-select GPU, launch training on server.

        Writes a *launcher script* to the server so that the slow ``conda
        activate`` happens inside a background process.  The SSH call returns
        in under 10 seconds regardless of conda startup time.

        Returns a ``job_id`` string: ``"user@host:run_id:remote_pid"``.
        """
        import base64

        # 1. Generate unique run_id ------------------------------------------
        run_id = time.strftime("run_%Y%m%d_%H%M%S")
        self._current_run_id = run_id
        run_dir = f"{self.workspace_dir}/runs/{run_id}"

        # 2. Handle .mat file (one scp if local) -----------------------------
        remote_mat_path: str
        if mat_file and Path(mat_file).exists():
            mat_name = Path(mat_file).name
            self._scp_upload(mat_file, f"{self._remote_upload_dir}/{mat_name}")
            remote_mat_path = f"{self._remote_upload_dir}/{mat_name}"
            # Persist paths back to config so Save Config / next session picks them up
            config["mat_file_path"] = str(Path(mat_file))
            config["mat_file_remote_path"] = remote_mat_path
        else:
            remote_mat_path = mat_file

        # 3. Pick best GPU (separate SSH call, quick) ------------------------
        gpu_id = self._pick_best_gpu()

        # 4. Build config + launcher script (both base64) --------------------
        config_json = json.dumps(config, indent=2)
        config_b64 = base64.b64encode(config_json.encode()).decode()

        # The launcher script does the slow conda activation *inside* the
        # background process, so the SSH call returns immediately.
        launcher_script = (
            f"#!/bin/bash\n"
            f"set -e\n"
            f"source $HOME/miniconda3/etc/profile.d/conda.sh\n"
            f"conda activate {self.conda_env}\n"
            f"cd {self.project_dir}\n"
            f"CUDA_VISIBLE_DEVICES={gpu_id} "
            f"python mainfor6_wl.py "
            f"  --config {run_dir}/train_config.json "
            f"  --mat_file {remote_mat_path} "
            f"  --output_dir {run_dir}\n"
        )
        script_b64 = base64.b64encode(launcher_script.encode()).decode()

        # 5. Single SSH call: mkdir + write files + launch (returns fast) ----
        launch_cmd = (
            f"mkdir -p {self._remote_upload_dir} {run_dir}/{{logs,checkpoints}} && "
            f"echo '{config_b64}' | base64 -d > {run_dir}/train_config.json && "
            f"echo '{script_b64}' | base64 -d > {run_dir}/launch.sh ; "
            f"nohup bash {run_dir}/launch.sh > {run_dir}/logs/training_wl.log 2>&1 < /dev/null & "
            f"echo $!"
        )
        # Cmd line length, so we use Python to pipe to SSH
        result = subprocess.run(
            [
                "ssh",
                "-p", str(self.port),
                "-o", "ConnectTimeout=10",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "BatchMode=yes",
                self.ssh_target,
                launch_cmd,
            ],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"SSH launch failed (exit {result.returncode}): "
                + (result.stderr or result.stdout or "unknown").strip()
            )
        remote_pid = result.stdout.strip()
        if not remote_pid.isdigit():
            raise RuntimeError(
                f"Failed to get PID from launch. stdout={result.stdout!r} stderr={result.stderr!r}"
            )

        return f"{self.ssh_target}:{run_id}:{remote_pid}"

    def stop_training(self, job_id: str) -> None:
        """Kill the training process group on the server.

        Uses ``kill -TERM -PGID`` so that DataLoader worker children are
        cleaned up together with the parent process.
        """
        _user_host, run_id, remote_pid = self._parse_job_id(job_id)
        kill_cmd = (
            f"pgid=$(ps -o pgid= -p {remote_pid} 2>/dev/null | grep -o '[0-9]*' | head -1); "
            f"[ -n \"$pgid\" ] && kill -TERM -$pgid; "
            f"true"
        )
        self._ssh_no_check(kill_cmd)

    def is_training_alive(self, job_id: str) -> bool:
        """Check whether the remote training process is still running."""
        _user_host, run_id, remote_pid = self._parse_job_id(job_id)
        result = self._ssh_no_check(
            f"ps -p {remote_pid} > /dev/null && echo ALIVE || echo DEAD"
        )
        return "ALIVE" in result.stdout

    # ------------------------------------------------------------------
    # Log & metrics access (remote)
    # ------------------------------------------------------------------

    def read_log_tail(self, n: int = 50) -> List[str]:
        """Return the last *n* lines of the current run's training log."""
        if not self._current_run_id:
            return []
        run_dir = f"{self.workspace_dir}/runs/{self._current_run_id}"
        result = self._ssh_no_check(
            f"tail -n {n} {run_dir}/logs/training_wl.log 2>/dev/null || true"
        )
        return result.stdout.splitlines()

    def fetch_metrics_jsonl(self) -> str:
        """Fetch the current run's full metrics JSONL from the server.

        Metrics files are small (<200 KB even for 600 epochs); fetching the
        whole file guarantees all layers are included in the chart data.
        """
        if not self._current_run_id:
            return ""
        run_dir = f"{self.workspace_dir}/runs/{self._current_run_id}"
        result = self._ssh_no_check(
            f"cat {run_dir}/logs/metrics_wl.jsonl 2>/dev/null || true"
        )
        return result.stdout

    # ------------------------------------------------------------------
    # Checkpoint discovery & download
    # ------------------------------------------------------------------

    def list_checkpoints(self) -> List[str]:
        """Return remote paths of all .pth files across all runs."""
        result = self._ssh_no_check(
            f"find {self.workspace_dir}/runs -name '*.pth' -type f 2>/dev/null | sort || true"
        )
        return [p for p in result.stdout.strip().split("\n") if p]

    def load_checkpoint_meta(self, pth_path: str) -> Dict[str, Any]:
        """Read the 'meta' dict from a remote checkpoint via a one-shot SSH Python call."""
        py_script = (
            f"import torch, json; "
            f"c = torch.load('{pth_path}', map_location='cpu', weights_only=False); "
            f"print(json.dumps(dict(c.get('meta', {{}}))))"
        )
        result = self._ssh_no_check(
            f"source $HOME/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate {self.conda_env} && "
            f"python -c '{py_script}'"
        )
        try:
            return json.loads(result.stdout.strip())
        except (json.JSONDecodeError, Exception):
            return {}

    def download_checkpoint(self, remote_path: str, local_dir: str) -> str:
        """Download a .pth from the server to a local directory.

        Returns the local path of the downloaded file.
        """
        local_dir_p = Path(local_dir)
        local_dir_p.mkdir(parents=True, exist_ok=True)
        fname = Path(remote_path).name
        local_path = str(local_dir_p / fname)
        self._scp_download(remote_path, local_path)
        return local_path

    # ------------------------------------------------------------------
    # Run history
    # ------------------------------------------------------------------

    def list_runs(self) -> List[Dict[str, Any]]:
        """List all training runs on the server with their status."""
        result = self._ssh_no_check(
            f"ls -1dt {self.workspace_dir}/runs/*/ 2>/dev/null | head -50 || true"
        )
        runs: List[Dict[str, Any]] = []
        for line in result.stdout.strip().split("\n"):
            line = line.strip().rstrip("/")
            if not line:
                continue
            run_id = Path(line).name
            run_dir = f"{self.workspace_dir}/runs/{run_id}"

            # check if metrics exist
            has_metrics_result = self._ssh_no_check(
                f"test -f {run_dir}/logs/metrics_wl.jsonl && echo YES || echo NO"
            )
            has_metrics = "YES" in has_metrics_result.stdout

            runs.append({
                "run_id": run_id,
                "run_dir": run_dir,
                "has_metrics": has_metrics,
            })
        return runs

    # ------------------------------------------------------------------
    # GPU status
    # ------------------------------------------------------------------

    def get_gpu_status(self) -> List[Dict[str, Any]]:
        """Query GPU utilisation via ``nvidia-smi`` on the remote server."""
        result = self._ssh_no_check(
            "nvidia-smi "
            "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,"
            "temperature.gpu,memory.free "
            "--format=csv,noheader 2>/dev/null || true"
        )
        gpus: List[Dict[str, Any]] = []
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            try:
                mem_used = int(parts[3].replace(" MiB", "").replace(" GiB", ""))
                mem_total = int(parts[4].replace(" MiB", "").replace(" GiB", ""))
                temp_gpu = int(parts[5].replace(" C", "").replace("C", ""))
                mem_free = int(parts[6].replace(" MiB", "").replace(" GiB", ""))
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_gpu": int(parts[2].replace(" %", "").replace("%", "")),
                    "memory_used_mib": mem_used,
                    "memory_total_mib": mem_total,
                    "temperature_gpu": temp_gpu,
                    "memory_free_mib": mem_free,
                })
            except (ValueError, IndexError):
                continue
        return gpus

    # ------------------------------------------------------------------
    # MATLAB generation
    # ------------------------------------------------------------------

    def run_matlab_generation(
        self, matlab_script: str, remote_dir: str = "~/matlab_gen"
    ) -> Dict[str, Any]:
        """Execute a MATLAB script on the remote server.

        Parameters
        ----------
        matlab_script:
            Full MATLAB script content as a string.
        remote_dir:
            Directory on the server to stage the script and log files.

        Returns
        -------
        dict with keys: ``pid``, ``log_path``, ``remote_dir``
        """
        import base64

        # Ensure remote directory exists
        self._ssh(f"mkdir -p {remote_dir}")

        # Upload the script via base64 pipe
        encoded = base64.b64encode(matlab_script.encode()).decode()
        self._ssh(
            f"echo '{encoded}' | base64 -d > {remote_dir}/gen_script.m",
            timeout=15,
        )

        log_path = f"{remote_dir}/gen.log"

        # Launch MATLAB in background with nohup
        result = self._ssh(
            f"cd {remote_dir} && "
            f"nohup matlab -nodisplay -nosplash -nodesktop "
            f"-r \"gen_script; exit\" > {log_path} 2>&1 & echo PID:$!",
            timeout=30,
        )

        pid_str = ""
        for line in result.stdout.splitlines():
            if "PID:" in line:
                pid_str = line.split("PID:")[-1].strip()
                break

        return {
            "pid": pid_str,
            "log_path": log_path,
            "remote_dir": remote_dir,
        }

    def tail_remote_log(self, log_path: str, n: int = 50) -> List[str]:
        """Return the last *n* lines of a remote log file."""
        try:
            result = self._ssh(f"tail -n {n} {log_path}", timeout=15)
            return result.stdout.splitlines()
        except Exception:
            return []

    def is_matlab_alive(self, remote_pid: str) -> bool:
        """Check whether a MATLAB process on the server is still running."""
        try:
            result = self._ssh(
                f"ps -p {remote_pid} > /dev/null && echo ALIVE || echo DEAD",
                timeout=10,
            )
            return "ALIVE" in result.stdout
        except Exception:
            return False

    def download_file(self, remote_path: str, local_dir: str) -> str:
        """Download a file from the server via SCP.

        Returns the local path of the downloaded file.
        """
        import subprocess
        local_dir_path = Path(local_dir)
        local_dir_path.mkdir(parents=True, exist_ok=True)
        local_path = local_dir_path / Path(remote_path).name

        cmd = [
            "scp", "-P", str(self.port),
            "-o", "ConnectTimeout=30",
            "-o", "StrictHostKeyChecking=accept-new",
            f"{self.ssh_target}:{remote_path}",
            str(local_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120)
        return str(local_path)

    def list_remote_mat_files(self, subdir: str = "") -> List[str]:
        """List .mat files in the workspace uploads directory (or *subdir*)."""
        target = f"{self._remote_upload_dir}/{subdir}".rstrip("/")
        try:
            result = self._ssh(
                f"find {target} -name '*.mat' -type f 2>/dev/null", timeout=15,
            )
            return [p for p in result.stdout.splitlines() if p.strip()]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # run_test — not implemented for remote (testing is local-only)
    # ------------------------------------------------------------------

    def run_test(
        self, config: Dict[str, Any], checkpoint_path: str, mat_file: str
    ) -> Dict[str, Any]:
        """Not implemented — testing is done locally.

        Use :meth:`download_checkpoint` to fetch a checkpoint first, then run
        ``Mainfor6WLAdapter.run_test()`` locally.
        """
        raise NotImplementedError(
            "Remote testing is not supported. "
            "Use download_checkpoint() to pull the .pth locally, then "
            "run Mainfor6WLAdapter.run_test()."
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_best_gpu(self) -> int:
        """Return the index of the GPU with the most free memory."""
        gpus = self.get_gpu_status()
        if not gpus:
            raise RuntimeError(
                "Cannot query GPU status from server. "
                "Check that the SSH connection is working."
            )
        best = max(gpus, key=lambda g: g.get("memory_free_mib", 0))
        return best["index"]

    @staticmethod
    def _parse_job_id(job_id: str) -> tuple:
        """Parse a job_id string into (user_host, run_id, remote_pid)."""
        parts = job_id.rsplit(":", 2)
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        return "", "", parts[0]
