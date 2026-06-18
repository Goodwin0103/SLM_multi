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
        (e.g. ``"/home/jslai/odnn_project"``).
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
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              check=check)

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
        subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)

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
        subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=True)

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

        Returns a ``job_id`` string: ``"user@host:run_id:remote_pid"``.
        """
        # 1. Generate unique run_id ------------------------------------------
        run_id = time.strftime("run_%Y%m%d_%H%M%S")
        self._current_run_id = run_id
        run_dir = f"{self.workspace_dir}/runs/{run_id}"

        # 2. Create run directory on server ----------------------------------
        self._ssh(f"mkdir -p {run_dir}/{{logs,checkpoints}}")

        # 3. Handle .mat file ------------------------------------------------
        remote_mat_path: str
        if mat_file and Path(mat_file).exists():
            # local file -> upload once, then reuse
            mat_name = Path(mat_file).name
            self._ssh(f"mkdir -p {self._remote_upload_dir}")
            self._scp_upload(mat_file, f"{self._remote_upload_dir}/{mat_name}")
            remote_mat_path = f"{self._remote_upload_dir}/{mat_name}"
        else:
            # assume already on the server at the given path
            remote_mat_path = mat_file

        # 4. Upload config JSON ----------------------------------------------
        local_tmp = Path.home() / ".odnn" / f"train_config_{run_id}.json"
        local_tmp.parent.mkdir(parents=True, exist_ok=True)
        local_tmp.write_text(json.dumps(config, indent=2))
        self._scp_upload(str(local_tmp), f"{run_dir}/train_config.json")
        local_tmp.unlink(missing_ok=True)

        # 5. Auto-select GPU (most free memory) ------------------------------
        gpu_id = self._pick_best_gpu()

        # 6. Launch training (conda activation required!) --------------------
        launch_cmd = (
            f"source ~/miniconda3/etc/profile.d/conda.sh && "
            f"conda activate {self.conda_env} && "
            f"cd {self.project_dir} && "
            f"CUDA_VISIBLE_DEVICES={gpu_id} "
            f"nohup python mainfor6_wl.py "
            f"  --config {run_dir}/train_config.json "
            f"  --mat_file {remote_mat_path} "
            f"  --output_dir {run_dir} "
            f"  > {run_dir}/logs/training_wl.log 2>&1 & "
            f"echo $!"
        )
        result = self._ssh(launch_cmd, timeout=15)
        remote_pid = result.stdout.strip()

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
        """Fetch the current run's metrics JSONL from the server.

        Uses ``tail -n`` (not ``cat``) to avoid transferring the entire file
        on every poll cycle.
        """
        if not self._current_run_id:
            return ""
        run_dir = f"{self.workspace_dir}/runs/{self._current_run_id}"
        result = self._ssh_no_check(
            f"tail -n 200 {run_dir}/logs/metrics_wl.jsonl 2>/dev/null || true"
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
            f"source ~/miniconda3/etc/profile.d/conda.sh && "
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
            if len(parts) < 6:
                continue
            try:
                mem_used = int(parts[3].replace(" MiB", "").replace(" GiB", ""))
                mem_total = int(parts[4].replace(" MiB", "").replace(" GiB", ""))
                mem_free = int(parts[5].replace(" MiB", "").replace(" GiB", ""))
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "utilization_gpu": int(parts[2].replace(" %", "").replace("%", "")),
                    "memory_used_mib": mem_used,
                    "memory_total_mib": mem_total,
                    "temperature_gpu": int(parts[4].replace(" C", "").replace("C", "")) if len(parts) > 4 else 0,
                    "memory_free_mib": mem_free,
                })
            except (ValueError, IndexError):
                continue
        return gpus

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
            return 0  # fallback to GPU 0
        best = max(gpus, key=lambda g: g.get("memory_free_mib", 0))
        return best["index"]

    @staticmethod
    def _parse_job_id(job_id: str) -> tuple:
        """Parse a job_id string into (user_host, run_id, remote_pid)."""
        parts = job_id.rsplit(":", 2)
        if len(parts) == 3:
            return parts[0], parts[1], parts[2]
        return "", "", parts[0]
