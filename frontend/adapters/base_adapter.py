from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseODNNAdapter(ABC):
    """Abstract base class for ODNN backend adapters.

    Each concrete adapter wraps one backend script (e.g. mainfor6.py).
    The interface is intentionally narrow: pages only call these methods
    and never touch subprocess/os directly.
    """

    @abstractmethod
    def load_default_config(self) -> Dict[str, Any]:
        """Return a dict of default hyperparameters for this backend script."""
        pass

    @abstractmethod
    def start_training(self, config: Dict[str, Any], mat_file: str = "") -> int:
        """Launch training with the given config.

        Args:
            config:   hyperparameter dict (will be serialised to JSON)
            mat_file: path to the .mat mode file (input data)

        Returns:
            PID of the spawned subprocess.
        """
        pass

    @abstractmethod
    def stop_training(self, pid: int) -> None:
        """Send SIGTERM to the training subprocess."""
        pass

    @abstractmethod
    def is_training_alive(self, pid: int) -> bool:
        """Return True if the training subprocess is still running."""
        pass

    @abstractmethod
    def run_test(
        self,
        config: Dict[str, Any],
        checkpoint_path: str,
        mat_file: str,
    ) -> Dict[str, Any]:
        """Load a checkpoint, run evaluation, and collect propagation frames.

        Returns a dict containing at least:
            'frames':             list of frame dicts for propagation timeline
            'metrics':            dict of scalar evaluation metrics
            'model_meta':         dict from the checkpoint meta field
            'evaluation_regions': list of (x0,x1,y0,y1) tuples
            'detect_radius':      int
        """
        pass

    @abstractmethod
    def list_checkpoints(self) -> List[str]:
        """Return absolute paths of all .pth files in the checkpoints directory."""
        pass

    @abstractmethod
    def load_checkpoint_meta(self, pth_path: str) -> Dict[str, Any]:
        """Read only the 'meta' dict from a checkpoint file (no weights loaded).

        Returns an empty dict if the file cannot be read or has no 'meta' key.
        """
        pass

    def read_log_tail(self, n: int = 50) -> List[str]:
        """Return the last *n* lines of the training log.

        Default implementation returns an empty list; subclasses that write
        a log file should override this to surface errors in the UI.
        """
        return []
