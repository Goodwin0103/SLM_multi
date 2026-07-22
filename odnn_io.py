from __future__ import annotations
from pathlib import Path
from typing import Tuple, Optional, Any
import mat73
import numpy as np
from scipy.io import loadmat


def load_complex_modes_from_mat(
    mat_path: str | Path,
    key: str | None = None,
    key_candidates: Tuple[str, ...] = (
        "eigenmodes_OM4_176",
        "modes_field",
        "modes",
        "E",
    ),
) -> Tuple[np.ndarray, Optional[Any]]:
    """Load complex mode field and optional mode_info from a .mat file.

    Returns:
        (modes_array, mode_info)
        modes_array: complex64 array of shape (H, W, M)
        mode_info: struct array from MATLAB, or None if not present in file
    """

    def _to_complex(arr):
        if isinstance(arr, np.ndarray) and np.iscomplexobj(arr):
            return arr.astype(np.complex64, copy=False)
        if isinstance(arr, dict):
            for re_key, im_key in (("real","imag"), ("realPart","imagPart"), ("Re","Im")):
                if re_key in arr and im_key in arr:
                    return (np.array(arr[re_key]) + 1j*np.array(arr[im_key])).astype(np.complex64, copy=False)
        if isinstance(arr, np.ndarray):
            return arr.astype(np.complex64, copy=False)
        if hasattr(arr, "dtype") and getattr(arr, "dtype", None) and np.iscomplexobj(arr):
            return arr.astype(np.complex64, copy=False)
        raise ValueError("Unsupported data format: expected complex array or dict with real/imag parts.")

    mat_path = Path(mat_path)
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")

    mode_info = None

    try:
        data = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
        keys = [key] if key else [k for k in key_candidates if k in data]
        if not keys:
            payload = [k for k in data.keys() if not k.startswith("__")]
            if not payload:
                raise KeyError("Unable to locate data key in MAT file.")
            keys = [payload[0]]
        arr = data[keys[0]]
        complex_modes = _to_complex(arr)
        if "mode_info" in data:
            mode_info = data["mode_info"]
    except Exception:
        data = mat73.loadmat(str(mat_path))
        keys = [key] if key else [k for k in key_candidates if k in data] or [next(iter(data.keys()))]
        arr = data[keys[0]]
        complex_modes = _to_complex(arr)
        if "mode_info" in data:
            mode_info = data["mode_info"]

    complex_modes = np.asarray(complex_modes)
    if complex_modes.ndim == 2:
        complex_modes = complex_modes[..., None]
    elif complex_modes.ndim == 3 and complex_modes.shape[0] == complex_modes.shape[1] != complex_modes.shape[2]:
        pass
    elif complex_modes.ndim == 3 and complex_modes.shape[1] == complex_modes.shape[2]:
        complex_modes = np.transpose(complex_modes, (1, 2, 0))
    elif complex_modes.ndim != 3:
        raise ValueError(f"Expected 2D/3D array, received ndim={complex_modes.ndim}.")

    return complex_modes.astype(np.complex64, copy=False), mode_info
