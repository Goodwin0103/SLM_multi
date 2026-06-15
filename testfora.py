import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import mat73
import torch
from matplotlib.colors import Normalize
from odnn_model import (
    D2NNModel,
    complex_pad_asymm,
    complex_pad,
    complex_crop,
    propagation,
)
from ODNN_functions import generate_fields_ts

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

INPUT_MODE = "synthetic"  # "synthetic" 或 "crop"

SYNTHETIC_CONFIG = {
    "num_samples": 5,
    "num_modes": 6,
    "field_size": 25,
    "layer_size": 100,
    "focus_radius": 5,
    "eigenmode_mat": "mmf_6modes_25_PD_1.15.mat",
    "eigenmode_key": "modes_field",
    "amplitude_weights": np.array(
        [0.3062, 0.4336, 0.7359, 0.2755, 0.1975, 0.2486], dtype=np.float32
    ),
    "random_amplitude": False,
    "amplitude_seed": 20251102,
    "amplitude_low": 0.3,
    "amplitude_high": 1.3,
    "phase_option": 4,  # option 4 -> random phase for every mode
    "phase_seed": 20251024,
    "phase_sets": None,  # 若提供 shape=(num_samples, num_modes) 的数组，则直接使用
    "amplitude_sets": None,  # 若提供 shape=(num_samples, num_modes) 的数组，则直接使用
}


PROPAGATION_SCAN_CONFIG = {
    "enabled": False,          # 是否生成传播切片
    "sample_filter": None,    # 可选: 仅对指定样本运行传播扫描
    "z_start": 0.0,
    "z_step": 5e-6,
    "kmax": 25,
    "ncols": 5,
    "cmap": "RdBu_r",
    "save_npz": False,         # 是否保存npz
}

_PROP_SCAN_FILTER = None if PROPAGATION_SCAN_CONFIG["sample_filter"] is None else set(
    PROPAGATION_SCAN_CONFIG["sample_filter"]
)


def _ensure_complex(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(t):
        return t.to(torch.complex64)
    return t.to(torch.complex64)


def _normalize_path(path_like):
    if path_like is None:
        return None
    return str(path_like)


def _scan_filter_accepts(sample_tag: str, numeric_key=None) -> bool:
    filt = _PROP_SCAN_FILTER
    if filt is None:
        return True
    if numeric_key is not None:
        if numeric_key in filt or str(numeric_key) in filt:
            return True
    if sample_tag in filt:
        return True
    return False


def load_complex_modes_from_mat(
    mat_path,
    key=None,
    key_candidates=("eigenmodes_OM4_176", "modes_field", "modes", "E"),
):
    """从 MAT 文件读取复数模场，返回 shape=(H, W, M)。"""

    def _to_complex(arr):
        if isinstance(arr, np.ndarray) and np.iscomplexobj(arr):
            return arr.astype(np.complex64, copy=False)
        if isinstance(arr, dict):
            for re_key, im_key in (("real", "imag"), ("realPart", "imagPart"), ("Re", "Im")):
                if re_key in arr and im_key in arr:
                    return (
                        np.asarray(arr[re_key]) + 1j * np.asarray(arr[im_key])
                    ).astype(np.complex64, copy=False)
        if hasattr(arr, "dtype") and np.iscomplexobj(arr):
            return np.asarray(arr, dtype=np.complex64)
        raise ValueError("未识别的数据格式：既不是复数数组，也没有(real/imag)字段。")

    mat_path = Path(mat_path).expanduser()
    if not mat_path.exists():
        raise FileNotFoundError(f"未找到模场文件: {mat_path}")

    try:
        data = sio.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        keys = [key] if key else [k for k in key_candidates if k in data]
        if not keys:
            payload_keys = [k for k in data.keys() if not k.startswith("__")]
            if not payload_keys:
                raise KeyError("文件里没找到有效数据键")
            keys = [payload_keys[0]]
        arr = data[keys[0]]
        complex_cube = _to_complex(arr)
    except Exception:
        data = mat73.loadmat(mat_path)
        keys = [key] if key else [k for k in key_candidates if k in data]
        if not keys:
            keys = [next(iter(data.keys()))]
        complex_cube = _to_complex(data[keys[0]])

    complex_cube = np.asarray(complex_cube)
    if complex_cube.ndim == 2:
        complex_cube = complex_cube[..., None]
    elif complex_cube.ndim == 3:
        if complex_cube.shape[0] != complex_cube.shape[1] and complex_cube.shape[1] == complex_cube.shape[2]:
            complex_cube = np.transpose(complex_cube, (1, 2, 0))
    else:
        raise ValueError(f"期望 2D/3D 数组，实际 ndim={complex_cube.ndim}")

    return complex_cube.astype(np.complex64, copy=False)


def sample_phase_weights(num_samples: int, num_modes: int, option: int, rng: np.random.Generator) -> np.ndarray:
    """根据选项生成相位权重（单位: 弧度），shape=(num_samples, num_modes)。"""
    phases = np.zeros((num_samples, num_modes), dtype=np.float32)
    if option == 1:
        return phases
    if option == 2:
        if num_modes > 1:
            phases[:, 1:] = rng.uniform(0.0, 2 * np.pi, size=(num_samples, num_modes - 1))
        return phases
    if option == 3:
        if num_modes > 1:
            phases[:, 1] = rng.uniform(0.0, np.pi, size=num_samples)
        if num_modes > 2:
            phases[:, 2:] = rng.uniform(0.0, 2 * np.pi, size=(num_samples, num_modes - 2))
        return phases
    if option == 4:
        phases[:, :] = rng.uniform(0.0, 2 * np.pi, size=(num_samples, num_modes))
        return phases
    if option == 5:
        if num_modes > 1:
            phases[:, 1] = rng.uniform(0.0, np.pi, size=num_samples)
        if num_modes > 2:
            phases[:, 2:] = rng.uniform(0.0, np.pi, size=(num_samples, num_modes - 2))
        return phases
    raise ValueError(f"暂不支持的 phase_option: {option}")


def prepare_synthetic_inputs(config: dict) -> dict:
    """生成幅度可变、相位随机的合成输入场。"""

    num_modes = int(config["num_modes"])
    field_size = int(config["field_size"])

    base_amp = np.asarray(config["amplitude_weights"], dtype=np.float32)
    if base_amp.size != num_modes:
        raise ValueError("amplitude_weights 的长度必须等于 num_modes")
    base_norm = np.linalg.norm(base_amp)
    if base_norm <= 0:
        raise ValueError("amplitude_weights 不能全为 0")
    base_amp = base_amp / base_norm

    amp_sets_cfg = config.get("amplitude_sets")
    if amp_sets_cfg is not None:
        amp_sets = np.asarray(amp_sets_cfg, dtype=np.float32)
        if amp_sets.ndim != 2 or amp_sets.shape[1] != num_modes:
            raise ValueError("amplitude_sets 需要是 shape=(num_samples, num_modes) 的二维数组")
        num_samples = amp_sets.shape[0]
    else:
        num_samples = int(config["num_samples"])
        if num_samples <= 0:
            raise ValueError("num_samples 必须为正整数")
        if config.get("random_amplitude", False):
            rng_amp = np.random.default_rng(config.get("amplitude_seed"))
            low = float(config.get("amplitude_low", 0.0))
            high = float(config.get("amplitude_high", 1.0))
            if high <= low:
                raise ValueError("amplitude_high 必须大于 amplitude_low")
            random_matrix = rng_amp.uniform(low, high, size=(num_samples, num_modes)).astype(np.float32)
            amp_sets = base_amp[None, :] * random_matrix
        else:
            amp_sets = np.repeat(base_amp[None, :], num_samples, axis=0)

    amp_sets = np.asarray(amp_sets, dtype=np.float32)
    norms = np.linalg.norm(amp_sets, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    amp_sets = amp_sets / norms

    phase_sets_cfg = config.get("phase_sets")
    if phase_sets_cfg is not None:
        phases = np.asarray(phase_sets_cfg, dtype=np.float32)
        if phases.ndim != 2 or phases.shape[0] != num_samples or phases.shape[1] != num_modes:
            raise ValueError("phase_sets 需要是 shape=(num_samples, num_modes) 的二维数组")
    else:
        rng_phase = np.random.default_rng(config.get("phase_seed"))
        phases = sample_phase_weights(num_samples, num_modes, int(config["phase_option"]), rng_phase)

    complex_weights = amp_sets * np.exp(1j * phases)

    modes_hwM = load_complex_modes_from_mat(
        config["eigenmode_mat"], key=config.get("eigenmode_key")
    )
    if modes_hwM.shape[2] < num_modes:
        raise ValueError(f"模场数量 {modes_hwM.shape[2]} 少于所需模式数 {num_modes}")
    modes_hwM = modes_hwM[:, :, :num_modes]
    modes_MHW = np.transpose(modes_hwM, (2, 0, 1))

    amp_abs = np.abs(modes_MHW)
    denom = np.ptp(amp_abs) + 1e-12
    amp_norm = (amp_abs - amp_abs.min()) / denom
    modes_norm = amp_norm * np.exp(1j * np.angle(modes_MHW))

    MMF_data_ts = torch.from_numpy(modes_norm.astype(np.complex64))
    complex_weights_ts = torch.from_numpy(complex_weights.astype(np.complex64))

    fields = generate_fields_ts(
        complex_weights_ts, MMF_data_ts, num_samples, num_modes, field_size
    )

    return {
        "fields": fields,  # shape=(N,1,field_size,field_size), complex64
        "amplitude_sets": amp_sets,
        "base_amplitude": base_amp,
        "phase_weights": phases,
        "complex_weights": complex_weights,
    }


def build_label_map_from_weights(weights: np.ndarray, spot_masks: np.ndarray) -> np.ndarray:
    """Generate an idealized detector map by stamping weights inside each spot mask."""

    if spot_masks.ndim != 3:
        raise ValueError("spot_masks 必须是 shape=(num_modes, H, W) 的三维数组")

    weights = np.asarray(weights, dtype=np.float32)
    num_modes = spot_masks.shape[0]
    if weights.shape[0] != num_modes:
        raise ValueError("weights 数量必须与 spot_masks 数量一致")

    label_map = np.zeros(spot_masks.shape[1:], dtype=np.float32)
    for idx_mode in range(num_modes):
        label_map[spot_masks[idx_mode]] = float(weights[idx_mode])
    return label_map


def plot_sample_summary(
    sample_tag: str,
    input_amplitude: np.ndarray,
    model_output: np.ndarray,
    label_map: np.ndarray,
    label_weights: np.ndarray,
    predicted_weights: np.ndarray,
    save_path,
) -> str:
    """Create a 2×2 summary panel for a single sample."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    vmax_input = float(np.percentile(input_amplitude, 99.0)) if input_amplitude.size else 0.0
    fallback_input = float(np.max(input_amplitude)) if input_amplitude.size else 1.0
    vmax_input = vmax_input if np.isfinite(vmax_input) and vmax_input > 0 else fallback_input
    vmax_input = max(vmax_input, 1e-8)

    vmax_output = float(np.percentile(model_output, 99.0)) if model_output.size else 0.0
    fallback_output = float(np.max(model_output)) if model_output.size else 1.0
    vmax_output = vmax_output if np.isfinite(vmax_output) and vmax_output > 0 else fallback_output
    vmax_output = max(vmax_output, 1e-8)

    vmax_label = float(np.percentile(label_map, 99.0)) if label_map.size else 0.0
    fallback_label = float(np.max(label_map)) if label_map.size else 1.0
    vmax_label = vmax_label if np.isfinite(vmax_label) and vmax_label > 0 else fallback_label
    vmax_label = max(vmax_label, 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    im0 = axes[0, 0].imshow(input_amplitude, cmap="turbo", vmin=0.0, vmax=vmax_input)
    axes[0, 0].set_title(f"Input amplitude ({sample_tag})")
    axes[0, 0].axis("off")

    im1 = axes[0, 1].imshow(model_output, cmap="turbo", vmin=0.0, vmax=vmax_output)
    axes[0, 1].set_title("Model output intensity")
    axes[0, 1].axis("off")

    im2 = axes[1, 0].imshow(label_map, cmap="turbo", vmin=0.0, vmax=vmax_label)
    axes[1, 0].set_title("Ideal label map")
    axes[1, 0].axis("off")

    indices = np.arange(1, label_weights.size + 1)
    bar_width = 0.35
    axes[1, 1].bar(
        indices - bar_width / 2,
        label_weights,
        width=bar_width,
        label="Label weight",
        color="tab:green",
    )
    axes[1, 1].bar(
        indices + bar_width / 2,
        predicted_weights,
        width=bar_width,
        label="Predicted weight",
        color="tab:blue",
    )
    axes[1, 1].set_xticks(indices)
    axes[1, 1].set_xlabel("Mode index")
    axes[1, 1].set_ylabel("Weight")
    axes[1, 1].set_ylim(0.0, 1.1 * max(1e-6, float(np.max([label_weights.max(), predicted_weights.max(), 1e-6]))))
    axes[1, 1].grid(axis="y", linestyle="--", alpha=0.4)
    axes[1, 1].legend(loc="upper right")
    axes[1, 1].set_title("Weight comparison")

    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)
    fig.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    return str(save_path)


def export_sample_summary_mat(
    sample_tag: str,
    input_field_complex: np.ndarray,
    input_amplitude: np.ndarray,
    input_intensity: np.ndarray,
    model_output: np.ndarray,
    label_map: np.ndarray,
    label_weights: np.ndarray,
    predicted_weights: np.ndarray,
    spot_indices: np.ndarray,
    save_path,
) -> str:
    """Save the arrays used in the summary plot into a MATLAB .mat file."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "sample_tag": np.array(sample_tag, dtype=object),
        "input_field_complex": np.asarray(input_field_complex, dtype=np.complex64),
        "input_amplitude": np.asarray(input_amplitude, dtype=np.float32),
        "input_intensity": np.asarray(input_intensity, dtype=np.float32),
        "model_output_intensity": np.asarray(model_output, dtype=np.float32),
        "label_map": np.asarray(label_map, dtype=np.float32),
        "label_weights": np.asarray(label_weights, dtype=np.float32),
        "predicted_weights": np.asarray(predicted_weights, dtype=np.float32),
        "spot_indices": np.asarray(spot_indices, dtype=np.int32),
    }

    var_notes = {
        "sample_tag": "样本标识 (z.b. sample_00)",
        "input_field_complex": "填充后的复数输入光场, 形状 layer_size×layer_size",
        "input_amplitude": "|input_field_complex| 幅度图",
        "input_intensity": "|input_field_complex|^2 强度图",
        "model_output_intensity": "模型实际输出",
        "label_map": "根据 label 幅度的理想输出",
        "label_weights": "标签 (理想) 的幅度权重, L2 归一化",
        "predicted_weights": "由相机输出得到的幅度权重",
        "spot_indices": "与权重对应的 1-based 光斑编号",
        "var_notes": "两列 cell, 变量名与中文说明",
    }

    notes_array = np.empty((len(var_notes), 2), dtype=object)
    for row_idx, (key, desc) in enumerate(var_notes.items()):
        notes_array[row_idx, 0] = key
        notes_array[row_idx, 1] = desc

    payload["var_notes"] = notes_array

    sio.savemat(str(save_path), payload)
    return str(save_path)


def export_panel_mat(
    sample_tags,
    amp_by_tag,
    output_by_tag,
    label_map_by_tag,
    label_weights_by_tag,
    predicted_weights_by_tag,
    spot_indices,
    save_path,
    figure_label: str,
):
    """Save multi-sample panel data into a MAT file."""

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    ordered_tags = list(sample_tags)
    if not ordered_tags:
        return ""

    amp_stack = np.stack(
        [np.asarray(amp_by_tag[tag], dtype=np.float32) for tag in ordered_tags], axis=0
    )
    output_stack = np.stack(
        [np.asarray(output_by_tag[tag], dtype=np.float32) for tag in ordered_tags], axis=0
    )
    label_stack = np.stack(
        [np.asarray(label_map_by_tag[tag], dtype=np.float32) for tag in ordered_tags], axis=0
    )
    label_weight_mat = np.stack(
        [np.asarray(label_weights_by_tag[tag], dtype=np.float32) for tag in ordered_tags], axis=0
    )
    predicted_weight_mat = np.stack(
        [np.asarray(predicted_weights_by_tag[tag], dtype=np.float32) for tag in ordered_tags], axis=0
    )

    payload = {
        "figure_label": np.array(figure_label, dtype=object),
        "sample_tags": np.array(ordered_tags, dtype=object),
        "input_amplitude_stack": amp_stack,
        "output_stack": output_stack,
        "label_stack": label_stack,
        "label_weights": label_weight_mat,
        "predicted_weights": predicted_weight_mat,
        "spot_indices": np.asarray(spot_indices, dtype=np.int32),
    }

    var_notes = {
        "figure_label": "图像标签 (random_phase 或 phase0)",
        "sample_tags": "列顺序对应的样本名",
        "input_amplitude_stack": "每个样本的输入幅度 |E|, shape=(N,h,w)",
        "output_stack": "对应模式下的网络输出强度, shape=(N,H,W)",
        "label_stack": "根据 label 权重生成的理想检测图, shape=(N,H,W)",
        "label_weights": "理想幅度权重 (L2 归一), shape=(N,num_modes)",
        "predicted_weights": "由输出积分得到的幅度权重, shape=(N,num_modes)",
        "spot_indices": "与权重对应的 1-based 光斑编号",
        "var_notes": "两列cell, 变量名及注释",
    }

    notes_array = np.empty((len(var_notes), 2), dtype=object)
    for row_idx, (key, desc) in enumerate(var_notes.items()):
        notes_array[row_idx, 0] = key
        notes_array[row_idx, 1] = desc

    payload["var_notes"] = notes_array
    sio.savemat(str(save_path), payload)
    return str(save_path)

def plot_propagated_field_padded(
    E0: torch.Tensor,
    z_start: float,
    z_end: float,
    z_step: float,
    dx: float,
    lam: float,
    *,
    pad_px: int = 0,
    plot: bool = False,
    kmax: int = 12,
    ncols: int = 5,
    save_path=None,
    mode: str = "intensity",
    dpi: int = 300,
    cmap: str = "turbo",
    add_colorbar: bool = True,
):
    if z_step <= 0:
        raise ValueError("z_step must be > 0")
    if z_end < z_start:
        raise ValueError("z_end must be ≥ z_start")

    device_local = E0.device
    E0_c = _ensure_complex(E0)

    num_steps = int(np.floor((z_end - z_start) / z_step)) + 1
    z_values = np.linspace(
        z_start, z_start + (num_steps - 1) * z_step, num_steps, dtype=np.float64
    )

    H, W = E0_c.shape[-2], E0_c.shape[-1]
    frames = []

    if pad_px and pad_px > 0:
        Np = H + 2 * pad_px
        fx = torch.fft.fftshift(torch.fft.fftfreq(Np, d=dx)).to(device_local)
        fxx, fyy = torch.meshgrid(fx, fx, indexing="ij")
        argument = (2 * torch.pi) ** 2 * ((1.0 / lam) ** 2 - fxx**2 - fyy**2)
        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j * tmp).to(torch.complex64)

        E0_pad = complex_pad(E0_c, pad_px, pad_px)
        for z in z_values:
            spectrum = torch.fft.fftshift(torch.fft.fft2(E0_pad))
            propagated = torch.fft.ifft2(
                torch.fft.ifftshift(spectrum * torch.exp(1j * kz * float(z)))
            )
            frames.append(
                complex_crop(propagated, H, W, pad_px, pad_px).detach().cpu()
            )
    else:
        for z in z_values:
            propagated = propagation(E0_c, float(z), lam, W, dx, device_local)
            frames.append(propagated.detach().cpu())

    fields = torch.stack(frames, dim=0)

    save_path_norm = _normalize_path(save_path)
    if plot or save_path_norm:
        total = fields.shape[0]
        k_eff = min(total, int(kmax))
        select_idx = np.linspace(0, total - 1, k_eff, dtype=int)
        show = fields[select_idx].numpy()

        if mode == "intensity":
            show = np.abs(show) ** 2
        else:
            show = np.abs(show)

        p99 = np.percentile(show, 99.0)
        if p99 > 0:
            show = np.clip(show / p99, 0, 1)

        ncols_eff = max(1, int(ncols))
        nrows = (k_eff + ncols_eff - 1) // ncols_eff
        fig, axes = plt.subplots(
            nrows, ncols_eff, figsize=(2.2 * ncols_eff, 2.2 * nrows)
        )
        axes = np.array(axes).reshape(-1)

        last_im = None
        for jj, idx_val in enumerate(select_idx):
            ax = axes[jj]
            last_im = ax.imshow(show[jj], cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"z={z_values[idx_val] * 1e6:.0f} µm", fontsize=8)
            ax.axis("off")
        for jj in range(k_eff, len(axes)):
            axes[jj].axis("off")
        if add_colorbar and last_im is not None:
            fig.colorbar(last_im, ax=axes[:k_eff].tolist(), fraction=0.02, pad=0.02)
        fig.tight_layout()

        if save_path_norm:
            os.makedirs(os.path.dirname(save_path_norm), exist_ok=True)
            fig.savefig(save_path_norm, dpi=dpi)
            print("Saved figure ->", os.path.abspath(save_path_norm))
        plt.close(fig)

    return fields, z_values


@torch.no_grad()
def run_propagation_scans(
    E_field: torch.Tensor,
    *,
    sample_tag: str,
    save_dir,
    z_start: float,
    z_step: float,
    kmax: int,
    ncols: int,
    cmap: str,
    save_npz: bool = True,
):
    """生成逐层传播切片，并保存"""
    sample_tag = str(sample_tag)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    figure_paths = []
    data_payload = {"E_input": _ensure_complex(E_field).detach().cpu().numpy()}

    E_current = _ensure_complex(E_field)
    pre_prop = D2NN.pre_propagation
    zo = float(pre_prop.z)
    pad_pre = int(getattr(pre_prop, "pad_px", 0))
    z_end_input = z_start + zo

    scan_input_stack, scan_input_z = plot_propagated_field_padded(
        E_current,
        z_start,
        z_end_input,
        z_step,
        pixel_size,
        wavelength,
        pad_px=pad_pre,
        kmax=kmax,
        ncols=ncols,
        save_path=save_dir / "scan_input.png",
        cmap=cmap,
    )
    figure_paths.append(str(save_dir / "scan_input.png"))
    data_payload["scan_input_stack"] = scan_input_stack.numpy()
    data_payload["scan_input_z"] = scan_input_z

    layer0 = D2NN.layers[0]
    units = layer0.units
    if abs(zo) > 0:
        kz_pre = pre_prop.kz_pad if pad_pre > 0 else pre_prop.kz_base
        Ein = complex_pad(E_current, pad_pre, pad_pre) if pad_pre > 0 else E_current
        Eout = pre_prop._propagate(Ein, kz_pre, zo)
        E_current = (
            complex_crop(Eout, units, units, pad_pre, pad_pre)
            if pad_pre > 0
            else Eout
        )

    for idx_layer, layer in enumerate(D2NN.layers):
        phase = layer.phase.detach().to(E_current.device, dtype=torch.float32)
        E_current = E_current * torch.exp(1j * phase)

        pad_layer = int(layer.pad_px)
        z_end_layer = z_start + float(layer.z)
        scan_name = f"scan_layer{idx_layer + 1}"
        figure_path = save_dir / f"{scan_name}.png"
        scan_stack, scan_z = plot_propagated_field_padded(
            E_current,
            z_start,
            z_end_layer,
            z_step,
            pixel_size,
            wavelength,
            pad_px=pad_layer,
            kmax=kmax,
            ncols=ncols,
            save_path=figure_path,
            cmap=cmap,
        )
        figure_paths.append(str(figure_path))
        data_payload[f"{scan_name}_stack"] = scan_stack.numpy()
        data_payload[f"{scan_name}_z"] = scan_z

        kz_layer = layer.kz_pad if pad_layer > 0 else layer.kz_base
        Ein = complex_pad(E_current, pad_layer, pad_layer) if pad_layer > 0 else E_current
        Eout = layer._propagate(Ein, kz_layer, float(layer.z))
        E_current = (
            complex_crop(Eout, units, units, pad_layer, pad_layer)
            if pad_layer > 0
            else Eout
        )

        if idx_layer == len(D2NN.layers) - 1:
            prop_layer = D2NN.propagation
            pad_prop = int(prop_layer.pad_px)
            z_end_prop = z_start + float(prop_layer.z)
            figure_path2 = save_dir / "scan_to_camera.png"
            scan_stack2, scan_z2 = plot_propagated_field_padded(
                E_current,
                z_start,
                z_end_prop,
                z_step,
                pixel_size,
                wavelength,
                pad_px=pad_prop,
                kmax=kmax,
                ncols=ncols,
                save_path=figure_path2,
                cmap=cmap,
            )
            figure_paths.append(str(figure_path2))
            data_payload["scan_to_camera_stack"] = scan_stack2.numpy()
            data_payload["scan_to_camera_z"] = scan_z2

            kz_prop = prop_layer.kz_pad if pad_prop > 0 else prop_layer.kz_base
            Ein_cam = complex_pad(E_current, pad_prop, pad_prop) if pad_prop > 0 else E_current
            Eout_cam = prop_layer._propagate(Ein_cam, kz_prop, float(prop_layer.z))
            E_camera = (
                complex_crop(Eout_cam, units, units, pad_prop, pad_prop)
                if pad_prop > 0
                else Eout_cam
            )
            data_payload["E_camera"] = E_camera.detach().cpu().numpy()

    npz_path = None
    if save_npz and data_payload:
        npz_path = save_dir / f"propagation_scans_{sample_tag}.npz"
        np.savez_compressed(npz_path, **data_payload)

    print(f"✔ Propagation scans saved for {sample_tag} -> {save_dir}")
    return {
        "sample_tag": sample_tag,
        "figure_paths": figure_paths,
        "npz_path": str(npz_path) if npz_path else None,
    }

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent
output_dir = BASE_DIR / "verify_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

if INPUT_MODE != "synthetic":
    raise NotImplementedError("当前脚本仅实现 synthetic 输入模式。")


# === Konfiguration ===
layer_size = int(SYNTHETIC_CONFIG["layer_size"])
z_layers = 40e-6
z_prop = 120e-6
pixel_size = 1e-6
wavelength = 1568e-9
z_input_to_first = 40e-6
field_size = int(SYNTHETIC_CONFIG["field_size"])
num_modes = int(SYNTHETIC_CONFIG["num_modes"])
focus_radius = int(SYNTHETIC_CONFIG["focus_radius"])
energy_radius = 8  # 用于能量统计的检测区域半径（像素）
camera_outputs = []
camera_outputs_phase0 = []
input_amplitudes = []
input_phases = []
weights_per_sample = []
weights_per_sample_phase0 = []
scan_metadata = []
sample_tags = []
label_maps = []
summary_figure_paths = []
sample_mat_paths = []

# === model ===
D2NN = D2NNModel(
    num_layers=3,
    layer_size=layer_size,
    z_layers=z_layers,
    z_prop=z_prop,
    pixel_size=pixel_size,
    wavelength=wavelength,
    device=device,
    padding_ratio=0.5,
    z_input_to_first=z_input_to_first,
).to(device)

# === layer masks ===
mask_dir = BASE_DIR
if not mask_dir.exists():
    raise FileNotFoundError(f"未找到掩膜目录: {mask_dir}")
mask_files = sorted(f for f in os.listdir(mask_dir) if f.endswith(".xlsx"))
if not mask_files:
    raise FileNotFoundError(f"在 {mask_dir} 下未找到任何 .xlsx 掩膜文件")
print(f"✔ 读取到 {len(mask_files)} 个 mask, 每个大小: {pd.read_excel(mask_dir / mask_files[0], header=None).shape}")
masks = [
    pd.read_excel(mask_dir / f, header=None).to_numpy(dtype=np.float32)
    for f in mask_files
]
for layer, mask_np in zip(D2NN.layers, masks):
    with torch.no_grad():
        layer.phase.copy_(torch.tensor(mask_np, dtype=torch.float32, device=device))
print("✔ 覆盖相位掩膜成功")


# === Prepare spot masks for energy evaluation ===
num_rows = int(np.floor(np.sqrt(num_modes)))
num_cols = int(np.ceil(num_modes / num_rows))
row_spacing = (layer_size - num_rows * 2 * focus_radius) / (num_rows + 1)
col_spacing = (layer_size - num_cols * 2 * focus_radius) / (num_cols + 1)

Y, X = np.ogrid[:layer_size, :layer_size]
spot_masks = []
for r in range(1, num_rows + 1):
    for c in range(1, num_cols + 1):
        if len(spot_masks) >= num_modes:
            break
        center_row = int(
            round((r - 1) * (2 * focus_radius + row_spacing) + row_spacing + focus_radius)
        )
        center_col = int(
            round((c - 1) * (2 * focus_radius + col_spacing) + col_spacing + focus_radius)
        )
        mask = (X - center_col) ** 2 + (Y - center_row) ** 2 <= energy_radius ** 2
        spot_masks.append(mask)
spot_masks = np.stack(spot_masks, axis=0).astype(bool)
if spot_masks.shape[0] != num_modes:
    raise RuntimeError("生成的光斑掩膜数量与 num_modes 不一致，请检查参数设置。")
print(f"✔ 能量统计半径: {energy_radius} 像素")

# === Synthetic inputs ===
synthetic_inputs = prepare_synthetic_inputs(SYNTHETIC_CONFIG)
fields = synthetic_inputs["fields"].to(device)
amp_sets = synthetic_inputs["amplitude_sets"]
amp_base = synthetic_inputs["base_amplitude"]
phase_weights = synthetic_inputs["phase_weights"]
complex_weights = synthetic_inputs["complex_weights"]
num_samples = fields.shape[0]
sample_tags = [f"sample_{idx:02d}" for idx in range(num_samples)]
spot_indices = np.arange(1, num_modes + 1, dtype=np.int32)

amp_columns = [f"amp_mode_{k+1}" for k in range(num_modes)]
amp_df = pd.DataFrame(amp_sets.astype(np.float32), columns=amp_columns)
amp_df.insert(0, "sample_index", np.arange(num_samples, dtype=np.int32))
amp_df.insert(1, "sample_tag", sample_tags)
amp_df.to_csv(output_dir / "synthetic_amplitude_sets.csv", index=False)
np.save(output_dir / "synthetic_amplitude_sets.npy", amp_sets.astype(np.float32))

base_amp_df = pd.DataFrame(
    {
        "mode": np.arange(1, num_modes + 1, dtype=np.int32),
        "base_amplitude_weight": amp_base.astype(np.float32),
    }
)
base_amp_df.to_csv(output_dir / "synthetic_base_amplitude_weights.csv", index=False)
np.save(output_dir / "synthetic_base_amplitude_weights.npy", amp_base.astype(np.float32))

phase_columns = [f"phase_mode_{k+1}" for k in range(num_modes)]
phase_df = pd.DataFrame(phase_weights, columns=phase_columns)
phase_df.insert(0, "sample_index", np.arange(num_samples, dtype=np.int32))
phase_df.insert(1, "sample_tag", sample_tags)
phase_df.to_csv(output_dir / "synthetic_phase_weights.csv", index=False)
np.save(output_dir / "synthetic_phase_weights.npy", phase_weights.astype(np.float32))

complex_weights_df = pd.DataFrame(
    {
        "sample_index": np.repeat(np.arange(num_samples, dtype=np.int32), num_modes),
        "sample_tag": np.repeat(sample_tags, num_modes),
        "mode": np.tile(np.arange(1, num_modes + 1, dtype=np.int32), num_samples),
        "amplitude": np.abs(complex_weights).astype(np.float32).ravel(),
        "phase": np.angle(complex_weights).astype(np.float32).ravel(),
    }
)
complex_weights_df.to_csv(output_dir / "synthetic_complex_weights.csv", index=False)

# === Evaluate samples ===

dh = layer_size - field_size
dw = layer_size - field_size
pt, pb = dh // 2, dh - dh // 2
pl, pr = dw // 2, dw - dw // 2

for idx, sample_tag in enumerate(sample_tags):
    field_small = fields[idx, 0]
    field_cpu = field_small.detach().cpu()
    field_np = field_cpu.numpy()
    label_weights = amp_sets[idx]

    amp_map = np.abs(field_np)
    phase_map = np.angle(field_np)
    phase0_map = np.zeros_like(phase_map, dtype=np.float32)
    input_amplitudes.append((sample_tag, amp_map))
    input_phases.append((sample_tag, phase_map))

    np.save(output_dir / f"input_field_complex_{sample_tag}.npy", field_np.astype(np.complex64))
    np.save(output_dir / f"input_field_amplitude_{sample_tag}.npy", amp_map.astype(np.float32))
    np.save(output_dir / f"input_field_phase_{sample_tag}.npy", phase_map.astype(np.float32))
    phase0_field_np = np.abs(field_np).astype(np.complex64)
    np.save(output_dir / f"input_field_phase0_complex_{sample_tag}.npy", phase0_field_np)
    np.save(output_dir / f"input_field_phase0_phase_{sample_tag}.npy", phase0_map.astype(np.float32))

    amp_max = float(amp_map.max()) if amp_map.size else 0.0
    amp_plot = amp_map / amp_max if amp_max > 0 else amp_map

    fig_in, axes_in = plt.subplots(1, 3, figsize=(9, 3))
    axes_in[0].imshow(amp_plot, cmap="turbo", vmin=0.0, vmax=1.0)
    axes_in[0].set_title(f"Amplitude ({sample_tag})")
    axes_in[0].axis("off")
    axes_in[1].imshow(phase_map, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes_in[1].set_title(f"Phase ({sample_tag})")
    axes_in[1].axis("off")
    axes_in[2].imshow(phase0_map, cmap="twilight", vmin=-np.pi, vmax=np.pi)
    axes_in[2].set_title(f"Phase=0 ({sample_tag})")
    axes_in[2].axis("off")
    fig_in.tight_layout()
    fig_in.savefig(output_dir / f"input_field_{sample_tag}.png", dpi=300)
    plt.close(fig_in)

    E0 = complex_pad_asymm(field_small, pt, pb, pl, pr)
    E0_np = E0.detach().cpu().numpy()
    input_amplitude_full = np.abs(E0_np).astype(np.float32)
    input_intensity_full = np.square(input_amplitude_full)
    with torch.no_grad():
        output_tensor = D2NN(E0.unsqueeze(0).unsqueeze(0))
        output_intensity = output_tensor.squeeze().detach().cpu().numpy()

    np.save(output_dir / f"camera_output_{sample_tag}.npy", output_intensity.astype(np.float32))
    camera_outputs.append((sample_tag, output_intensity))

    fig_cam, ax_cam = plt.subplots(figsize=(4, 4))
    im_cam = ax_cam.imshow(output_intensity, cmap="turbo")
    ax_cam.set_title(f"Camera intensity ({sample_tag})")
    ax_cam.axis("off")
    fig_cam.colorbar(im_cam, ax=ax_cam, fraction=0.046, pad=0.04)
    fig_cam.tight_layout()
    fig_cam.savefig(output_dir / f"camera_output_{sample_tag}.png", dpi=300)
    plt.close(fig_cam)

    field_zero_phase = torch.abs(field_small).to(torch.complex64)
    E0_phase0 = complex_pad_asymm(field_zero_phase, pt, pb, pl, pr)
    with torch.no_grad():
        output_tensor_phase0 = D2NN(E0_phase0.unsqueeze(0).unsqueeze(0))
        output_intensity_phase0 = output_tensor_phase0.squeeze().detach().cpu().numpy()

    np.save(
        output_dir / f"camera_output_phase0_{sample_tag}.npy",
        output_intensity_phase0.astype(np.float32),
    )
    camera_outputs_phase0.append((sample_tag, output_intensity_phase0))

    fig_cam_phase0, ax_cam_phase0 = plt.subplots(figsize=(4, 4))
    im_cam_phase0 = ax_cam_phase0.imshow(output_intensity_phase0, cmap="turbo")
    ax_cam_phase0.set_title(f"Camera intensity (phase=0, {sample_tag})")
    ax_cam_phase0.axis("off")
    fig_cam_phase0.colorbar(im_cam_phase0, ax=ax_cam_phase0, fraction=0.046, pad=0.04)
    fig_cam_phase0.tight_layout()
    fig_cam_phase0.savefig(output_dir / f"camera_output_phase0_{sample_tag}.png", dpi=300)
    plt.close(fig_cam_phase0)

    if PROPAGATION_SCAN_CONFIG["enabled"] and _scan_filter_accepts(sample_tag, idx):
        scan_info = run_propagation_scans(
            E0,
            sample_tag=sample_tag,
            save_dir=output_dir / "propagation_scans" / sample_tag,
            z_start=PROPAGATION_SCAN_CONFIG["z_start"],
            z_step=PROPAGATION_SCAN_CONFIG["z_step"],
            kmax=PROPAGATION_SCAN_CONFIG["kmax"],
            ncols=PROPAGATION_SCAN_CONFIG["ncols"],
            cmap=PROPAGATION_SCAN_CONFIG["cmap"],
            save_npz=PROPAGATION_SCAN_CONFIG["save_npz"],
        )
        scan_metadata.append(scan_info)

    energies = np.array(
        [float(np.sum(output_intensity[mask])) for mask in spot_masks], dtype=np.float64
    )
    total_energy = float(np.sum(energies))
    if total_energy <= 0:
        weights_sq = np.zeros_like(energies)
        weights = np.zeros_like(energies)
    else:
        weights_sq = energies / total_energy
        weights = np.sqrt(np.clip(weights_sq, a_min=0.0, a_max=None))
    weights_per_sample.append((sample_tag, weights, weights_sq, energies, label_weights))
    print(f"✔ {sample_tag} weights: {np.round(weights, 4)}")

    label_map = build_label_map_from_weights(label_weights, spot_masks)
    label_maps.append((sample_tag, label_map))
    summary_fig_path = output_dir / f"sample_summary_{sample_tag}.png"
    summary_mat_path = output_dir / f"sample_summary_{sample_tag}.mat"
    summary_figure_paths.append(
        plot_sample_summary(
            sample_tag,
            input_amplitude_full,
            output_intensity,
            label_map,
            label_weights,
            weights,
            summary_fig_path,
        )
    )
    sample_mat_paths.append(
        export_sample_summary_mat(
            sample_tag,
            E0_np,
            input_amplitude_full,
            input_intensity_full,
            output_intensity,
            label_map,
            label_weights,
            weights,
            spot_indices,
            summary_mat_path,
        )
    )

    energies_phase0 = np.array(
        [float(np.sum(output_intensity_phase0[mask])) for mask in spot_masks],
        dtype=np.float64,
    )
    total_energy_phase0 = float(np.sum(energies_phase0))
    if total_energy_phase0 <= 0:
        weights_sq_phase0 = np.zeros_like(energies_phase0)
        weights_phase0 = np.zeros_like(energies_phase0)
    else:
        weights_sq_phase0 = energies_phase0 / total_energy_phase0
        weights_phase0 = np.sqrt(np.clip(weights_sq_phase0, a_min=0.0, a_max=None))
    weights_per_sample_phase0.append(
        (sample_tag, weights_phase0, weights_sq_phase0, energies_phase0, label_weights)
    )
    print(f"✔ {sample_tag} weights (phase=0): {np.round(weights_phase0, 4)}")

label_map_by_tag = dict(label_maps)

if weights_per_sample:
    records = []
    for tag, weights, weights_sq, energies, amp_target in weights_per_sample:
        row = {"sample_tag": tag}
        for idx_mode in range(num_modes):
            row[f"target_amp_{idx_mode + 1}"] = float(amp_target[idx_mode])
            row[f"weight_{idx_mode + 1}"] = float(weights[idx_mode])
            row[f"weight_sq_{idx_mode + 1}"] = float(weights_sq[idx_mode])
            row[f"energy_{idx_mode + 1}"] = float(energies[idx_mode])
        records.append(row)
    weights_df = pd.DataFrame(records)
    weights_df.to_csv(output_dir / "synthetic_energy_weights.csv", index=False)

if weights_per_sample_phase0:
    records_phase0 = []
    for tag, weights, weights_sq, energies, amp_target in weights_per_sample_phase0:
        row = {"sample_tag": tag}
        for idx_mode in range(num_modes):
            row[f"target_amp_{idx_mode + 1}"] = float(amp_target[idx_mode])
            row[f"weight_{idx_mode + 1}"] = float(weights[idx_mode])
            row[f"weight_sq_{idx_mode + 1}"] = float(weights_sq[idx_mode])
            row[f"energy_{idx_mode + 1}"] = float(energies[idx_mode])
        records_phase0.append(row)
    weights_df_phase0 = pd.DataFrame(records_phase0)
    weights_df_phase0.to_csv(output_dir / "synthetic_energy_weights_phase0.csv", index=False)

if camera_outputs:
    amp_by_tag = {tag: amp for tag, amp in input_amplitudes}
    phase_by_tag = {tag: phase for tag, phase in input_phases}
    output_by_tag = dict(camera_outputs)
    weights_by_tag = {tag: weights for tag, weights, _, _, _ in weights_per_sample}
    weights_sq_by_tag = {tag: weights_sq for tag, _, weights_sq, _, _ in weights_per_sample}
    label_weights_by_tag = {tag: amp_target for tag, _, _, _, amp_target in weights_per_sample}

    amp_vmax = max((float(np.max(amp)) for amp in amp_by_tag.values()), default=1.0)
    if amp_vmax <= 0:
        amp_vmax = 1.0
    vmax_out = max((float(np.max(img)) for img in output_by_tag.values()), default=1.0)
    if vmax_out <= 0:
        vmax_out = 1.0

    label_vmax = max((float(np.max(lbl)) for _, lbl in label_map_by_tag.items()), default=1.0)
    if label_vmax <= 0:
        label_vmax = 1.0

    panel_fig, panel_axes = plt.subplots(
        5,
        len(sample_tags),
        figsize=(4 * len(sample_tags), 15),
        squeeze=False,
    )

    for col, tag in enumerate(sample_tags):
        ax_amp = panel_axes[0, col]
        ax_phase = panel_axes[1, col]
        ax_out = panel_axes[2, col]
        ax_label = panel_axes[3, col]
        ax_bar = panel_axes[4, col]

        amp_map = amp_by_tag[tag]
        phase_map = phase_by_tag[tag]
        out_img = output_by_tag[tag]
        weights = weights_by_tag[tag]
        weights_sq = weights_sq_by_tag[tag]
        label_weights = label_weights_by_tag[tag]
        label_map = label_map_by_tag[tag]

        amp_plot = amp_map / amp_vmax if amp_vmax > 0 else amp_map
        ax_amp.imshow(amp_plot, cmap="turbo", vmin=0.0, vmax=1.0)
        ax_amp.set_title(f"Input amplitude ({tag})")
        ax_amp.axis("off")

        ax_phase.imshow(phase_map, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax_phase.set_title("Input phase")
        ax_phase.axis("off")

        ax_out.imshow(out_img, cmap="turbo", vmin=0.0, vmax=vmax_out)
        ax_out.set_title("Model output")
        ax_out.axis("off")

        ax_label.imshow(label_map, cmap="turbo", vmin=0.0, vmax=label_vmax)
        ax_label.set_title("Ideal label")
        ax_label.axis("off")

        bar_positions = np.arange(1, num_modes + 1)
        bar_width = 0.4
        ax_bar.bar(
            bar_positions - bar_width / 2,
            label_weights,
            width=bar_width,
            label="Actual amp weight",
            color="tab:green",
        )
        ax_bar.bar(
            bar_positions + bar_width / 2,
            weights,
            width=bar_width,
            label="Predicted weight",
            color="tab:blue",
        )
        ax_bar.set_ylim(0.0, 1.05)
        ax_bar.set_xticks(bar_positions)
        ax_bar.set_xlabel("Mode index")
        if col == 0:
            ax_bar.set_ylabel("Weight")
            ax_bar.legend(loc="upper right")
        ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
        ax_bar.set_title("Weight comparison")
        ax_bar.text(
            0.02,
            0.92,
            "label Σw²={:.3f}\npred Σw²={:.3f}".format(
                float(np.sum(label_weights**2)), float(weights_sq.sum())
            ),
            transform=ax_bar.transAxes,
            fontsize=8,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, linewidth=0),
        )

    panel_fig.tight_layout(h_pad=0.8)

    sm_amp = plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap="turbo")
    sm_amp.set_array([])
    panel_fig.colorbar(
        sm_amp, ax=panel_axes[0], fraction=0.046, pad=0.04, orientation="vertical"
    )

    sm_phase = plt.cm.ScalarMappable(norm=Normalize(-np.pi, np.pi), cmap="twilight")
    sm_phase.set_array([])
    panel_fig.colorbar(
        sm_phase, ax=panel_axes[1], fraction=0.046, pad=0.04, orientation="vertical"
    )

    sm_out = plt.cm.ScalarMappable(norm=Normalize(0.0, vmax_out), cmap="turbo")
    sm_out.set_array([])
    panel_fig.colorbar(
        sm_out, ax=panel_axes[2], fraction=0.046, pad=0.04, orientation="vertical"
    )

    sm_label = plt.cm.ScalarMappable(norm=Normalize(0.0, label_vmax), cmap="turbo")
    sm_label.set_array([])
    panel_fig.colorbar(
        sm_label, ax=panel_axes[3], fraction=0.046, pad=0.04, orientation="vertical"
    )

    panel_path = output_dir / "synthetic_input_output_bargrid.png"
    panel_fig.savefig(panel_path, dpi=300)
    plt.close(panel_fig)
    panel_mat_path = output_dir / "synthetic_input_output_bargrid.mat"
    export_panel_mat(
        sample_tags,
        amp_by_tag,
        output_by_tag,
        label_map_by_tag,
        label_weights_by_tag,
        weights_by_tag,
        spot_indices,
        panel_mat_path,
        figure_label="random_phase",
    )

# ##
# if camera_outputs_phase0:
#         output_by_tag_phase0 = dict(camera_outputs_phase0)
#         weights_by_tag_phase0 = {
#             tag: weights for tag, weights, _, _, _ in weights_per_sample_phase0
#         }
#         weights_sq_by_tag_phase0 = {
#             tag: weights_sq for tag, _, weights_sq, _, _ in weights_per_sample_phase0
#         }

#         vmax_out_phase0 = max(
#             (float(np.max(img)) for img in output_by_tag_phase0.values()), default=1.0
#         )
#         if vmax_out_phase0 <= 0:
#             vmax_out_phase0 = 1.0

#         panel_phase0_fig, panel_phase0_axes = plt.subplots(
#             4,
#             len(sample_tags),
#             figsize=(4 * len(sample_tags), 12),
#             squeeze=False,
#         )

#         for col, tag in enumerate(sample_tags):
#             ax_amp_p0 = panel_phase0_axes[0, col]
#             ax_out_p0 = panel_phase0_axes[1, col]
#             ax_label_p0 = panel_phase0_axes[2, col]
#             ax_bar_p0 = panel_phase0_axes[3, col]

#             amp_map = amp_by_tag[tag]
#             out_img_phase0 = output_by_tag_phase0[tag]
#             weights_phase0 = weights_by_tag_phase0[tag]
#             weights_sq_phase0 = weights_sq_by_tag_phase0[tag]
#             label_weights = label_weights_by_tag[tag]
#             label_map = label_map_by_tag[tag]

#             amp_plot = amp_map / amp_vmax if amp_vmax > 0 else amp_map
#             ax_amp_p0.imshow(amp_plot, cmap="turbo", vmin=0.0, vmax=1.0)
#             ax_amp_p0.set_title(f"Input amplitude ({tag})")
#             ax_amp_p0.axis("off")

#             ax_out_p0.imshow(out_img_phase0, cmap="turbo", vmin=0.0, vmax=vmax_out_phase0)
#             ax_out_p0.set_title(f"Camera output phase=0 ({tag})")
#             ax_out_p0.axis("off")

#             ax_label_p0.imshow(label_map, cmap="turbo", vmin=0.0, vmax=label_vmax)
#             ax_label_p0.set_title("Ideal label map")
#             ax_label_p0.axis("off")

#             bar_positions = np.arange(1, num_modes + 1)
#             bar_width = 0.4
#             ax_bar_p0.bar(
#                 bar_positions - bar_width / 2,
#                 label_weights,
#                 width=bar_width,
#                 label="Label weight",
#                 color="tab:green",
#             )
#             ax_bar_p0.bar(
#                 bar_positions + bar_width / 2,
#                 weights_phase0,
#                 width=bar_width,
#                 label="Predicted weight",
#                 color="tab:orange",
#             )
#             ax_bar_p0.set_ylim(0.0, 1.05)
#             ax_bar_p0.set_xticks(bar_positions)
#             ax_bar_p0.set_xlabel("Spot index")
#             if col == 0:
#                 ax_bar_p0.set_ylabel("Weight")
#                 ax_bar_p0.legend(loc="upper right")
#             ax_bar_p0.set_title("Energy vs. label weights (phase=0)")
#             ax_bar_p0.grid(axis="y", linestyle="--", alpha=0.4)
#             ax_bar_p0.text(
#                 0.02,
#                 0.92,
#                 "label Σw²={:.3f}\npred Σw²={:.3f}".format(
#                     float(np.sum(label_weights**2)), float(weights_sq_phase0.sum())
#                 ),
#                 transform=ax_bar_p0.transAxes,
#                 fontsize=8,
#                 ha="left",
#                 va="top",
#                 bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, linewidth=0),
#             )

#         panel_phase0_fig.tight_layout(h_pad=0.8)

#         sm_amp_phase0 = plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap="turbo")
#         sm_amp_phase0.set_array([])
#         panel_phase0_fig.colorbar(
#             sm_amp_phase0, ax=panel_phase0_axes[0], fraction=0.046, pad=0.04, orientation="vertical"
#         )

#         sm_out_phase0 = plt.cm.ScalarMappable(norm=Normalize(0.0, vmax_out_phase0), cmap="turbo")
#         sm_out_phase0.set_array([])
#         panel_phase0_fig.colorbar(
#             sm_out_phase0, ax=panel_phase0_axes[1], fraction=0.046, pad=0.04, orientation="vertical"
#         )

#         sm_label_phase0 = plt.cm.ScalarMappable(norm=Normalize(0.0, label_vmax), cmap="turbo")
#         sm_label_phase0.set_array([])
#         panel_phase0_fig.colorbar(
#             sm_label_phase0, ax=panel_phase0_axes[2], fraction=0.046, pad=0.04, orientation="vertical"
#         )

#         panel_phase0_path = output_dir / "synthetic_input_output_phase0_bargrid.png"
#         panel_phase0_fig.savefig(panel_phase0_path, dpi=300)
#         plt.close(panel_phase0_fig)
#         panel_phase0_mat = output_dir / "synthetic_input_output_phase0_bargrid.mat"
#         export_panel_mat(
#             sample_tags,
#             amp_by_tag,
#             output_by_tag_phase0,
#             label_map_by_tag,
#             label_weights_by_tag,
#             weights_by_tag_phase0,
#             spot_indices,
#             panel_phase0_mat,
#             figure_label="phase0",
#         )

#         fig_compare, axes_compare = plt.subplots(
#             2,
#             len(sample_tags),
#             figsize=(4 * len(sample_tags), 6),
#             squeeze=False,
#         )

#         for col, tag in enumerate(sample_tags):
#             ax_top = axes_compare[0, col]
#             ax_bottom = axes_compare[1, col]

#             ax_top.imshow(output_by_tag[tag], cmap="turbo", vmin=0.0, vmax=vmax_out)
#             ax_top.set_title(f"Random phase ({tag})")
#             ax_top.axis("off")

#             ax_bottom.imshow(
#                 output_by_tag_phase0[tag], cmap="turbo", vmin=0.0, vmax=vmax_out_phase0
#             )
#             ax_bottom.set_title(f"Phase=0 ({tag})")
#             ax_bottom.axis("off")

#         fig_compare.tight_layout(h_pad=0.6)

#         sm_top = plt.cm.ScalarMappable(norm=Normalize(0.0, vmax_out), cmap="turbo")
#         sm_top.set_array([])
#         fig_compare.colorbar(
#             sm_top, ax=axes_compare[0].tolist(), fraction=0.046, pad=0.04, orientation="vertical"
#         )

#         sm_bottom = plt.cm.ScalarMappable(
#             norm=Normalize(0.0, vmax_out_phase0), cmap="turbo"
#         )
#         sm_bottom.set_array([])
#         fig_compare.colorbar(
#             sm_bottom,
#             ax=axes_compare[1].tolist(),
#             fraction=0.046,
#             pad=0.04,
#             orientation="vertical",
#         )

#         compare_path = output_dir / "synthetic_camera_outputs_comparison.png"
#         fig_compare.savefig(compare_path, dpi=300)
#         plt.close(fig_compare)

#     fig_phase, axes_phase = plt.subplots(1, len(sample_tags), figsize=(4 * len(sample_tags), 3))
#     axes_phase = np.atleast_1d(axes_phase)
#     for col, tag in enumerate(sample_tags):
#         ax_phase = axes_phase[col]
#         ax_phase.imshow(phase_by_tag[tag], cmap="twilight", vmin=-np.pi, vmax=np.pi)
#         ax_phase.set_title(f"Phase ({tag})")
#         ax_phase.axis("off")
#     fig_phase.tight_layout()
#     cbar = fig_phase.colorbar(
#         plt.cm.ScalarMappable(norm=Normalize(-np.pi, np.pi), cmap="twilight"),
#         ax=axes_phase.tolist(),
#         fraction=0.046,
#         pad=0.04,
#         orientation="vertical",
#     )
#     cbar.set_label("Phase [rad]")
#     fig_phase.savefig(output_dir / "synthetic_input_phases.png", dpi=300)
#     plt.close(fig_phase)

# if scan_metadata:
#     print("✔ 已生成以下样本的传播切片数据:")
#     for info in scan_metadata:
#         npz_msg = info["npz_path"] if info["npz_path"] else "未保存 npz"
#         print(f"  - {info['sample_tag']} -> {npz_msg}")    

# print(f"✔ 所有仿真样本处理完成，结果已保存到: {output_dir}")

# ## 
