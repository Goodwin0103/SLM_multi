import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio
import torch
from matplotlib.colors import Normalize
from skimage.transform import resize
from odnn_model import (
    D2NNModel,
    complex_pad_asymm,
    complex_pad,
    complex_crop,
    propagation,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 按照中心点和大小要求切图
def crop_square(image: np.ndarray, center_x: float, center_y: float, size: int):
    img_h, img_w = image.shape
    half = size / 2.0
    x1 = int(np.floor(center_x - half))
    x2 = int(np.floor(center_x + half - 1))
    y1 = int(np.floor(center_y - half))
    y2 = int(np.floor(center_y + half - 1))

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"裁剪尺寸 {size} 与中心点 ({center_x}, {center_y}) 导致空切片。")
    return image[y1 : y2 + 1, x1 : x2 + 1], (x1, x2, y1, y2)



PROPAGATION_SCAN_CONFIG = {
    "enabled": False,          # 是否生成传播切片
    "crop_sizes": None,       
    "z_start": 0.0,
    "z_step": 5e-6,
    "kmax": 25,
    "ncols": 5,
    "cmap": "RdBu_r",
    "save_npz": False,         # 是否保存npz
}

_PROP_SCAN_CROP_SIZES = None if PROPAGATION_SCAN_CONFIG["crop_sizes"] is None else set(
    PROPAGATION_SCAN_CONFIG["crop_sizes"]
)

RANDOM_PHASE_CONFIG = {
    "enabled": True,      # 是否为输入叠加随机相位
    "seed": 20251101,     # 可设为 None 表示不同运行使用不同随机相位
    "low": 0.0,           # 相位下界（弧度）
    "high": 2 * np.pi,    # 相位上界（弧度）
}

_RANDOM_PHASE_RNG = None


def _ensure_complex(t: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(t):
        return t.to(torch.complex64)
    return t.to(torch.complex64)


def _normalize_path(path_like):
    if path_like is None:
        return None
    return str(path_like)


def _get_random_phase_rng():
    global _RANDOM_PHASE_RNG
    if _RANDOM_PHASE_RNG is None:
        seed = RANDOM_PHASE_CONFIG.get("seed")
        _RANDOM_PHASE_RNG = np.random.default_rng(seed)
    return _RANDOM_PHASE_RNG


def _sample_random_phase(shape, device):
    if not RANDOM_PHASE_CONFIG.get("enabled", False):
        return None
    rng = _get_random_phase_rng()
    low = float(RANDOM_PHASE_CONFIG.get("low", 0.0))
    high = float(RANDOM_PHASE_CONFIG.get("high", 2 * np.pi))
    phase_np = rng.uniform(low, high, size=shape).astype(np.float32)
    return torch.from_numpy(phase_np).to(device)


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
    crop_size: int,
    save_dir,
    z_start: float,
    z_step: float,
    kmax: int,
    ncols: int,
    cmap: str,
    save_npz: bool = True,
):
    """生成逐层传播切片，并保存"""
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
        npz_path = save_dir / f"propagation_scans_{crop_size}px.npz"
        np.savez_compressed(npz_path, **data_payload)

    print(
        f"✔ Propagation scans saved for {crop_size}px crop -> {save_dir}"
    )
    return {
        "crop_size": crop_size,
        "figure_paths": figure_paths,
        "npz_path": str(npz_path) if npz_path else None,
    }

# === Paths ===
BASE_DIR = Path(__file__).resolve().parent
MAT_NAME = "fiber_output_no_glass.mat"
MAT_PATH_OVERRIDE = '/media/mst32/xchange/_FC/Backup/Demo_ODNN_20251014/sentIPC_20251009/results20251021/fiber_output_8/fiber_output.mat' 

if MAT_PATH_OVERRIDE:
    mat_path = Path(MAT_PATH_OVERRIDE).expanduser()
else:
    mat_path = BASE_DIR / MAT_NAME

if not mat_path.exists():
    raise FileNotFoundError(
        f"未找到输入 MAT 文件: {mat_path}. 请将 {MAT_NAME} 放到 {BASE_DIR} 或修改 MAT_PATH_OVERRIDE。"
    )

output_dir = BASE_DIR / "verify_outputs"
output_dir.mkdir(parents=True, exist_ok=True)

# === 实验数据来===
exp_data = sio.loadmat(str(mat_path))
exp_image = exp_data["image"]
if exp_image.ndim != 2:
    raise ValueError(f"期望 MAT 中的 'image' 为二维阵列，实际形状为 {exp_image.shape}")

fig_orig, ax_orig = plt.subplots(figsize=(5, 5))
im = ax_orig.imshow(exp_image, cmap="turbo")
ax_orig.set_title("Original image")
ax_orig.axis("off")
fig_orig.colorbar(im, ax=ax_orig, fraction=0.046, pad=0.04)
fig_orig.tight_layout()
fig_orig.savefig(output_dir / "original_image.png", dpi=300)
plt.show()
plt.close(fig_orig)


# === Konfiguration ===
layer_size = 100
z_layers = 40e-6
z_prop = 120e-6
pixel_size = 1e-6
wavelength = 1568e-9
z_input_to_first = 40e-6
field_size = 25 
num_modes = 6
focus_radius = 5
energy_radius = 8  # 能量统计时使用更大的半径,自己设置就行了哈
center_xy = (151, 352)  # (x, y)
crop_sizes = range(60, 81, 5)  # 60, 65, 70, 75, 80
camera_outputs = []
resized_inputs = []
weights_per_crop = []
scan_metadata = []
random_phase_maps = []

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

#切实验图
for crop_size in crop_sizes:
    cropped_raw, bounds = crop_square(
        exp_image, center_xy[0], center_xy[1], crop_size
    )

    amp = np.sqrt(np.clip(cropped_raw, a_min=0.0, a_max=None))
    amp_min, amp_max = amp.min(), amp.max()
    if np.isclose(amp_max, amp_min):
        amp_norm = np.zeros_like(amp)
    else:
        amp_norm = (amp - amp_min) / (amp_max - amp_min)

    img_cropped_resized = resize(
        amp_norm, (field_size, field_size), anti_aliasing=True
    ).astype(np.float32)

    np.save(output_dir / f"img_cropped_{crop_size}.npy", amp_norm.astype(np.float32))
    np.save(
        output_dir / f"img_cropped_resized_{crop_size}.npy",
        img_cropped_resized,
    )
    # 缩放成25
    resized_inputs.append((crop_size, img_cropped_resized))
    fig_crop, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(amp_norm, cmap="turbo")
    axes[0].set_title(f"Cropped {crop_size}px")
    axes[0].axis("off")
    axes[1].imshow(img_cropped_resized, cmap="turbo")
    axes[1].set_title(f"Resized to {field_size}x{field_size}")
    axes[1].axis("off")
    fig_crop.tight_layout()
    fig_crop.savefig(output_dir / f"crop_preview_{crop_size}.png", dpi=300)
    plt.show()
    plt.close(fig_crop)
    # 生成随机相位并构造复振幅
    amplitude_tensor = torch.tensor(
        img_cropped_resized, dtype=torch.float32, device=device
    )
    phase_tensor = _sample_random_phase(amplitude_tensor.shape, device)
    if phase_tensor is not None:
        random_phase_np = phase_tensor.detach().cpu().numpy()
        random_phase_maps.append((crop_size, random_phase_np))
        np.save(
            output_dir / f"input_random_phase_{crop_size}.npy",
            random_phase_np.astype(np.float32),
        )
        E0_small = amplitude_tensor.to(torch.complex64) * torch.exp(1j * phase_tensor)
    else:
        E0_small = amplitude_tensor.to(torch.complex64)

    dh = layer_size - field_size
    dw = layer_size - field_size
    pt, pb = dh // 2, dh - dh // 2
    pl, pr = dw // 2, dw - dw // 2
    E0 = complex_pad_asymm(E0_small, pt, pb, pl, pr)
    # 跑model
    with torch.no_grad():
        output_intensity = (
            D2NN(E0.unsqueeze(0).unsqueeze(0)).squeeze().cpu().numpy()
        )
    np.save(
        output_dir / f"camera_output_{crop_size}.npy",
        output_intensity.astype(np.float32),
    )
    camera_outputs.append((crop_size, output_intensity))
    
    # 画传播过程切片图
    if PROPAGATION_SCAN_CONFIG["enabled"]:
        crop_filter = _PROP_SCAN_CROP_SIZES
        if crop_filter is None or crop_size in crop_filter:
            scan_info = run_propagation_scans(
                E0,
                crop_size=crop_size,
                save_dir=output_dir / "propagation_scans" / f"{crop_size}px",
                z_start=PROPAGATION_SCAN_CONFIG["z_start"],
                z_step=PROPAGATION_SCAN_CONFIG["z_step"],
                kmax=PROPAGATION_SCAN_CONFIG["kmax"],
                ncols=PROPAGATION_SCAN_CONFIG["ncols"],
                cmap=PROPAGATION_SCAN_CONFIG["cmap"],
                save_npz=PROPAGATION_SCAN_CONFIG["save_npz"],
            )
            scan_metadata.append(scan_info)
    # 看最后output，算energie
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
    weights_per_crop.append((crop_size, weights, weights_sq))
    print(f"✔ {crop_size}px crop weights: {np.round(weights, 4)}")

    fig_cam, ax_cam = plt.subplots(figsize=(4, 4))
    im_cam = ax_cam.imshow(output_intensity, cmap="turbo")
    ax_cam.set_title(f"Camera intensity ({crop_size}px crop)")
    ax_cam.axis("off")
    fig_cam.colorbar(im_cam, ax=ax_cam, fraction=0.046, pad=0.04)
    fig_cam.tight_layout()
    fig_cam.savefig(
        output_dir / f"camera_output_{crop_size}.png",
        dpi=300,
    )
    plt.show()
    plt.close(fig_cam)

# 保存权重数据
if weights_per_crop:
    records = []
    for size, weights, weights_sq in weights_per_crop:
        row = {"crop_size": size}
        for idx in range(num_modes):
            row[f"weight_{idx+1}"] = float(weights[idx])
            row[f"weight_sq_{idx+1}"] = float(weights_sq[idx])
        records.append(row)
    weights_df = pd.DataFrame(records)
    weights_df.to_csv(output_dir / "weights_summary.csv", index=False)
# 图画一起
if camera_outputs:
    sizes = [size for size, _ in camera_outputs]
    input_by_size = {size: img for size, img in resized_inputs}
    output_by_size = dict(camera_outputs)
    weights_by_size = {size: weights for size, weights, _ in weights_per_crop}
    weights_sq_by_size = {size: weights_sq for size, _, weights_sq in weights_per_crop}
    vmax_out = max(float(np.max(intensity)) for _, intensity in camera_outputs)

    fig_all, axes_all = plt.subplots(
        3, len(sizes), figsize=(4 * len(sizes), 9), squeeze=False
    )

    for col, size in enumerate(sizes):
        ax_in = axes_all[0, col]
        ax_out = axes_all[1, col]
        ax_bar = axes_all[2, col]

        in_img = input_by_size.get(size)
        out_img = output_by_size[size]
        weights = weights_by_size[size]
        weights_sq = weights_sq_by_size[size]

        im_in = ax_in.imshow(in_img, cmap="turbo", vmin=0.0, vmax=1.0)
        ax_in.set_title(f"Resized input ({size}px crop)")
        ax_in.axis("off")

        im_out = ax_out.imshow(out_img, cmap="turbo", vmin=0.0, vmax=vmax_out)
        ax_out.set_title(f"Camera output ({size}px crop)")
        ax_out.axis("off")

        ax_bar.bar(np.arange(1, num_modes + 1), weights, color="tab:blue")
        ax_bar.set_ylim(0.0, 1.05)
        ax_bar.set_xticks(np.arange(1, num_modes + 1))
        ax_bar.set_xlabel("Spot index")
        if col == 0:
            ax_bar.set_ylabel("Weight")
        ax_bar.set_title("Energy-derived weights")
        ax_bar.grid(axis="y", linestyle="--", alpha=0.4)
        ax_bar.text(
            0.02,
            0.92,
            f"∑w²={weights_sq.sum():.3f}",
            transform=ax_bar.transAxes,
            fontsize=8,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.5, linewidth=0),
        )

    fig_all.tight_layout(h_pad=0.6)

    sm_in = plt.cm.ScalarMappable(norm=Normalize(0.0, 1.0), cmap="turbo")
    sm_in.set_array([])
    fig_all.colorbar(
        sm_in, ax=axes_all[0], fraction=0.046, pad=0.04, orientation="vertical"
    )

    sm_out = plt.cm.ScalarMappable(norm=Normalize(0.0, vmax_out), cmap="turbo")
    sm_out.set_array([])
    fig_all.colorbar(
        sm_out, ax=axes_all[1], fraction=0.046, pad=0.04, orientation="vertical"
    )

    fig_all.savefig(output_dir / "camera_outputs_all.png", dpi=300)
    plt.show()
    plt.close(fig_all)

    if random_phase_maps:
        phase_by_size = {size: phase for size, phase in random_phase_maps}
        fig_phase, axes_phase = plt.subplots(
            1, len(sizes), figsize=(4 * len(sizes), 3), squeeze=False
        )
        axes_phase = axes_phase.reshape(-1)
        for col, size in enumerate(sizes):
            ax_phase = axes_phase[col]
            phase_img = phase_by_size.get(size)
            if phase_img is None:
                ax_phase.axis("off")
                ax_phase.set_title(f"{size}px (phase N/A)")
                continue
            im_phase = ax_phase.imshow(phase_img, cmap="twilight", vmin=0.0, vmax=2 * np.pi)
            ax_phase.set_title(f"Random phase ({size}px)")
            ax_phase.axis("off")
            fig_phase.colorbar(
                im_phase, ax=ax_phase, fraction=0.046, pad=0.04, orientation="vertical"
            )
        fig_phase.tight_layout()
        fig_phase.savefig(output_dir / "input_random_phases.png", dpi=300)
        plt.show()
        plt.close(fig_phase)

if scan_metadata:
    print("✔ 已生成以下裁剪的传播切片数据:")
    for info in scan_metadata:
        npz_msg = info["npz_path"] if info["npz_path"] else "未保存 npz"
        print(f"  - {info['crop_size']}px -> {npz_msg}")

print(f"✔ 所有裁剪结果与相机输出已保存到: {output_dir}")
