import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from scipy.io import loadmat
from odnn_model import D2NNModel, complex_pad, complex_crop


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 0) 基准参数（） =====
layer_size = 100
z_layers = 40e-6
z_prop = 120e-6
pixel_size = 1e-6
lambda_base = 1568e-9      # 基准波长
z_input_to_first = 40e-6

# ===== 1) 读取面膜（相位，单位应为弧度）=====
mask_dir = "results_MD"
mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".xlsx")])
base_masks = [pd.read_excel(os.path.join(mask_dir, f), header=None).to_numpy(dtype=np.float32)
              for f in mask_files]
print(f"✔ 读取到 {len(base_masks)} 个 mask, 每个大小: {base_masks[0].shape}")


# 检查尺寸匹配
for i, m in enumerate(base_masks):
    if m.shape != (layer_size, layer_size):
        raise ValueError(f"第{i}层mask尺寸 {m.shape} 与 layer_size={layer_size} 不匹配。请重采样或改 layer_size。")

# ===== 2) 按目标波长缩放相位 =====
def scale_masks_for_wavelength(masks_rad, lambda_base, lambda_target, do_mod=True):
    k = float(lambda_base / lambda_target)  # 缩放系数
    scaled = [m * k for m in masks_rad]
    if do_mod:
        scaled = [np.mod(m, 2*np.pi) for m in scaled]
    return scaled

# ===== 3) 把 numpy 相位图灌进模型层参数=====
@torch.no_grad()
def apply_phase_masks_to_model(model: D2NNModel, masks_rad):
    # 逐层写入
    for i, m in enumerate(masks_rad):
        # 确保 tensor 类型与设备
        t = torch.from_numpy(m).to(device=device, dtype=torch.float32)
        phase_param = model.layers[i].phase
        if phase_param.shape == t.shape:
            model.layers[i].phase.data.copy_(t)
        elif phase_param.ndim == 4 and phase_param.shape[-2:] == t.shape:
            model.layers[i].phase.data.copy_(t[None,None,...])
        else:
            raise ValueError(f"第{i}层 phase 形状 {phase_param.shape} 与 mask {t.shape} 不匹配")

# ===== 4) 运行：对每个目标波长，(1) 缩放面膜；(2) 设置模型波长；(3) 推理 =====
def run_for_wavelengths(target_wavelengths, input_field):
    """
    target_wavelengths: list/array of λ（单位米）
    input_field: torch.complex64 张量，形状 [B,1,H,W]，与 layer_size 对齐
    """
    outs = {}
    for lam in target_wavelengths:
        scaled_masks = scale_masks_for_wavelength(base_masks, lambda_base, lam, do_mod=True)
        model = D2NNModel(
            num_layers=len(base_masks),
            layer_size=layer_size,
            z_layers=z_layers,
            z_prop=z_prop,
            pixel_size=pixel_size,
            wavelength=lam,                 # 传播用当前 λ
            device=device,
            padding_ratio=0.5,
            z_input_to_first=z_input_to_first
        ).to(device)
        apply_phase_masks_to_model(model, scaled_masks)
        model.eval()
        with torch.no_grad():
            out = model(input_field)        # e.g. return [B,1,H,W] complex
        outs[lam] = out.detach().clone()
    return outs

# ===== 5)对多波长跑=====
B = 16
inp = (torch.randn(B,1,layer_size,layer_size) + 1j*torch.randn(B,1,layer_size,layer_size)).to(torch.complex64).to(device)

lambda_list = [1538e-9, 1310e-9, 1064e-9]  # 测试的波长
results = run_for_wavelengths(lambda_list, inp)

print("完成。输出键为各个波长（米），值为对应的输出场张量。")
