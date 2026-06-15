# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 16:51:22 2024

@author: zhang
"""
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import savemat
from odnn_model import propagation
import numpy as np, os
from scipy.io import savemat
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import time
from datetime import datetime
import json
import os
import pandas as pd
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mat73
import h5py # save the data as MATLAB 7.3
# from light_propagation_simulation import center_crop_2d
from ODNN_functions import generate_complex_weights,generate_fields_ts,create_labels,create_evaluation_regions,create_labels_4_MMF3_phase,create_evaluation_regions_4_MMF3_phase
#from ODNN_MG_OM4_functions import create_evaluation_regions
from save_function import save_to_mat_MD_pro
from scipy.ndimage import gaussian_filter
import os, csv, math
from scipy.io import loadmat
from mpl_toolkits.mplot3d import Axes3D  
import random
SEED = 424242
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 让 cuDNN/算子走确定性分支
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

## save excel mask
def save_masks_one_file_per_layer(temp_model, out_dir, base_name="mask", save_degree=False, use_xlsx=True):
    os.makedirs(out_dir, exist_ok=True)
    for i, mask in enumerate(temp_model, start=1):
        arr = np.asarray(mask, dtype=np.float32)
        if save_degree:
            arr = np.degrees(arr)
        if use_xlsx:
          
            pd.DataFrame(arr).to_excel(
                os.path.join(out_dir, f"{base_name}_layer{i}.xlsx"),
                index=False, header=False, engine="openpyxl"
            )
        else:
            np.savetxt(os.path.join(out_dir, f"{base_name}_layer{i}.csv"), arr, delimiter=",")



def plot_propagated_field_padded(
    E0: torch.Tensor,
    z_start: float,
    z_end: float,
    z_step: float,
    dx: float,          # 像素尺寸 (m)，例如 pixel_size
    lam: float,         # 波长 (m)
    *,
    pad_px: int = 0,    
    plot: bool = False, # 需要拼图可视化时 True
    kmax: int = 12,     # 最多显示切片数
    ncols: int = 5,     # 拼图列数
    save_path: str | None = None,
    mode: str = "intensity",  # "intensity" 或 "amplitude"
    dpi: int = 300,
    cmap: str = "turbo",         
    add_colorbar: bool = True,   
):
   
    assert torch.is_complex(E0), "E0 must be complex."
    device = E0.device

    if z_step <= 0:
        raise ValueError("z_step must be > 0")
    num_steps = int(np.floor((z_end - z_start) / z_step)) + 1
    z_values = np.linspace(z_start, z_start + (num_steps-1)*z_step, num_steps)

    H, W = E0.shape[-2], E0.shape[-1]

    frames = []

    if pad_px and pad_px > 0:
        # ===== 带 padding 的传播 =====
        Np = H + 2*pad_px
        fx = torch.fft.fftshift(torch.fft.fftfreq(Np, d=dx)).to(device)
        fxx, fyy = torch.meshgrid(fx, fx, indexing='ij')
        arg = (2*torch.pi)**2 * ((1./lam)**2 - fxx**2 - fyy**2)
        kz  = torch.where(arg >= 0, torch.sqrt(torch.abs(arg)), 1j*torch.sqrt(torch.abs(arg))).to(torch.complex64)

        E0p = complex_pad(E0, pad_px, pad_px)  # 先 pad
        for z in z_values:
            C   = torch.fft.fftshift(torch.fft.fft2(E0p.to(torch.complex64)))
            Epz = torch.fft.ifft2(torch.fft.ifftshift(C * torch.exp(1j * kz * float(z))))
            Epz_c = complex_crop(Epz, H, W, pad_px, pad_px)  # 裁回
            frames.append(Epz_c.detach().cpu())
    else:
   
        Nx = W
        for z in z_values:
            Ez = propagation(E0, float(z), lam, Nx, dx, device) 
            frames.append(Ez.detach().cpu())

    fields = torch.stack(frames, dim=0)  # (S,H,W) complex64/complex128

    # ---- 可视化（抽样 + 保存）----
    if plot or save_path:
        S = fields.shape[0]
        K = min(S, int(kmax))
        idx = np.linspace(0, S-1, K, dtype=int)
        show = fields[idx].numpy()

        # 强度或振幅
        if mode == "intensity":
            show = np.abs(show)**2
        else:
            show = np.abs(show)

        p99 = np.percentile(show, 99.0)
        if p99 > 0:
            
            show = np.clip(show / p99, 0, 1)

        ncols_eff = max(1, int(ncols))
        nrows = (K + ncols_eff - 1) // ncols_eff
        fig, axes = plt.subplots(nrows, ncols_eff, figsize=(2.2*ncols_eff, 2.2*nrows))
        axes = np.array(axes).reshape(-1)
        
        last_im = None
        for j, s in enumerate(idx):
            ax = axes[j]
            ax.imshow(show[j], cmap=cmap, vmin=0, vmax=1)
            ax.set_title(f"z={z_values[s]*1e6:.0f} µm", fontsize=8)
            ax.axis("off")
        for j in range(K, len(axes)):
            axes[j].axis("off")
        if add_colorbar and last_im is not None:
            cbar = fig.colorbar(last_im, ax=axes[:K].tolist(), fraction=0.02, pad=0.02)
            cbar.set_label("Normalized " + ("Intensity" if mode=="intensity" else "Amplitude"))
     
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, dpi=dpi)
            print("Saved figure ->", os.path.abspath(save_path))
        plt.close(fig)

    return fields, z_values  # 新增返回 z 轴



#%算能量占比的相关函数
def _circle_masks_from_regions(shape_hw, evaluation_regions, r, offset=(0,0)):
    """把 evaluation_regions 的中心点变成圆形 ROI 掩膜（可整体偏移 offset）。"""
    H, W = shape_hw
    Y, X = np.ogrid[:H, :W]
    offx, offy = offset
    masks = []
    for (x0, x1, y0, y1) in evaluation_regions:
        cx = int(round((x0 + x1) / 2.0 + offx))
        cy = int(round((y0 + y1) / 2.0 + offy))
        m  = (X - cx)**2 + (Y - cy)**2 <= (r*r)
        masks.append(m)
    return masks

def _sum_signal_energy_circle(I, evaluation_regions, r, offset=(0,0)):
    """多个圆形 ROI 的能量并集"""
    masks = _circle_masks_from_regions(I.shape, evaluation_regions, r, offset)
    return float(sum(I[m].sum() for m in masks))

def _spot_energy_ratios_circle(I, evaluation_regions, r, offset=(0,0), eps=1e-12):
    masks = _circle_masks_from_regions(I.shape, evaluation_regions, r, offset)
    energies = np.array([float(I[m].sum()) for m in masks], dtype=np.float64)
    total = float(I.sum()) + eps
    ratios = energies / total
    return energies, ratios

def _masks_from_eval_regions(shape_hw, evaluation_regions, r_sig, center_offset=(0,0)):
    H, W = shape_hw
    Y, X = np.ogrid[:H, :W]
    offx, offy = center_offset  # 注意：offx 对应 x（列），offy 对应 y（行）

    masks, centers = [], []
    for (x0, x1, y0, y1) in evaluation_regions:
        cx = int(round((x0 + x1) / 2.0 + offx))
        cy = int(round((y0 + y1) / 2.0 + offy))
        m  = (X - cx)**2 + (Y - cy)**2 <= (r_sig**2)
        masks.append(m); centers.append((cx, cy))
    union_signal = np.any(np.stack(masks, axis=0), axis=0)
    return np.stack(masks, axis=0), union_signal, centers


#%之前用的算SNR的函数
def spot_energy_and_snr(I, evaluation_regions, r_sig,
                        ring_inner_pad=3, ring_thickness=8,
                        union_mode="global",
                        center_offset=(0,0)):
    """
    I: 强度图；可为裁剪小图或大画布图
    center_offset: (offx, offy)，把 ROI 中心整体平移到 I 的坐标系里。
                   例如大画布时应设为 (p, p)。
    """
    if np.iscomplexobj(I):
        I = np.abs(I)**2
    I = np.asarray(I, dtype=np.float64)
    eps = 1e-12

    H, W = I.shape
    Y, X = np.ogrid[:H, :W]

    # ROI 放到目标坐标系
    sig_masks, union_sig, centers = _masks_from_eval_regions((H, W), evaluation_regions, r_sig,
                                                             center_offset=center_offset)

    # --- 全局量（用 I 的全部像素作为“全图”）---
    total_energy  = float(I.sum())
    signal_energy = float(I[union_sig].sum())
    ratio_union   = signal_energy / (total_energy + eps)

    # --- 每斑点的环形 SNR（局部背景）---
    energies, snr_db_each = [], []
    for (cx, cy), sig in zip(centers, sig_masks):
        rsq = (X - cx)**2 + (Y - cy)**2
        r_in2  = (r_sig + ring_inner_pad)**2
        r_out2 = (r_sig + ring_inner_pad + ring_thickness)**2
        ring = (rsq >= r_in2) & (rsq <= r_out2) & (~union_sig)

        Ek = float(I[sig].sum())
        energies.append(Ek)

        if np.any(ring):
            bg_mean = float(I[ring].mean())
            bg_est  = bg_mean * int(sig.sum())
            snr_lin = Ek / (bg_est + eps)
            snr_db  = 10.0 * np.log10(max(snr_lin, eps))
        else:
            snr_db = float('nan')
        snr_db_each.append(snr_db)

    energies    = np.array(energies,   dtype=np.float64)
    snr_db_each = np.array(snr_db_each, dtype=np.float64)

    # --- 并集 SNR（全局或合并环形背景）---
    if union_mode == "global":
        bg_union = max(total_energy - signal_energy, 0.0)
    elif union_mode == "ring":
        ring_union_mask = np.zeros_like(union_sig, dtype=bool)
        for (cx, cy) in centers:
            rsq = (X - cx)**2 + (Y - cy)**2
            r_in2  = (r_sig + ring_inner_pad)**2
            r_out2 = (r_sig + ring_inner_pad + ring_thickness)**2
            ring_union_mask |= ((rsq >= r_in2) & (rsq <= r_out2))
        ring_union_mask &= ~union_sig
        if np.any(ring_union_mask):
            bg_mean_union = float(I[ring_union_mask].mean())
            bg_union = bg_mean_union * int(union_sig.sum())
        else:
            bg_union = float('nan')
    else:
        raise ValueError("union_mode must be 'global' or 'ring'.")

    snr_union_db = (10.0 * np.log10(max(signal_energy / (bg_union + eps), eps))
                    if not np.isnan(bg_union) else float('nan'))

    return {
        "energies":       energies,
        "snr_each_db":    snr_db_each,
        "ratio_union":    ratio_union,
        "snr_union_db":   snr_union_db,
    }

def build_circular_roi_masks(H, W, num_spots, focus_radius, radius_scale=1.2):
    
    # 半径 = focus_radius * radius_scale
    
    r = int(round(focus_radius * radius_scale))
    masks = []
    for k in range(num_spots):
        
        m = create_labels(H, W, num_spots, r, k+1)  # float32, 0/1
        masks.append((m > 0.5))
    stack = np.stack(masks, axis=0)               # (K,H,W) bool
    union = np.any(stack, axis=0)                 # (H,W)   bool
    return masks, union


#%归一化函数
def l2_normalize_rows(A, eps=1e-12):
    n = np.linalg.norm(A, axis=1, keepdims=True)
    return A / (n + eps)

def sample_tensor_slices(t: torch.Tensor, kmax: int = 25) -> torch.Tensor:
    """
    把 (S,H,W) 抽样到 <=kmax；(H,W) 原样返回。
    返回 CPU tensor，便于保存。
    """
    t = t.detach().cpu()
    if t.ndim == 2:
        return t
    S = t.shape[0]
    k = min(S, kmax)
    idx = np.linspace(0, S-1, k, dtype=int)
    return t[idx]

# #% Specify GPU
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# print('Using Device: ', device)
# torch.cuda.set_device(device)
#% Select device (CPU/GPU 自动)

if torch.cuda.is_available():
    device = torch.device('cuda')           # 或者 'cuda:0'
    print('Using Device:', device)
else:
    device = torch.device('cpu')
    print('Using Device: CPU')

## mat 
def save_to_mat_light(filepath, temp_model_stack, temp_E_cpu, propagated_fields_list, *,
                    kmax=16, save_amplitude_only=True):

    masks = np.asarray(temp_model_stack, dtype=np.float32)  # (L,H,W)
    # 输入场 -> 振幅
    E = temp_E_cpu.detach().cpu()
    E_amp = torch.abs(E).to(torch.float32).numpy() if save_amplitude_only else E.numpy()

    # 从第一个传播序列里抽样 K 张
    prop_slices = None
    if len(propagated_fields_list) > 0:
        t = propagated_fields_list[0].detach().cpu()  
        if t.ndim == 2:
            a = torch.abs(t).to(torch.float32)[None, ...] if save_amplitude_only else t[None, ...]
        else:
            S = t.shape[0]
            K = min(int(kmax), S)
            idx = np.linspace(0, S-1, K, dtype=int)
            t = t[idx]  # (K,H,W)
            a = torch.abs(t).to(torch.float32) if save_amplitude_only else t
        prop_slices = a.numpy().astype(np.float32, copy=False)

    mdict = {
        "temp_model": masks,      # (L,H,W) float32
        "temp_E": E_amp,          # (H,W)   float32
    }
    if prop_slices is not None:
        mdict["propagated_slices"] = prop_slices  # (K,H,W) float32

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    savemat(filepath, mdict, do_compression=True)
    print("Saved (v5 light):", filepath)

def save_to_mat_light_plus(
    filepath,
    *,
    temp_model_stack,        # (L,H,W) float32 相位 (rad)
    E0,                      # (H,W) complex64 固定样本输入场
    scans_dict,              # dict: name -> {"stack": (S,H,W) complex64, "z": (S,) float64}
    E_camera=None,           # (H,W) complex64，最终到相机面的场（
    sample_stacks_kmax=20,   
    save_amplitude_only=False,
    meta: dict | None = None,
):
    mdict = {}
    mdict["temp_model"] = np.asarray(temp_model_stack, dtype=np.float32)

    # E0：可选只存振幅，或存复数
    if save_amplitude_only:
        mdict["E0_amp"] = np.abs(np.asarray(E0))
    else:
        mdict["E0"] = np.asarray(E0)

    # 扫描堆栈：逐个存 name_stack / name_z
    for name, pack in scans_dict.items():
        stk = np.asarray(pack["stack"])  # (S,H,W) complex64
        z   = np.asarray(pack["z"])      # (S,) float64

        if sample_stacks_kmax is not None and stk.ndim == 3 and stk.shape[0] > sample_stacks_kmax:
            idx = np.linspace(0, stk.shape[0]-1, sample_stacks_kmax, dtype=int)
            stk = stk[idx]
            z   = z[idx]

        if save_amplitude_only:
            mdict[f"{name}_amp"] = np.abs(stk)
        else:
            mdict[f"{name}"] = stk
        mdict[f"{name}_z"] = z

    # 相机面的场
    if E_camera is not None:
        if save_amplitude_only:
            mdict["E_camera_amp"] = np.abs(np.asarray(E_camera))
        else:
            mdict["E_camera"] = np.asarray(E_camera)

    if meta is not None:
        mdict["meta_json"] = json.dumps(meta, ensure_ascii=False)

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    savemat(filepath, mdict, do_compression=True)
    print("Saved (v5 plus):", filepath)


#% padding funktion
def complex_pad(E, pad_h, pad_w):
    # E: (..., H, W) complex64
    Er = torch.view_as_real(E)                         # (..., H, W, 2)
    Er_pad = F.pad(Er, (0, 0, pad_w, pad_w, pad_h, pad_h), mode='constant', value=0)
    return torch.view_as_complex(Er_pad.contiguous())  # 确保存储连续

def complex_crop(E_pad, H, W, pad_h, pad_w):
    return E_pad[..., pad_h:pad_h+H, pad_w:pad_w+W].contiguous()

def make_pad_slices(H, W, padding_ratio=None, pad_px=None):
    """根据比例或像素给出 pad_h/pad_w 以及中心切片"""
    if pad_px is None:
        pad_h = int(round(H * padding_ratio))
        pad_w = int(round(W * padding_ratio))
    else:
        pad_h = pad_w = int(pad_px)
    sl_h = slice(pad_h, pad_h + H)
    sl_w = slice(pad_w, pad_w + W)
    return pad_h, pad_w, (sl_h, sl_w)

def complex_pad_asymm(E, pad_top, pad_bottom, pad_left, pad_right):
    # E: (..., H, W) complex64
    Er = torch.view_as_real(E)  # (..., H, W, 2)
    Er_pad = F.pad(Er, (0, 0, pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    return torch.view_as_complex(Er_pad.contiguous())


#% robust测试：平移
def _shift_phase_bilinear(phase_2d: torch.Tensor, dx_mm: float, dy_mm: float, pixel_size_m: float) -> torch.Tensor:
    """
    把 2D 相位（弧度）在 x/y 方向平移 (dx_mm, dy_mm)，双线性插值。
    +x 向右，+y 向下；超界补相位 0（= 透明）。
    """
    with torch.no_grad():
        device = phase_2d.device
        H, W   = phase_2d.shape
        dx_pix = (dx_mm * 1e-3) / pixel_size_m
        dy_pix = (dy_mm * 1e-3) / pixel_size_m
        sx = 2.0 * dx_pix / (W - 1)
        sy = 2.0 * dy_pix / (H - 1)
        yy = torch.linspace(-1, 1, H, device=device)
        xx = torch.linspace(-1, 1, W, device=device)
        gy, gx = torch.meshgrid(yy, xx, indexing='ij')
        grid = torch.stack([gx - sx, gy - sy], dim=-1)[None]  # (1,H,W,2)
        out  = F.grid_sample(phase_2d[None,None], grid, mode='bilinear',
                             padding_mode='zeros', align_corners=True)
        return out[0,0]

def apply_shift_to_model_masks(D2NN, dx_mm: float, dy_mm: float, pixel_size_m: float):
    """
    把模型中每一层的 phase 替换为“平移后的相位”，返回原相位列表，便于之后 restore。
    """
    originals = []
    for layer in D2NN.layers:
        # 取出原相位（leaf tensor），clone 一份保存
        p_orig = layer.phase.data.clone()
        originals.append(p_orig)

        # 生成平移后的相位
        p_shift = _shift_phase_bilinear(p_orig, dx_mm, dy_mm, pixel_size_m)
        layer.phase.data.copy_(p_shift)  

    return originals

def restore_model_masks(D2NN, originals):

    for layer, p in zip(D2NN.layers, originals):
        layer.phase.data.copy_(p)


#%% data generation (lightfield)
"""
case1:
    beam size: 176pixel
    layer size: 300pixel

case2:
    beam size: 176pixel
    layer size: 400pixel
"""

field_size = 50 #200#50 #the field size in eigenmodes_OM4 is 50 pixels
layer_size = 150 #400#300#100
#!!Parameter adjustment when mode switching： 1000
num_data = 1000 # options: 1. random datas 2.eigenmodes
num_modes = 55 #the mode number of MMF 3 6 10
focus_radius = 5 #4,15 #the size of detection regions(circle)


#用npy的代码
#eigenmodes_OM4 = np.load('eigenmodes_OM4_176.npy')
eigenmodes_OM4 = np.load('eigenmodes_OM4.npy')

MMF_data =  eigenmodes_OM4[:,:,0:num_modes].transpose(2,0,1)
#MMF_data =  eigenmodes_OM4[:,:,0:num_modes] # (H, W, Number)

# Normalize amplitude and maintain phase
MMF_data_amp_norm = (np.abs(MMF_data) - np.min(np.abs(MMF_data))) / (np.max(np.abs(MMF_data)) - np.min(np.abs(MMF_data)))
MMF_data_amp_norm = (np.abs(MMF_data) - np.min(np.abs(MMF_data))) / (np.max(np.abs(MMF_data)) - np.min(np.abs(MMF_data)))
MMF_data = MMF_data_amp_norm * np.exp(1j * np.angle(MMF_data))
# # Convert data to tensor
# MMF_data_tensor = torch.tensor(MMF_data, dtype=torch.complex64).to(device)
# generate the random datas
#Parameter adjustment when mode switching： 3
phase_option = 4
#phase_option 1: (0,0,...,0)
#phase_option 2: (0,2pi,...,2pi)
#phase_option 3: (0,pi,...,2pi)
#phase_option 4: eigenmodes
#phase_option 5: (0,pi,...,pi)


if phase_option == 4:
    num_data = num_modes # use the eigenmodes to train ODNN
    amplitudes = np.eye(num_modes)#[[1,0,0][0,1,0][0,0,1]]
    phases = np.zeros_like(amplitudes)   # 或 phases = np.zeros((num_modes, num_modes))

amplitudes_phases_ori = np.hstack((amplitudes[:, :], phases[:, 1:]))  # amplitudes (l2 norm) phases
amplitudes_phases = np.hstack((amplitudes[:, :], phases[:, 1:]/(2*np.pi)))  # amplitudes (l2 norm) phases (0-1)

# Generate complex weights vector with specified amplitudes and phases
complex_weights = amplitudes * np.exp(1j * phases)


MMF_data_ts = torch.from_numpy(MMF_data)
complex_weights_ts = torch.from_numpy(complex_weights)
image_data = generate_fields_ts(complex_weights_ts,MMF_data_ts,num_data,num_modes,field_size)



#%% labels generation upto the prediction case
'''
pred_case = 1: only amplitudes prediction
pred_case = 2: only phases prediction
pred_case = 3: amplitudes and phases prediction
pred_case = 4: amplitudes and phases prediction (extra energy phase area)
'''
#!! Parameter adjustment when mode switching： 3
pred_case = 1
label_data = torch.zeros([num_data,1,layer_size,layer_size])
label_size = layer_size

if pred_case == 1: # 3
    num_detector = num_modes
    MMF_Label_data = torch.zeros([layer_size,layer_size,num_detector])
    for index in range(num_detector):
        MMF_Label_data[:,:,index] =torch.from_numpy(create_labels(layer_size, layer_size, num_detector, focus_radius, index+1))
    for index in range(num_data):
        label_data[index,:,:,:] =  (torch.from_numpy(amplitudes_phases[index,0:num_modes]) * MMF_Label_data).sum(dim=2)


image_test_data = image_data
label_test_data = label_data
#%% train dataset generation
def preprocess(images, label):
    # images: (1, field_size, field_size), label: (1, label_size, label_size)
    _, Himg, Wimg = images.shape
    _, Hlab, Wlab = label.shape

    # --- image → pad 到 layer_size ---
    dh_img = layer_size - Himg
    dw_img = layer_size - Wimg
    pt_img = dh_img // 2
    pb_img = dh_img - pt_img
    pl_img = dw_img // 2
    pr_img = dw_img - pl_img

    zero_padded_image = complex_pad_asymm(images.squeeze(0), pt_img, pb_img, pl_img, pr_img).unsqueeze(0)

    # --- label → pad 到 layer_size（奇数差）---
    dh_lab = layer_size - Hlab
    dw_lab = layer_size - Wlab
    pt_lab = dh_lab // 2
    pb_lab = dh_lab - pt_lab
    pl_lab = dw_lab // 2
    pr_lab = dw_lab - pl_lab

    zero_padded_label = F.pad(label, (pl_lab, pr_lab, pt_lab, pb_lab))

    return zero_padded_image, zero_padded_label


batch_size = 16
# Create training dataset
train_dataset = [(preprocess(image_data[i], label_data[i])) for i in range(len(label_data))]
train_tensor_data = TensorDataset(*[torch.stack(tensors) for tensors in zip(*train_dataset)])
pin_mem = torch.cuda.is_available()
g = torch.Generator()
g.manual_seed(SEED)

train_loader = DataLoader(
    train_tensor_data,
    batch_size=batch_size,
    shuffle=True,               # 顺序会被 g 固定
    generator=g,                # 固定打乱
    # num_workers=0,
    # pin_memory=pin_mem,
    # persistent_workers=False
)


#%% Define test dataset
#test_loader = train_loader
#test_loader = DataLoader(train_tensor_data, batch_size=num_data, shuffle=False)

test_dataset = train_dataset          
test_tensor_data = train_tensor_data  
test_loader = DataLoader(test_tensor_data, batch_size=16, shuffle=False)


#%% Model definition
class DiffractionLayer(nn.Module):
    def __init__(self, units, dx, lam, z, device, pad_px=0):
        super().__init__()
        self.units = units      # 原始 H=W=units
        self.dx    = dx
        self.lam   = lam
        self.z     = z
        self.pad_px = pad_px    # 每边像素 padding

        self.phase = nn.Parameter(torch.randn(units, units, dtype=torch.float32))

        self.register_buffer("kz_base", self._make_kz(units, dx, lam, device))
        if pad_px > 0:
            units_pad = units + 2*pad_px
            self.register_buffer("kz_pad", self._make_kz(units_pad, dx, lam, device))
        else:
            self.kz_pad = None

    def _make_kz(self, N, dx, lam, device):
        fx = torch.fft.fftshift(torch.fft.fftfreq(N, d=dx)).to(device)
        fxx, fyy = torch.meshgrid(fx, fx, indexing='ij')
        argument = (2*torch.pi)**2 * ((1./lam)**2 - fxx**2 - fyy**2)
        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j*tmp).to(torch.complex64)
        return kz

    def _propagate(self, E, kz, z):
        E = E.to(torch.complex64)
        C = torch.fft.fftshift(torch.fft.fft2(E))
        return torch.fft.ifft2(torch.fft.ifftshift(C * torch.exp(1j * kz * z)))

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        phase_c = torch.exp(1j * self.phase.to(inputs.device, dtype=torch.float32)).to(torch.complex64)

        if self.pad_px > 0:
            pad_h = pad_w = self.pad_px
            Ein = complex_pad(inputs.squeeze(1), pad_h, pad_w)     # 光场外圈为0 
             
            phase_big = torch.ones(H+2*pad_h, W+2*pad_w, dtype=torch.complex64, device=inputs.device) #相位圈为1
            #phase_big = torch.zeros(H+2*pad_h, W+2*pad_w, dtype=torch.complex64, device=inputs.device) #相位圈为0，好像无所谓
            phase_big[pad_h:pad_h+H, pad_w:pad_w+W] = phase_c
            Ein = Ein * phase_big 

            Eout = self._propagate(Ein, self.kz_pad, self.z)

            # 5) 裁回原始尺寸
            Eout = complex_crop(Eout, H, W, pad_h, pad_w).unsqueeze(1)  # (B,1,H,W)
            return Eout
        else:
            Ein = inputs * phase_c.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            Eout = self._propagate(Ein, self.kz_base, self.z)
            return Eout



class Propagation(nn.Module):
    """
    自由传播层
    """
    def __init__(self, units, dx, lam, z, device, pad_px=0):
        super().__init__()
        self.units  = units     # 原始 H=W=units
        self.dx     = dx
        self.lam    = lam
        self.z      = z
        self.pad_px = int(pad_px)

        self.register_buffer("kz_base", self._make_kz(units, dx, lam, device))

        if self.pad_px > 0:
            units_pad = units + 2 * self.pad_px
            self.register_buffer("kz_pad", self._make_kz(units_pad, dx, lam, device))
        else:
            self.kz_pad = None

    def _make_kz(self, N, dx, lam, device):
        fx = torch.fft.fftshift(torch.fft.fftfreq(N, d=dx)).to(device)
        fxx, fyy = torch.meshgrid(fx, fx, indexing='ij')
        argument = (2 * torch.pi) ** 2 * ((1. / lam) ** 2 - fxx ** 2 - fyy ** 2)
        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j * tmp).to(torch.complex64)
        return kz

    def _propagate(self, E, kz, z):
        E = E.to(torch.complex64)
        C = torch.fft.fftshift(torch.fft.fft2(E))
        return torch.fft.ifft2(torch.fft.ifftshift(C * torch.exp(1j * kz * z)))

    def forward(self, inputs):
       
        assert inputs.is_complex(), "Propagation expects complex64 inputs."
        B, C, H, W = inputs.shape

        if self.pad_px > 0:
            p = self.pad_px
            # 去掉通道做 padding
            Ein = complex_pad(inputs.squeeze(1), p, p)           # (B, H+2p, W+2p)
            Eout = self._propagate(Ein, self.kz_pad, self.z)     # 传播在大画布
            Eout = complex_crop(Eout, H, W, p, p).unsqueeze(1)   # 裁回并补通道 (B,1,H,W)
            return Eout
        else:
            
            Eout = self._propagate(inputs, self.kz_base, self.z)
            return Eout


class RegressionDetector(nn.Module):
    def __init__(self):
        super(RegressionDetector, self).__init__()

    def forward(self, inputs):
        # Compute intensity of the field
        return torch.square(torch.abs(inputs)) #取了平方


#%% define the evaluation area
detectsize = 8 #15
# Generate detection regions using existing function
if pred_case ==1 or pred_case ==2:
    evaluation_regions = create_evaluation_regions(layer_size, layer_size, num_detector, focus_radius, detectsize)
    print("Detection Regions:", evaluation_regions)

#%% Define multiple D2NN models and train them
#layer层数:目前是3层 
num_layer_option = [5]   #, 3]#, 4]  # Define the different layer-number ODNN
#num_layer_option = [1]
# Lists to store results for different layer configurations
all_losses = [] #the loss for each epoch of each ODNN model
all_phase_masks = [] #the phase masks field of each ODNN model
all_predictions = [] #the output light field of each ODNN model
all_weights_pred_ODNN = [] #all the prediction weights on one ODNN
all_weights_pred = [] #all the prediction weights on each ODNN

# model parameters
"""
resoluton: 12.5um
layer distance 45.5768mm
layer numbers reflection times	2
z_prop: 20cm -> 10cm
"""
# SLM
z_layers   = 40e-6        # 原 47.571e-3  -> 40 μm
pixel_size = 1e-6
z_prop     = 120e-6        # 原 16.74e-2   -> 60 μm plus 40（最后一层到相机）
wavelength = 1568e-9      # 原 654e-9     -> 1550 nm
z_input_to_first = 40e-6  # 40 μm # 新增：输入面到第一层的传播距离

for num_layer in num_layer_option:
    print(f"\nTraining D2NN with {num_layer} layers...\n")

    # Define D2NN model
    class D2NNModel(nn.Module):
        def __init__(self, num_layers, layer_size, z_layers, z_prop, pixel_size, wavelength, device,
                    padding_ratio=0.5, z_input_to_first=0.0):
            super().__init__()
            pad_px = int(round(layer_size * padding_ratio))
            #加上了第一层的传播
            self.pre_propagation = Propagation(layer_size, pixel_size, wavelength, z_input_to_first, device, pad_px=pad_px)
            self.layers = nn.ModuleList([
                DiffractionLayer(layer_size, pixel_size, wavelength, z_layers, device, pad_px=pad_px)
                for _ in range(num_layers)
            ])
            self.propagation = Propagation(layer_size, pixel_size, wavelength, z_prop, device, pad_px=pad_px)
            self.regression  = RegressionDetector()  

        def forward(self, x):
            x = self.pre_propagation(x)
            for layer in self.layers:
                x = layer(x)            #  pad->prop->crop
            x = self.propagation(x)     #  pad->prop->crop
            x = self.regression(x)     
            return x


    # Initialize the D2NN model
    D2NN = D2NNModel(
        num_layers=num_layer,
        layer_size=layer_size,
        z_layers=z_layers,
        z_prop=z_prop,
        pixel_size=pixel_size,
        wavelength=wavelength,
        device=device,
        padding_ratio=0.5,
        z_input_to_first=z_input_to_first,   # NEW
    ).to(device)

    print(D2NN)

    # Training
    criterion = nn.MSELoss()  # Define loss function (对比的是loss)
    optimizer = optim.Adam(D2NN.parameters(), lr=1.99) 
    scheduler = ExponentialLR(optimizer, gamma=0.99)  
    epochs = 1000
    losses = []


    for epoch in range(epochs):
        start_time = time.time()
        D2NN.train()
        epoch_loss = 0
        for images, labels in train_loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32,   non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            outputs = D2NN(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)  # Calculate average loss for the epoch
        losses.append(avg_loss) # the loss for each model
        end_time = time.time()
        elapsed_time = end_time - start_time

        if epoch % 100 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.18f}, Time: {elapsed_time*100:.2f} seconds')
    all_losses.append(losses) #save the loss for each model
   
    # === after training ===
    ckpt_dir = "checkpoints"
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt = {
        "state_dict": D2NN.state_dict(),
        "meta": {
            "num_layers":        len(D2NN.layers),
            "layer_size":        layer_size,
            "z_layers":          z_layers,
            "z_prop":            z_prop,
            "pixel_size":        pixel_size,
            "wavelength":        wavelength,
            "padding_ratio":     0.5,         
            "field_size":        field_size,  
            "num_modes":         num_modes, 
            "z_input_to_first":  z_input_to_first, 
        }
    }
    save_path = os.path.join(ckpt_dir, f"odnn_{len(D2NN.layers)}layers.pth")
    torch.save(ckpt, save_path)
    print("✔ Saved model ->", save_path)
    # Free GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
 

    # Evaluate and visualize predictions
    D2NN.eval()

    phase_masks = []
    for i, layer in enumerate(D2NN.layers):
        phase = layer.phase.detach().cpu().numpy()
        phase = np.remainder(phase, 2 * np.pi)
        phase_masks.append(phase)
    all_phase_masks.append(phase_masks)

    all_weights_pred_ODNN = []
    all_predictions_batches = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)
            preds = D2NN(images)                                # (B,1,H,W)
            preds_np = preds.cpu().numpy()
            all_predictions_batches.append(preds_np)

            for b in range(preds_np.shape[0]):
                pred_hw = preds_np[b].squeeze()                 # (H,W)
                weights_pred = []
                for (x_start, x_end, y_start, y_end) in evaluation_regions:
                    weights_pred.append(np.mean(pred_hw[y_start:y_end, x_start:x_end]))
                weights_pred = np.array(weights_pred)

                if pred_case == 1:
                    l2 = np.linalg.norm(weights_pred, ord=2)
                    if l2 > 0: weights_pred = weights_pred / l2
                elif pred_case == 3:
                    w_amp = weights_pred[0:3] / np.linalg.norm(weights_pred[0:3], ord=2)
                    weights_pred = np.hstack((w_amp, weights_pred[-2:]))
                elif pred_case == 4:
                    w_amp = weights_pred[0:3] / np.linalg.norm(weights_pred[0:3], ord=2)
                    w_phase_extra = weights_pred[-3:] / np.linalg.norm(weights_pred[-3:], ord=2)
                    weights_pred = np.hstack((w_amp, w_phase_extra[:-1]))

                all_weights_pred_ODNN.append(weights_pred)

    predictions_np_full = np.concatenate(all_predictions_batches, axis=0).squeeze()

    all_predictions.append(predictions_np_full)
    all_weights_pred.append(all_weights_pred_ODNN)

num_samples_to_plot = min(10, len(train_dataset))
fig, axs = plt.subplots(num_samples_to_plot, 4, figsize=(20, num_samples_to_plot * 4))

for idx in range(num_samples_to_plot):
    image, label = train_dataset[idx]

    # Convert tensors to NumPy arrays
    image_abs_np = np.squeeze(np.abs(image.cpu().numpy()))
    image_phase_np = np.squeeze(np.angle(image.cpu().numpy()))
    label_np = np.squeeze(label.cpu().numpy())
    #pred_np = np.squeeze(all_predictions_np)[idx, :, :]

    # Plot the image (Amplitude)
    im1 = axs[idx, 0].imshow(image_abs_np, cmap='viridis')
    axs[idx, 0].set_title(f"Sample {idx + 1} - Amplitude")
    axs[idx, 0].axis('off')
    # Add colorbar to the amplitude image
    cbar1 = fig.colorbar(im1, ax=axs[idx, 0], fraction=0.046, pad=0.04)
    cbar1.set_label('Amplitude')

    # Plot the image (Phase)
    im2 = axs[idx, 1].imshow(image_phase_np, vmin=-1*np.pi, vmax=1*np.pi, cmap='viridis')
    axs[idx, 1].set_title(f"Sample {idx + 1} - Phase")
    axs[idx, 1].axis('off')
    # Add colorbar to the amplitude image
    cbar2 = fig.colorbar(im2, ax=axs[idx, 1], fraction=0.046, pad=0.04)
    cbar2.set_label('Amplitude')

    # Plot the label (Grayscale, with range [0,1])
    im3 = axs[idx, 2].imshow(label_np, vmin=0, vmax=1)# cmap='gray', vmin=0, vmax=1)
    axs[idx, 2].set_title(f"Sample {idx + 1} - Label")
    axs[idx, 2].axis('off')
    # Add colorbar to the label image
    cbar3 = fig.colorbar(im3, ax=axs[idx, 2], fraction=0.046, pad=0.04)
    cbar3.set_label('Label Value')

    # Plot the pred label (Grayscale, with range [0,1])
    # im4 = axs[idx, 3].imshow(np.squeeze(all_predictions_np)[idx,:,:])#, cmap='gray', vmin=0, vmax=1)
    # axs[idx, 3].set_title(f"Sample {idx + 1} - Pred Label")
    # axs[idx, 3].axis('off')
    # # Add colorbar to the label image
    # cbar4 = fig.colorbar(im4, ax=axs[idx, 3], fraction=0.046, pad=0.04)
    # cbar4.set_label('Label Pred')

plt.tight_layout()
plt.savefig("2.png", dpi=300)  
plt.close(fig)  


#%% evaluation
all_amplitudes_relative_diff = []   # save the relative amplitudes_diff of each models

# Normalize the vector
vec_normalized =np.mean( np.ones((num_modes, 1)) / np.linalg.norm(np.ones((num_modes, 1)),ord = 2))

all_amplitudes_diff = []  # save the amplitudes_diff of each models
all_phases_diff = []      # save the phases_diff of each models
all_average_amplitudes_diff = []  # save the average amplitudes_diff of each models
all_average_phases_diff = []  # save the average phases_diff of each models
all_complex_weights_pred = []  # save the predictions of each models
all_image_data_pred = [] # save the reconstruction images of each models

all_cc_real = []
all_cc_imag = []
all_cc_recon_amp = []
all_cc_recon_phase = []

for idx, num_layer in enumerate(num_layer_option):
    print(f"\nEvaluating ODNN with {num_layer} layers...\n")
    current_weights_pred = np.array(all_weights_pred[idx])

    if pred_case == 1:  # only amplitude weights
        current_amplitude_weights_pred = current_weights_pred         # (N, num_modes)
        N = len(current_amplitude_weights_pred)

        # 目标幅度/相位（与训练一致）
        if phase_option == 4:
            # 本征模 one-hot 幅度 + 你保留的单位阵相位（单位：rad）
            target_amp = amplitudes[:N, :num_modes]                   # (N, num_modes)
            current_phase_weights_pred = phases[:N, :num_modes]       # (N, num_modes)
        else:
            target_amp = amplitudes_phases[:N, :num_modes]
            current_phase_weights_pred = phases[:N, :]

        # L2 归一化后比较幅度误差
        target_amp = l2_normalize_rows(target_amp)
        current_amplitude_weights_pred = l2_normalize_rows(current_amplitude_weights_pred)
        amplitudes_diff = np.abs(target_amp - current_amplitude_weights_pred)
        average_amplitudes_diff = float(np.mean(amplitudes_diff))

        print("||pred||₂:", np.linalg.norm(current_amplitude_weights_pred, axis=1)[:5])
        print("||true||₂:", np.linalg.norm(target_amp, axis=1)[:5])

        #current_phase_weights_pred = phases_test  # use the true phases in test dataset


    
    all_amplitudes_diff.append(amplitudes_diff)
    all_average_amplitudes_diff.append(average_amplitudes_diff)


    print(f"{num_modes} Modes - Phase option {phase_option} - Pred case {pred_case} - ODNN with {num_layer} layers: amplitude weights error = {average_amplitudes_diff:.6f}")
    print(f"{num_modes} Modes - Phase option {phase_option} - Pred case {pred_case} - ODNN with {num_layer} layers: phase weights error = {all_average_phases_diff}")

    all_amplitudes_relative_diff.append(np.mean(amplitudes_diff/ vec_normalized))
    print(f"{num_modes} Modes - Phase option {phase_option} - Pred case {pred_case} - ODNN with {num_layer} layers: amplitude weights relative error = {all_amplitudes_relative_diff[idx]:.6f}")

    complex_weights_pred = current_amplitude_weights_pred * np.exp(1j * current_phase_weights_pred)
    all_complex_weights_pred.append(complex_weights_pred)
    complex_weights_pred_ts = torch.from_numpy(complex_weights_pred)

    image_data_pred = generate_fields_ts(complex_weights_pred_ts,MMF_data_ts,len(image_test_data),num_modes,field_size)
    image_data_pred_np = image_data_pred.cpu().numpy().squeeze()
    all_image_data_pred.append(image_data_pred_np)

    image_test_data_np = image_test_data.cpu().numpy().squeeze()

    cc_real = []
    cc_imag = []
    cc_recon_amp = []
    cc_recon_phase = []

    for i in range(len(image_test_data)):
        # cc of the amplitudes and phase distribution
        image_flat = image_test_data_np[i].ravel()
        pred_flat = image_data_pred_np[i].ravel()

        corr_amp = np.corrcoef(np.abs(image_flat), np.abs(pred_flat))[0, 1]
        cc_recon_amp.append(corr_amp)
        corr_phase = np.corrcoef(np.angle(image_flat), np.angle(pred_flat))[0, 1]
        cc_recon_phase.append(corr_phase)


        # cc of the real and imag part
        if phase_option == 4:
            # 真实复权重统一用训练时的 complex_weights
            corr_real = np.corrcoef(np.real(complex_weights[i]), np.real(complex_weights_pred[i]))[0, 1]
            cc_real.append(corr_real)
            corr_imag = np.corrcoef(np.imag(complex_weights[i]), np.imag(complex_weights_pred[i]))[0, 1]
            cc_imag.append(corr_imag)

        
    cc_recon_amp = np.array(cc_recon_amp)
    cc_recon_phase = np.array(cc_recon_phase)
    cc_real = np.array(cc_real)
    cc_imag = np.array(cc_imag)
    print(f"{num_modes} Modes - Phase option {phase_option} - Pred case {pred_case} - ODNN with {num_layer} layers: cc_recon_amp = {np.mean(cc_recon_amp):.6f}")
    print(f"{num_modes} Modes - Phase option {phase_option} - Pred case {pred_case} - ODNN with {num_layer} layers: cc_recon_amp_std = {np.std(cc_recon_amp):.6f}")
    print(f"{num_modes} Modes - Phase option {phase_option} - Pred case {pred_case} - ODNN with {num_layer} layers: cc_recon_phase = {np.mean(cc_recon_phase):.6f}")


    all_cc_real.append(cc_real)
    all_cc_imag.append(cc_imag)
    all_cc_recon_amp.append(cc_recon_amp)
    all_cc_recon_phase.append(cc_recon_phase)

    all_cc_real_np = np.array(all_cc_real)
    all_cc_imag_np = np.array(all_cc_imag)
    all_cc_recon_amp_np = np.array(all_cc_recon_amp)
    all_cc_recon_phase_np = np.array(all_cc_recon_phase)
    print(f"{num_modes} Modes - Phase option {phase_option} - Pred case {pred_case} - ODNN with {num_layer} layers: cc_recon_real = {np.mean(all_cc_real_np):.6f}")
    print(f"{num_modes} Modes - Phase option {phase_option} - Pred case {pred_case} - ODNN with {num_layer} layers: cc_recon_real_std = {np.std(all_cc_real_np):.6f}")
    print(f"{num_modes} Modes - Phase option {phase_option} - Pred case {pred_case} - ODNN with {num_layer} layers: cc_recon_imag = {np.mean(all_cc_imag_np):.6f}")
    print(f"{num_modes} Modes - Phase option {phase_option} - Pred case {pred_case} - ODNN with {num_layer} layers: cc_recon_imag_std = {np.std(all_cc_imag_np):.6f}")

@torch.no_grad()

def forward_full_intensity(D2NN, inputs):
 
    assert inputs.is_complex(), "inputs must be complex64/complex128"
    device = inputs.device
    B, C, H, W = inputs.shape
    p = int(D2NN.propagation.pad_px)
    x = complex_pad(inputs.squeeze(1), p, p)  # (B, H+2p, W+2p)
    #第一层前的传播
    if hasattr(D2NN, "pre_propagation") and D2NN.pre_propagation is not None:
        pre = D2NN.pre_propagation
        x = pre._propagate(x, pre.kz_pad, pre.z)

    for layer in D2NN.layers:
        phase_c = torch.exp(1j * layer.phase.to(device, dtype=torch.float32)).to(torch.complex64)
        phase_big = torch.ones(H+2*p, W+2*p, dtype=torch.complex64, device=device)
        phase_big[p:p+H, p:p+W] = phase_c
        x = x * phase_big
        x = layer._propagate(x, layer.kz_pad, layer.z)  # 保持大画布尺寸

    prop = D2NN.propagation
    x = prop._propagate(x, prop.kz_pad, prop.z)        # 仍是 (B,H+2p,W+2p)

    # 大画布强度（含溢出）
    I_big = torch.abs(x)**2                             # (B,H+2p,W+2p)

    # 裁剪回相机FOV得到 I_crop（与你网络输出口径一致）
    x_crop = complex_crop(x, H, W, p, p)                # (B,H,W)
    I_crop = torch.abs(x_crop)**2                       # (B,H,W)

    return I_crop, I_big


@torch.no_grad()
def evaluate_once(
    D2NN, *,
    use_circle_roi=True,
    roi_radius=None
):
    D2NN.eval()
    eps = 1e-12
    r_sig = int(detectsize if roi_radius is None else roi_radius)

    # 并集占比（全画布/裁剪）
    snr_ratio_full_list = []
    snr_ratio_crop_list = []
    throughput_list = []

    # 每个圆点各自占比（全画布/裁剪），逐样本收集，结尾再取均值
    ratio_each_full_batch = []
    ratio_each_crop_batch = []

    all_weights_pred_ODNN = []

    for images, _ in test_loader:
        images = images.to(device, dtype=torch.complex64, non_blocking=True)

        I_crop_t, I_big_t = forward_full_intensity(D2NN, images)
        I_crop = I_crop_t.detach().cpu().numpy()  # (B,H,W)
        I_big  = I_big_t.detach().cpu().numpy()   # (B,H+2p,W+2p)

        p = int(D2NN.propagation.pad_px)

        for b in range(I_crop.shape[0]):
            Ic = I_crop[b]
            Ib = I_big[b]

            total_full = float(Ib.sum())
            total_crop = float(Ic.sum())

            # --- 并集（所有圆）占比 ---
            if use_circle_roi:
                signal_full = _sum_signal_energy_circle(Ib, evaluation_regions, r_sig, offset=(p, p))
                signal_crop = _sum_signal_energy_circle(Ic, evaluation_regions, r_sig, offset=(0, 0))

                # 每个圆点各自的占比（相对全画布 & 相对裁剪）===
                _, ratios_full_vec = _spot_energy_ratios_circle(Ib, evaluation_regions, r_sig, offset=(p, p), eps=eps)
                _, ratios_crop_vec = _spot_energy_ratios_circle(Ic, evaluation_regions, r_sig, offset=(0, 0), eps=eps)
                ratio_each_full_batch.append(ratios_full_vec)
                ratio_each_crop_batch.append(ratios_crop_vec)
            else:
                
                signal_full = 0.0
                signal_crop = 0.0
                for (x0, x1, y0, y1) in evaluation_regions:
                    signal_full += float(Ib[y0+p:y1+p, x0+p:x1+p].sum())
                    signal_crop += float(Ic[y0:y1,   x0:x1].sum())

            snr_ratio_full_list.append(signal_full / (total_full + eps))
            snr_ratio_crop_list.append(signal_crop / (total_crop + eps))

            # 通光率（FOV能量 / 全画布能量）
            throughput_list.append(total_crop / (total_full + eps))
            ws = []
            for (x0, x1, y0, y1) in evaluation_regions:
                ws.append(float(Ic[y0:y1, x0:x1].mean()))
            w = np.asarray(ws, dtype=np.float64)

            if pred_case == 3:
                # 只对前 3 个“幅度”分量做归一化，
                n = np.linalg.norm(w[:num_modes])  # num_modes=3
                if n > 0:
                    w[:num_modes] = w[:num_modes] / n
            else:
                # 其他情形（如纯幅度）可保持整向量归一化
                n = np.linalg.norm(w)
                if n > 0:
                    w = w / n

            all_weights_pred_ODNN.append(w)


    # ====== 误差/重建 ======
    ## 1) 取目标幅度（与训练标签一致）
    current_amplitude_weights_pred = np.array(all_weights_pred_ODNN)
    N = len(current_amplitude_weights_pred)

    if phase_option == 4:
        # 本征模：幅度是 one-hot
        target_amp = amplitudes[:N, :num_modes]              # (N, num_modes)
    else:
        # 随机权重：用你构造的标签幅度部分
        target_amp = amplitudes_phases[:N, :num_modes]       # (N, num_modes)

    ## 2) L2 归一化后计算幅度误差
    target_amp = l2_normalize_rows(target_amp)
    current_amplitude_weights_pred = l2_normalize_rows(current_amplitude_weights_pred)
    amplitudes_diff = np.abs(target_amp - current_amplitude_weights_pred)
    avg_amp_err = float(np.mean(amplitudes_diff))

    ## 3) 
    if phase_option == 4:
        # 单位阵相位
        current_phase_weights_pred = phases[:N, :num_modes]   # (N, num_modes)
    else:
        current_phase_weights_pred = phases[:N, :num_modes]   # 保守写法，确保形状匹配

    ## 4) 重建并评估
    complex_weights_pred = current_amplitude_weights_pred * np.exp(1j * current_phase_weights_pred)

    image_data_pred_np = generate_fields_ts(
        torch.from_numpy(complex_weights_pred), MMF_data_ts, len(complex_weights_pred), num_modes, field_size
    ).cpu().numpy().squeeze()

    image_test_data_np = image_test_data.cpu().numpy().squeeze()

    cc_amp_list, cc_phase_list = [], []
    for i in range(min(len(image_test_data_np), len(image_data_pred_np))):
        a = image_test_data_np[i].ravel()
        b = image_data_pred_np[i].ravel()
        cc_amp_list.append(np.corrcoef(np.abs(a), np.abs(b))[0, 1])
        cc_phase_list.append(np.corrcoef(np.angle(a), np.angle(b))[0, 1])

    ratio_full = float(np.mean(snr_ratio_full_list))
    ratio_crop = float(np.mean(snr_ratio_crop_list))

    if len(ratio_each_full_batch) > 0:
        ratio_each_full_mean_vec = np.mean(np.vstack(ratio_each_full_batch), axis=0)
    else:
        ratio_each_full_mean_vec = np.array([], dtype=np.float64)

    def ratio_to_db(r):
        r = np.clip(r, eps, 1.0 - eps)
        return 10.0 * np.log10(r / (1.0 - r))

    snr_db_from_ratio_full = ratio_to_db(ratio_full)
    snr_each_db_mean_vec = ratio_to_db(ratio_each_full_mean_vec) if ratio_each_full_mean_vec.size else np.array([])

    return {
        "avg_amp_weight_err": avg_amp_err,
        "cc_amp": float(np.nanmean(cc_amp_list)),
        "snr_ratio_full": ratio_full,
    
    }



# 1) 基线（不平移）
baseline_metrics = evaluate_once(D2NN)
print("[baseline]", baseline_metrics)

# save.mat
save_dir = "results/plots"
os.makedirs(save_dir, exist_ok=True)

num_samples_to_display = 5
K = min(num_samples_to_display, image_test_data.shape[0])

# ✅ 先用 torch 计算，再转 numpy，避免 DeprecationWarning
image_data_abs   = torch.abs(image_test_data[:K]).cpu().numpy()      # (K, 1, H, W)
image_data_angle = torch.angle(image_test_data[:K]).cpu().numpy()    # (K, 1, H, W)

for idx, num_layer in enumerate(num_layer_option):
    # 预测这边本来就是 (N, H, W)，直接切 K
    image_data_pred_abs   = np.abs(all_image_data_pred[idx][:K])      # (K, H, W)
    image_data_pred_angle = np.angle(all_image_data_pred[idx][:K])    # (K, H, W)
    cc_amp_sample = all_cc_recon_amp[idx][:K]

    fig, axes = plt.subplots(K, 2, figsize=(8, 4*K))
    for i in range(K):
        # 左：真实幅度 —— 去掉通道维
        axes[i, 0].imshow(image_data_abs[i, 0], vmin=0, vmax=1)
        axes[i, 0].set_title(f"Sample {i+1}")
        axes[i, 0].axis('off')

        # 右：预测幅度
        axes[i, 1].imshow(image_data_pred_abs[i], vmin=0, vmax=1)
        axes[i, 1].set_title(f"reconstruct CC: {cc_amp_sample[i]:.4f}")
        axes[i, 1].axis('off')

    plt.suptitle(f"Amp. distribution of Real and Predicted Images({num_layer}_layer_ODNN)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, f"Amp_{num_layer}layers.png"), dpi=300)
    plt.close(fig)

    # —— 相位图 —— #
    fig, axes = plt.subplots(K, 2, figsize=(8, 4*K))
    for i in range(K):
        axes[i, 0].imshow(image_data_angle[i, 0], vmin=-np.pi, vmax=np.pi)
        axes[i, 0].set_title(f"Sample {i+1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(image_data_pred_angle[i], vmin=-np.pi, vmax=np.pi)
        axes[i, 1].set_title(f"reconstruct CC: {all_cc_recon_phase[idx][i]:.4f}")
        axes[i, 1].axis('off')

    plt.suptitle(f"Phase distribution of Real and Predicted Images({num_layer}_layer_ODNN)")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(save_dir, f"Phase_{num_layer}layers.png"), dpi=300)
    plt.close(fig)


## 存mask.mat
temp_dataset = test_dataset
zo  = z_input_to_first   # 原来是 0 -> 40e-6
z_start  = 0
z_step   = 5e-6   
z_prop_plus = z_prop + 0e-6    

# —— 固定切片：始终切同一张 —— #
FIXED_E_INDEX = 5  
def get_fixed_E(dataset, idx, device):
    if isinstance(dataset, list):
        img = dataset[idx][0]                 # (1,H,W) complex
    else:
        # TensorDataset(images, labels) 的第 0 个张量就是 images
        img = dataset.tensors[0][idx]         # (1,H,W) complex
    return img.squeeze(0).to(device)          # (H,W) complex

# 使用统一入口取“同一张图”
N = len(temp_dataset)
assert N > 0, "test_dataset 为空"
idx_safe = FIXED_E_INDEX % N                  # 防越界
temp_E = get_fixed_E(temp_dataset, idx_safe, device)

flag_savemat = 1
save_root = "./results_MD"
os.makedirs(save_root, exist_ok=True)
run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename_prefix = f"ODNN_vis_{run_stamp}"

def maybe_to_device(x, device):
    return x.to(device) if torch.is_tensor(x) else x

propagated_fields = []

# 切片和mask都存的mat
for i_model in range(len(all_phase_masks)):
    print(f'\nVisualizing model {i_model + 1}/{len(all_phase_masks)} (Model index: {i_model})')
    temp_model = all_phase_masks[i_model]  # list of (H,W) phase masks [rad]
    save_dir = os.path.join(save_root, f"m{i_model+1}")
    os.makedirs(save_dir, exist_ok=True)

    # 用字典收集每一段扫描堆栈 + z 轴
    scans = {}

    # 固定输入场
    Eo = temp_E  # (H,W) complex
    # ---- 输入 -> 第一层 的 z-scan（带 padding）----
    scan_input_stack, scan_input_z = plot_propagated_field_padded(
        Eo, z_start, zo, z_step, pixel_size, wavelength,
        pad_px=int(D2NN.propagation.pad_px), 
        kmax=25, ncols=5, save_path=os.path.join(save_dir, f"scan_input_m{i_model+1}.png"),
        cmap="RdBu_r"
    )
    scans["scan_input"] = {"stack": scan_input_stack.detach().cpu().numpy(), "z": scan_input_z.copy()}

    # 传播到第一层处的“截面”（与网络一致）
    if abs(zo) > 0:
        layer0 = D2NN.layers[0]
        p0 = int(layer0.pad_px); H = W = layer0.units
        Ein = complex_pad(Eo, p0, p0)
        Eout = layer0._propagate(Ein, layer0.kz_pad, zo)
        Ei = complex_crop(Eout, H, W, p0, p0)  # (H,W) complex
    else:
        Ei = Eo

    # 逐层：相位 → z-scan → 层间传播
    for i_layer in range(len(temp_model)):
        print(f'  Layer {i_layer + 1}/{len(temp_model)}...')

        # 乘相位
        temp_mask = torch.from_numpy(temp_model[i_layer]).to(Ei.device).to(torch.float32)
        Ei = Ei * torch.exp(1j * temp_mask)

        # 层间 z 扫描（带 padding）
        scan_name = f"scan_layer{i_layer+1}"
        scan_stack, scan_z = plot_propagated_field_padded(
            Ei, z_start, z_layers, z_step, pixel_size, wavelength,
            pad_px=int(D2NN.layers[0].pad_px), 
            kmax=25, ncols=5, save_path=os.path.join(save_dir, f"{scan_name}_m{i_model+1}.png"),
            cmap="RdBu_r"
        )
        scans[scan_name] = {"stack": scan_stack.detach().cpu().numpy(), "z": scan_z.copy()}

        # 层间传播（与模型一致的核 + 裁剪）
        layer = D2NN.layers[i_layer]
        p = int(layer.pad_px); H = W = layer.units
        Ein = complex_pad(Ei, p, p)
        Eout = layer._propagate(Ein, layer.kz_pad, z_layers)
        Ei = complex_crop(Eout, H, W, p, p)

        # 最后一层 -> 探测面
        if i_layer == len(temp_model) - 1:
            scan_name2 = "scan_to_camera"
            scan_stack2, scan_z2 = plot_propagated_field_padded(
                Ei, z_start, z_prop_plus, z_step, pixel_size, wavelength,
                pad_px=int(D2NN.propagation.pad_px), 
                kmax=25, ncols=5, save_path=os.path.join(save_dir, f"{scan_name2}_m{i_model+1}.png"),
                cmap="RdBu_r"
            )
            scans[scan_name2] = {"stack": scan_stack2.detach().cpu().numpy(), "z": scan_z2.copy()}

            # 用网络里的 Propagation 层传播到相机（与训练推理一致）
            prop = D2NN.propagation
            p_cam = int(prop.pad_px); H = W = D2NN.layers[0].units
            Ein  = complex_pad(Ei, p_cam, p_cam)
            Eout = prop._propagate(Ein, prop.kz_pad, prop.z)
            Ei_cam   = complex_crop(Eout, H, W, p_cam, p_cam)  # (H,W) complex
            E_camera_np = Ei_cam.detach().cpu().numpy()

    # # ——— 保存 .mat（相位 + E0 + 各段扫描 + 相机面）———
    # if flag_savemat == 1:
    #     temp_model_stack = np.stack([np.asarray(m, dtype=np.float32) for m in temp_model], axis=0)
    #     light_mat = os.path.join(save_dir, f"{filename_prefix}_LIGHT_m{i_model+1}.mat")

    #     meta = {
    #         "z_start": float(z_start),
    #         "z_step":  float(z_step),
    #         "z_layers": float(z_layers),
    #         "z_prop":   float(z_prop),
    #         "z_prop_plus": float(z_prop_plus),
    #         "pixel_size": float(pixel_size),
    #         "wavelength": float(wavelength),
    #         "layer_size": int(layer_size),
    #         "padding_ratio": 0.5,
    #     }

    #     save_to_mat_light_plus(
    #         light_mat,
    #         temp_model_stack=temp_model_stack,
    #         E0=temp_E.detach().cpu().numpy(),
    #         scans_dict=scans,
    #         E_camera=E_camera_np,
    #         sample_stacks_kmax=20,      #
    #         save_amplitude_only=False,  # 置 True 可只存振幅，体积更小
    #         meta=meta
    #     )
    #     print("Saved ->", light_mat)

# === 只导出“每个模式在相机面的最终成像” ===
from math import ceil

out_dir = "./results_camera_each_mode"
os.makedirs(out_dir, exist_ok=True)

D2NN.eval()
all_cam_chunks = []  # 收集每个batch的相机面强度 (B,H,W)

with torch.no_grad():
    idx_global = 0
    cam_loader = DataLoader(train_tensor_data, batch_size=16, shuffle=False)
    for images, _ in cam_loader:
        # 与训练一致：complex64 + 放到同一 device
        images = images.to(device, dtype=torch.complex64, non_blocking=True)

        # 直接走“完整前向到相机面”
        I_crop, _ = forward_full_intensity(D2NN, images)   # I_crop: (B,H,W)
        I_crop_np = I_crop.detach().cpu().numpy()
        all_cam_chunks.append(I_crop_np)

        # 逐张保存 PNG（相机面强度）
        for b in range(I_crop_np.shape[0]):
            mode_id = idx_global + b + 1  # 1-based: 1..55
            arr = I_crop_np[b]
            vmax = np.percentile(arr, 99.0)  # 抑制少量高亮点，便于观察
            plt.figure(figsize=(3, 3))
            plt.imshow(arr, vmin=0, vmax=vmax, cmap="turbo")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, f"mode_{mode_id:02d}.png"), dpi=200)
            plt.close()
        idx_global += I_crop_np.shape[0]

# 拼成 (num_modes, H, W) 并保存 mat/npy，便于后处理
I_camera_all = np.concatenate(all_cam_chunks, axis=0)
savemat(os.path.join(out_dir, "camera_per_mode.mat"), {"I_camera": I_camera_all})
np.save(os.path.join(out_dir, "camera_per_mode.npy"), I_camera_all)

# 做一个 55 张小图的拼图总览
cols = 11
rows = ceil(I_camera_all.shape[0] / cols)
fig, axes = plt.subplots(rows, cols, figsize=(1.8*cols, 1.8*rows))
axes = np.array(axes).reshape(-1)
for i in range(rows*cols):
    ax = axes[i]
    if i < I_camera_all.shape[0]:
        arr = I_camera_all[i]
        vmax = np.percentile(arr, 99.0)
        ax.imshow(arr, vmin=0, vmax=vmax, cmap="turbo")
        ax.set_title(f"m{i+1}", fontsize=7)
    ax.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "camera_per_mode_montage.png"), dpi=250)
plt.close()

# ==== 统一色标并重新导出 ====
# 用全体像素的同一分位数做 vmax（也可以用 .max() 或固定数值）
vmin_global = 0.0
vmax_global = np.percentile(I_camera_all, 99.0)  
out_dir_uniform = "./results_camera_each_mode_uniform"
os.makedirs(out_dir_uniform, exist_ok=True)

for i in range(I_camera_all.shape[0]):
    arr = I_camera_all[i]
    plt.figure(figsize=(3, 3))
    plt.imshow(arr, vmin=vmin_global, vmax=vmax_global, cmap="turbo")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir_uniform, f"mode_{i+1:02d}.png"), dpi=200)
    plt.close()


cols = 11
rows = ceil(I_camera_all.shape[0] / cols)
fig, axes = plt.subplots(rows, cols, figsize=(1.8*cols, 1.8*rows))
axes = np.array(axes).reshape(-1)
for i, ax in enumerate(axes):
    ax.axis("off")
    if i < I_camera_all.shape[0]:
        arr = I_camera_all[i]
        im = ax.imshow(arr, vmin=vmin_global, vmax=vmax_global, cmap="turbo")
        ax.set_title(f"m{i+1}", fontsize=7)
plt.tight_layout()
plt.savefig(os.path.join(out_dir_uniform, "camera_per_mode_montage_uniform.png"), dpi=250)
plt.close()

# 合成图（平均）
I_mean = I_camera_all.mean(axis=0)

# 用 99 分位做色标
vmax_mean = np.percentile(I_mean, 99.0)   
plt.figure(figsize=(4, 4))
plt.imshow(I_mean, vmin=0.0, vmax=vmax_mean, cmap="turbo")
plt.title("Mean of 55 camera intensities")
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(out_dir_uniform, "camera_mean_55.png"), dpi=300)
plt.close()



