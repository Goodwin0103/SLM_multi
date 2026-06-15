import torch, numpy as np, pandas as pd, matplotlib.pyplot as plt, os
from scipy.io import loadmat
from odnn_model import D2NNModel, complex_pad, complex_crop
import numpy as np
import pandas as pd
from scipy.io import loadmat
from ODNN_functions import create_evaluation_regions
import numpy as np
import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
import mat73

def load_complex_modes_from_mat73(mat_path, key=None, key_candidates=("modes_field","eigenmodes_OM4_176","modes","E")):
    D = mat73.loadmat(mat_path)  # 支持 v7.3
    # 选键
    k = key if key else next((kk for kk in key_candidates if kk in D), None)
    if k is None:
        # 兜底：取第一个非空数组
        for kk,v in D.items():
            if isinstance(v, np.ndarray):
                k = kk; break
        if k is None:
            raise KeyError("在 mat 里没找到数组键")
    A = np.array(D[k])

    # 复数还原（mat73 通常已是复数 ndarray）
    if not np.iscomplexobj(A) and isinstance(A, dict) and "real" in A and "imag" in A:
        A = np.array(A["real"]) + 1j*np.array(A["imag"])

    # 统一形状 -> (H,W,M)
    A = np.asarray(A)
    if A.ndim == 2:
        A = A[..., None]
    elif A.ndim == 3:
        # (M,H,W) -> (H,W,M)
        if A.shape[1] == A.shape[2]:
            A = np.transpose(A, (1,2,0))
    else:
        raise ValueError(f"期望 2D/3D，得到 ndim={A.ndim}")
    return A.astype(np.complex64)

def _to_show(img, mode="amplitude", pclip=99.0):
    """
    mode 可选: 'amplitude' | 'intensity' | 'real' | 'imag' | 'phase'
    """
    a = np.asarray(img)
    if np.iscomplexobj(a):
        if mode == "amplitude":
            a = np.abs(a)
        elif mode == "intensity":
            a = np.abs(a)**2
        elif mode == "real":
            a = a.real
        elif mode == "imag":
            a = a.imag
        elif mode == "phase":
            a = np.angle(a)   # [-pi, pi]
        else:
            raise ValueError("mode not supported")
    else:
        # 实数输入：直接用
        pass

    a = a.astype(np.float32)
    # 对相位不做百分位归一（避免环状断层），其余做 pclip 归一
    if mode != "phase":
        vmax = np.percentile(a, pclip)
        vmax = vmax if vmax > 0 else (a.max() if a.max() > 0 else 1.0)
        a = np.clip(a / vmax, 0, 1)
    return a

def plot_inputs_outputs_grid(inputs_complex_hw_list, outsA_hw, outsB_hw,
                             titleA="Output", titleB="Expected Output",
                             mode_in="amplitude", mode_out="intensity",
                             pclip=99.0, save_path="inputs_outputs_grid.png",
                             dpi=300, cmap="turbo"):
    N = len(inputs_complex_hw_list)
    assert outsA_hw.shape[0] == N and outsB_hw.shape[0] == N

    fig, axes = plt.subplots(N, 3, figsize=(9, 3*N), squeeze=False)
    for i in range(N):
        img_in = _to_show(inputs_complex_hw_list[i], mode=mode_in, pclip=pclip)
        img_A  = _to_show(outsA_hw[i],               mode=mode_out, pclip=pclip)
        img_B  = _to_show(outsB_hw[i],               mode=mode_out, pclip=pclip)

        ax = axes[i, 0]; ax.imshow(img_in, cmap=cmap); ax.set_axis_off(); ax.set_title(f"Input {i+1}")
        ax = axes[i, 1]; ax.imshow(img_A,  cmap=cmap); ax.set_axis_off(); ax.set_title(f"{titleA} ({i+1})")
        ax = axes[i, 2]; ax.imshow(img_B,  cmap=cmap); ax.set_axis_off(); ax.set_title(f"{titleB} ({i+1})")

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi)
    plt.close(fig)
    print("✔ 保存图：", os.path.abspath(save_path))


def _masks_from_eval_regions(shape_hw, evaluation_regions, r_sig, center_offset=(0,0)):
    H, W = shape_hw
    Y, X = np.ogrid[:H, :W]
    offx, offy = center_offset  # 注意：offx 对列(x)，offy 对行(y)

    masks, centers = [], []
    for (x0, x1, y0, y1) in evaluation_regions:
        cx = int(round((x0 + x1) / 2.0 + offx))
        cy = int(round((y0 + y1) / 2.0 + offy))
        m  = (X - cx)**2 + (Y - cy)**2 <= (r_sig**2)
        masks.append(m); centers.append((cx, cy))
    return np.stack(masks, axis=0), centers

def _ring_mask(shape_hw, cx, cy, r_in, r_out):
    H, W = shape_hw
    Y, X = np.ogrid[:H, :W]
    rsq = (X - cx)**2 + (Y - cy)**2
    return (rsq >= r_in**2) & (rsq <= r_out**2)

def compare_spots_area_energy(
    IA: np.ndarray,
    IB: np.ndarray,
    evaluation_regions,   # [(x0,x1,y0,y1), ...] 共6个
    r_sig: int,           # 信号圆半径(像素)
    center_offset=(0,0),  # 大画布有padding时用 (p,p)，否则(0,0)
    ring_inner_pad=3,     # 信号圆外再空3px起做环
    ring_thickness=8,     # 环宽度
    area_thresh_ratio=0.5 # “面积”阈值= 局部峰值的50%
):
    # 强度图 -> float64
    IA = np.asarray(IA, dtype=np.float64)
    IB = np.asarray(IB, dtype=np.float64)

    sig_masks, centers = _masks_from_eval_regions(IA.shape, evaluation_regions, r_sig, center_offset)

    rows = []
    for k, (sig, (cx, cy)) in enumerate(zip(sig_masks, centers), start=1):
        # 局部峰值（在圆内）
        peakA = IA[sig].max()
        peakB = IB[sig].max()
        thrA  = peakA * area_thresh_ratio
        thrB  = peakB * area_thresh_ratio

        # 信号能量（圆内求和）
        EA = float(IA[sig].sum())
        EB = float(IB[sig].sum())

        # “面积(>阈值)”——仅统计圆内且高于阈值的像素个数
        areaA = int((IA[sig] > thrA).sum())
        areaB = int((IB[sig] > thrB).sum())

        # 背景环（不含信号圆）
        r_in  = r_sig + ring_inner_pad
        r_out = r_in + ring_thickness
        ring  = _ring_mask(IA.shape, cx, cy, r_in, r_out) & (~sig)

        if np.any(ring):
            bgA = float(IA[ring].mean())
            bgB = float(IB[ring].mean())
        else:
            bgA = np.nan; bgB = np.nan

        rows.append({
            "spot": k,
            "E_A": EA, "E_B": EB, "E_B/A": EB / (EA + 1e-12),
            "peak_A": peakA, "peak_B": peakB, 
        })

    df = pd.DataFrame(rows)
    return df


# ---- 读图并转灰度 ----
def load_gray(path: str) -> np.ndarray:
    img = plt.imread(path).astype(np.float32)
    if img.ndim == 3:  # RGB 或 RGBA
        img = img[..., :3]  # 忽略 alpha
        img = img @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    img -= img.min()
    if img.max() > 0: img /= img.max()
    return img

# ---- 中心裁剪到相同尺寸 ----
def center_crop_to_same(a: np.ndarray, b: np.ndarray):
    h = min(a.shape[0], b.shape[0]); w = min(a.shape[1], b.shape[1])
    def cc(x):
        y0 = (x.shape[0] - h)//2; x0 = (x.shape[1] - w)//2
        return x[y0:y0+h, x0:x0+w]
    return cc(a), cc(b)

# ---- 简易相位相关平移估计 & 整像素roll对齐 ----
def estimate_shift(a: np.ndarray, b: np.ndarray):
    a = a - a.mean(); b = b - b.mean()
    Fa = np.fft.rfftn(a); Fb = np.fft.rfftn(b)
    R = Fa * np.conj(Fb); R /= (np.abs(R) + 1e-12)
    r = np.fft.irfftn(R, s=a.shape)
    peak = np.unravel_index(np.argmax(r), r.shape)
    dy, dx = peak
    if dy > a.shape[0]//2: dy -= a.shape[0]
    if dx > a.shape[1]//2: dx -= a.shape[1]
    return int(dy), int(dx)

def roll_align(img: np.ndarray, dy: int, dx: int):
    return np.roll(np.roll(img, dy, axis=0), dx, axis=1)

# ---- 指标 ----
def metrics(I1: np.ndarray, I2: np.ndarray):
    diff = I1 - I2
    mse = float(np.mean(diff**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    psnr = float(10.0 * np.log10(1.0/(mse + 1e-12)))
    v1 = I1.reshape(-1); v2 = I2.reshape(-1)
    v1c = v1 - v1.mean(); v2c = v2 - v2.mean()
    pearson = float(np.dot(v1c, v2c) / ((np.linalg.norm(v1c)*np.linalg.norm(v2c))+1e-12))
    ncc = float(np.sum(I1*I2) / ((np.linalg.norm(I1)*np.linalg.norm(I2))+1e-12))
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "PSNR": psnr, "Pearson_r": pearson, "NCC": ncc}

# ---- 一把梭函数 ----
def compare_pngs(p1: str, p2: str, out_prefix="cmp", align=True):
    I1 = load_gray(p1)
    I2 = load_gray(p2)
    I1, I2 = center_crop_to_same(I1, I2)

    # 归一化到相同总能量（也可把 'sum' 改成 'max'）
    I1 /= (I1.sum() + 1e-12)
    I2 /= (I2.sum() + 1e-12)

    dy = dx = 0
    if align:
        dy, dx = estimate_shift(I1, I2)
        I2 = roll_align(I2, dy, dx)

    m = metrics(I1, I2)
    print(f"[{out_prefix}] registered shift dy={dy}, dx={dx}")
    for k,v in m.items():
        print(f"{k:>10}: {v:.6f}")

    # 可视化与保存
    plt.figure(figsize=(9,3))
    plt.subplot(1,3,1); plt.imshow(I1, cmap="turbo"); plt.axis('off'); plt.title("Image A")
    plt.subplot(1,3,2); plt.imshow(I2, cmap="turbo"); plt.axis('off'); plt.title("Image B (aligned)" if align else "Image B")
    plt.subplot(1,3,3); plt.imshow(I1 - I2, cmap="turbo"); plt.axis('off'); plt.title("A - B")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_side_by_side.png", dpi=300)
    plt.show()

    # 单独保存差异热图
    plt.figure(figsize=(4,4))
    plt.imshow(I1 - I2, cmap="turbo"); plt.axis('off'); plt.title("Difference (A-B)")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_diff.png", dpi=300)
    plt.close('all')


def center_crop_or_pad_complex(E: np.ndarray, target_hw):
    """把复数场 E 居中裁剪/零填充到 target_hw=(H,W)。"""
    Ht, Wt = target_hw
    h, w = E.shape
    if h == Ht and w == Wt:
        return E.astype(np.complex64, copy=False)

    out = np.zeros((Ht, Wt), dtype=np.complex64)
    # 计算源/目标的居中起止
    y_src0 = max(0, (h - Ht)//2);          y_src1 = y_src0 + min(h, Ht)
    x_src0 = max(0, (w - Wt)//2);          x_src1 = x_src0 + min(w, Wt)
    y_dst0 = max(0, (Ht - h)//2);          y_dst1 = y_dst0 + min(h, Ht)
    x_dst0 = max(0, (Wt - w)//2);          x_dst1 = x_dst0 + min(w, Wt)
    out[y_dst0:y_dst1, x_dst0:x_dst1] = E[y_src0:y_src1, x_src0:x_src1]
    return out

@torch.no_grad()
def run_on_inputs_with_masks(model, mask_dir, inputs_complex_hw_list):
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".xlsx")])
    masks = [pd.read_excel(os.path.join(mask_dir, f), header=None).to_numpy(dtype=np.float32)
             for f in mask_files]
    assert len(masks) == len(model.layers), f"层数不匹配: {len(masks)} vs {len(model.layers)}"

    # 加载相位掩膜
    for layer, m in zip(model.layers, masks):
        layer.phase.copy_(torch.tensor(m, dtype=torch.float32, device=device))

    # 目标尺寸 = 掩膜尺寸
    Ht, Wt = masks[0].shape

    outs = []
    for Ek in inputs_complex_hw_list:
        Ek = center_crop_or_pad_complex(Ek, (Ht, Wt))  # ★ 关键一步：对齐到掩膜大小
        Ek = torch.tensor(Ek, dtype=torch.complex64, device=device).unsqueeze(0).unsqueeze(0)
        Ik = model(Ek).squeeze().detach().cpu().numpy()
        outs.append(Ik.astype(np.float64))
    return np.stack(outs, axis=0)  # (N,Ht,Wt)

modes_hwM = load_complex_modes_from_mat73('mmf_6modes_25_PD_1.15.mat', key='modes_field')
modes_hwM = modes_hwM[..., :6]  # 只取前6个
inputs_modes = [modes_hwM[..., i] for i in range(modes_hwM.shape[-1])]  # list of 6*(H,W) complex
inputs = inputs_modes       

# === 两套掩膜目录（都要是每层一个 .xlsx 的文件夹）===
mask_dir_A = "results_MD/m1"
mask_dir_B = "results_MD"     # 对的model

layer_size = 100
z_layers = 40e-6
z_prop = 120e-6
pixel_size = 1e-6
wavelength = 1568e-9
z_input_to_first = 40e-6

# 创建同结构模型
D2NN = D2NNModel(
    num_layers=3,
    layer_size=layer_size,
    z_layers=z_layers,
    z_prop=z_prop,
    pixel_size=pixel_size,
    wavelength=wavelength,
    device=device,
    padding_ratio=0.5,
    z_input_to_first=z_input_to_first
).to(device)

# 前向得到 (N,H,W)
outsA = run_on_inputs_with_masks(D2NN, mask_dir_A, inputs)
outsB = run_on_inputs_with_masks(D2NN, mask_dir_B, inputs)

# === 统一 ROI & 半径 ===
H, W = outsA.shape[1], outsA.shape[2]
focus_radius = 5
detectsize = 10
evaluation_regions = create_evaluation_regions(H, W,N=6, radius=focus_radius, detectsize=detectsize)

# === 对每个输入做 6 斑点对比，并合并成一个表 ===
all_df = []
for i in range(len(inputs)):
    IA, IB = outsA[i], outsB[i]
    df_i = compare_spots_area_energy(IA, IB, evaluation_regions,
                                     r_sig=focus_radius, center_offset=(0,0))
    df_i.insert(0, "input_idx", i+1)   # 1..N
    all_df.append(df_i)

big = pd.concat(all_df, ignore_index=True)
print(big)
big.to_csv("spot_compare_ALL_inputs.csv", index=False)


inputs = [modes_hwM[..., i] for i in range(6)]    # 每个是复数 (H,W)
plot_inputs_outputs_grid(inputs, outsA, outsB,
                         titleA="Output",
                         titleB="Expected Output",
                         mode_in="amplitude",     # ← 输入
                         mode_out="intensity",    # ← 输出若已是强度就保持
                         pclip=99.0,
                         save_path="inputs_outputs_triptych.png")

# 1) 读入 E0（numpy 复数阵列）
E0_np = loadmat("results_MD/ODNN_vis_20251009_155450_LIGHT_m1.mat")["E0"].astype(np.complex64)

# 2) 组装为列表（长度 N=1）
inputs = [E0_np]  # 别用 torch，这里要 numpy；而且要列表

# 3) 两套掩膜目录（确认 B 目录是每层一个 .xlsx 的文件夹）
mask_dir_A = "results_MD/m1"
mask_dir_B = "results_MD"   # 

# 4) 前向得到 (N,H,W)
outsA1 = run_on_inputs_with_masks(D2NN, mask_dir_A, inputs)  # 形状: (1,H,W)
outsB1 = run_on_inputs_with_masks(D2NN, mask_dir_B, inputs)  # 形状: (1,H,W)

plot_inputs_outputs_grid(inputs, outsA1, outsB1,
                         titleA="Output",
                         titleB="Expected Output",
                         mode_in="amplitude",
                         mode_out="intensity",
                         pclip=99.0,
                         save_path="inputs_outputs_super.png")


# # ===== 3. 加载每层 mask（从 .xlsx） =====
# mask_dir = "results_MD/m1"  # 你的mask保存路径
# mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".xlsx")])
# masks = [pd.read_excel(os.path.join(mask_dir, f), header=None).to_numpy(dtype=np.float32)
#          for f in mask_files]
# print(f"✔ 读取到 {len(masks)} 个 mask, 每个大小: {masks[0].shape}")


# # ===== 4. 加载固定输入场 E0 =====
# # 若你已经保存过 .mat，就读回来；或者直接从之前 temp_E 保存的文件加载
# E0 = loadmat("results_MD/ODNN_vis_20251009_155450_LIGHT_m1.mat")["E0"]  # 注意改路径
# E0 = torch.tensor(E0, dtype=torch.complex64, device=device)
# print("✔ 加载固定输入场 E0")

# # ===== 5. 将 mask 手动赋值到模型中（覆盖训练参数） =====
# for layer, mask_np in zip(D2NN.layers, masks):
#     with torch.no_grad():
#         layer.phase.copy_(torch.tensor(mask_np, dtype=torch.float32, device=device))
# print("✔ 覆盖相位掩膜成功")

# # ===== 6. 前向传播到相机 =====
# with torch.no_grad():
#     E0 = E0.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
#     output_intensity = D2NN(E0).squeeze().cpu().numpy()

# # ===== 7. 可视化输出 =====
# plt.figure(figsize=(5,5))
# plt.imshow(output_intensity, cmap="turbo")
# plt.colorbar(label="Intensity")
# plt.title("Camera plane intensity (from loaded masks)")
# plt.axis("off")
# plt.tight_layout()
# plt.savefig("verify_camera_output_npy.png", dpi=300)
# plt.show()
# print("✔ 已保存: verify_camera_output_npy.png")

# compare_pngs("verify_camera_output_npy.png", "verify_camera_output.png",
#              out_prefix="verify_compare", align=True)

