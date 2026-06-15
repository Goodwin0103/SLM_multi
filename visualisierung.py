import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import json
from scipy.io import loadmat
import imageio.v2 as imageio
import math
import pandas as pd
import os, re, glob
from matplotlib.cm import get_cmap


## 可视化生成的xlsx的mask
def _read_mask_any(path_no_ext):
    """
    优先尝试读 .xlsx；失败则读 .csv
    返回 np.ndarray(float32)。无法读取则抛异常。
    """
    xlsx = path_no_ext + ".xlsx"
    csv  = path_no_ext + ".csv"
    # try xlsx
    if os.path.exists(xlsx):
        try:
            arr = pd.read_excel(xlsx, header=None).values.astype(np.float32)
            return arr
        except Exception as e:
            print(f"[info] 读取 {os.path.basename(xlsx)} 失败：{e}，尝试 CSV …")
    # try csv
    if os.path.exists(csv):
        return np.loadtxt(csv, delimiter=",", dtype=np.float32)
    raise FileNotFoundError(f"既没有 {xlsx} 也没有 {csv}")

def plot_all_layer_masks(save_dir, filename_prefix):
    """
    在 save_dir 内查找 {prefix}_MASK_layer*.xlsx/.csv，
    读取并按层号排序绘制。
    """
    # 找文件（xlsx 或 csv）
    patt_xlsx = os.path.join(save_dir, f"{filename_prefix}_MASK_layer*.xlsx")
    patt_csv  = os.path.join(save_dir, f"{filename_prefix}_MASK_layer*.csv")
    files = glob.glob(patt_xlsx) + glob.glob(patt_csv)
    if not files:
        raise FileNotFoundError("未找到任何 layer 的 xlsx/csv 文件。")

    # 提取层号并去重成“无扩展”的前缀
    rx = re.compile(r"_layer(\d+)\.(xlsx|csv)$", re.IGNORECASE)
    items = []
    for f in files:
        m = rx.search(f)
        if m:
            layer_id = int(m.group(1))
            base_noext = re.sub(r"\.(xlsx|csv)$", "", f, flags=re.IGNORECASE)
            items.append((layer_id, base_noext))
    # 去重（同一层若同时存在 xlsx/csv，保留一个“无扩展”的路径）
    uniq = {}
    for lid, base in items:
        uniq[lid] = base
    # 按层号排序
    layers = sorted(uniq.items(), key=lambda t: t[0])
    if not layers:
        raise RuntimeError("匹配到文件，但未解析到层号。请检查命名格式 *_layerX.xlsx/csv")

    # 读取所有层
    masks = []
    for lid, base in layers:
        arr = _read_mask_any(base)
        masks.append((lid, arr))

    # 布局：尽量接近方阵
    L = len(masks)
    cols = int(np.ceil(np.sqrt(L)))
    rows = int(np.ceil(L / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.atleast_2d(axes)
    vmax = 2*np.pi  # 你保存的是弧度；若保存的是度数请改成 360

    for idx, (lid, arr) in enumerate(masks):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        im = ax.imshow(np.mod(arr, 2*np.pi), vmin=0, vmax=vmax, cmap="hsv")
        ax.set_title(f"Layer {lid} (rad)")
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # 把空白子图关掉
    for k in range(len(masks), rows*cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")

    plt.tight_layout()
    plt.savefig("excel_visualsierung.png", dpi=300)  
    plt.close(fig) 

save_dir = "results_MD/m1"                    
filename_prefix = "ODNN_vis_20251009_155450"   # run_stamp 前缀
plot_all_layer_masks(save_dir, filename_prefix)




## 存的mat部分内容，加载最后一段出来看看结果，和verify的结果应该一样
mat_path = "results_MD/m1/ODNN_vis_20251009_155450_LIGHT_m1.mat"  
D = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

stack = D["scan_to_camera"]                 # (S,H,W) 或 (H,W)
z = D["scan_to_camera_z"].ravel() if "scan_to_camera_z" in D else None
if stack.ndim == 2:
    stack = stack[None, ...]
S = stack.shape[0]

# --- 选幅度或强度 ---
A = np.abs(stack)         # 幅度
# A = np.abs(stack)**2    # 若想看强度就用这一行

# 统一归一化（防止极端值影响）：[0, p99]
vmax = np.percentile(A, 99)
A_n = np.clip(A / (vmax + 1e-12), 0, 1)

# 彩色映射
cmap = get_cmap("turbo")  # 可改: 'viridis'/'plasma'/'magma'...
cols = int(math.ceil(math.sqrt(S)))
rows = int(math.ceil(S / cols))
fig, axes = plt.subplots(rows, cols, figsize=(3.2*cols, 3.2*rows))
axes = np.atleast_2d(axes)

# 用 0..1 范围作图，这样色条一致
for i in range(rows*cols):
    r, c = divmod(i, cols)
    ax = axes[r, c]
    if i < S:
        im = ax.imshow(A_n[i], vmin=0, vmax=1, cmap=cmap)
        title = f"z={z[i]*1e6:.0f} μm" if z is not None else f"frame {i}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    else:
        ax.axis("off")

# 单一色条（对应 0..p99 原始幅度）
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
cbar.set_label("Normalized amplitude (0..p99)", fontsize=8)

plt.tight_layout()
plt.savefig("scan_to_camera_montage_color.png", dpi=300)
plt.close(fig)
print("✔ 已保存: scan_to_camera_montage_color.png")