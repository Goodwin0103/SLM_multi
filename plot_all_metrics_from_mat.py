"""
从 metrics_vs_layers_*.mat 文件一键重画所有指标图
用法:
    python plot_all_metrics_from_mat.py --mat path/to/metrics_vs_layers_xxx.mat
    或直接修改下面的 MAT_PATH
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

# ============================================================
# 参数(命令行 or 直接改这里)
# ============================================================
DEFAULT_MAT_PATH = ""
DEFAULT_OUT_DIR  = None   # None 则保存到 .mat 同目录的 replot/ 子目录

# ============================================================
# 解析命令行
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--mat", type=str, default=DEFAULT_MAT_PATH,
                    help="path to metrics_vs_layers_*.mat")
parser.add_argument("--out", type=str, default=DEFAULT_OUT_DIR,
                    help="output directory (default: <mat_dir>/replot)")
parser.add_argument("--dpi", type=int, default=300)
args = parser.parse_args()

mat_path = Path(args.mat)
assert mat_path.exists(), f"MAT file not found: {mat_path}"

out_dir = Path(args.out) if args.out else mat_path.parent / "replot"
out_dir.mkdir(parents=True, exist_ok=True)
print(f"📂 Loading: {mat_path}")
print(f"💾 Output : {out_dir}")

# ============================================================
# 加载 .mat
# ============================================================
data = loadmat(str(mat_path))

layers          = np.asarray(data["layers"]).ravel().astype(int)
wavelengths_m   = np.asarray(data["wavelengths_m"]).ravel()
wavelengths_nm  = np.asarray(data["wavelengths_nm"]).ravel()

M_amp_err       = np.asarray(data["avg_amp_error"])           # (NL, L)
M_rel_err       = np.asarray(data["avg_relative_amp_error"])
M_cc_mean       = np.asarray(data["cc_amp_mean"])
M_cc_std        = np.asarray(data["cc_amp_std"])
M_snr_db        = np.asarray(data["snr_db_full"])
M_iso_mean      = np.asarray(data["isolation_db_mean"])
M_iso_wc        = np.asarray(data["isolation_db_worst"])
M_iso_mean_all  = np.asarray(data["isolation_db_mean_allroi"])
M_iso_wc_all    = np.asarray(data["isolation_db_worst_allroi"])
M_target_wl     = np.asarray(data["target_wl_ratio"])

NL, L = M_amp_err.shape
print(f"✔ Loaded: {NL} layer counts × {L} wavelengths")
print(f"  layers      = {layers.tolist()}")
print(f"  wavelengths = {wavelengths_nm.tolist()} nm")

# ============================================================
# 全局样式
# ============================================================
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.8,
    "lines.markersize": 6,
})

wl_labels = [f"{w:.1f} nm" for w in wavelengths_nm]
cmap_wl   = plt.cm.viridis(np.linspace(0.15, 0.85, L))

# ============================================================
# 通用绘图函数
# ============================================================
def plot_metric(
    matrix,               # (NL, L)
    *,
    fname,
    ylabel,
    title,
    yerr=None,            # (NL, L) or None
    marker="o",
    linestyle="-",
    annotate_best=False,  # 是否标注最优层数
    best_mode="max",      # "max" or "min"
    higher_is_better=True,
):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for li in range(L):
        if yerr is not None:
            ax.errorbar(layers, matrix[:, li], yerr=yerr[:, li],
                        marker=marker, linestyle=linestyle, capsize=3,
                        color=cmap_wl[li], label=wl_labels[li])
        else:
            ax.plot(layers, matrix[:, li],
                    marker=marker, linestyle=linestyle,
                    color=cmap_wl[li], label=wl_labels[li])

    if annotate_best:
        agg = np.nanmean(matrix, axis=1)
        idx_best = np.nanargmax(agg) if best_mode == "max" else np.nanargmin(agg)
        ax.axvline(layers[idx_best], color="red", linestyle=":",
                   linewidth=1.2, alpha=0.7,
                   label=f"best={layers[idx_best]}L (avg={agg[idx_best]:.3f})")

    ax.set_xlabel("Number of layers")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(layers)
    ax.legend(loc="best")
    fig.tight_layout()
    out = out_dir / fname
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ {ylabel:<35s} -> {out.name}")
    return out

# ============================================================
# 1) 单图(每个指标一张)
# ============================================================
print("\n📊 [1/3] Generating per-metric plots ...")

plot_metric(M_amp_err, fname="01_avg_amp_error.png",
            ylabel="Average amplitude error",
            title="Average amplitude error vs. layers",
            annotate_best=True, best_mode="min")

plot_metric(M_rel_err, fname="02_avg_relative_amp_error.png",
            ylabel="Average relative amplitude error",
            title="Average relative amplitude error vs. layers",
            annotate_best=True, best_mode="min")

plot_metric(M_cc_mean, fname="03_cc_amp.png",
            ylabel="Reconstruction correlation (cc)",
            title="Reconstruction amplitude correlation vs. layers",
            yerr=M_cc_std, annotate_best=True, best_mode="max")

plot_metric(M_snr_db, fname="04_snr_db_full.png",
            ylabel="SNR (dB)",
            title="Signal containment (SNR, dB) vs. layers",
            annotate_best=True, best_mode="max")

plot_metric(M_iso_mean, fname="05_isolation_db_mean.png",
            ylabel="Isolation mean (dB)",
            title="Mode isolation — mean, same-λ (dB) vs. layers",
            annotate_best=True, best_mode="max")

plot_metric(M_iso_wc, fname="06_isolation_db_worst.png",
            ylabel="Isolation worst-case (dB)",
            title="Mode isolation — worst-case, same-λ (dB) vs. layers",
            marker="s", linestyle="--",
            annotate_best=True, best_mode="max")

plot_metric(M_iso_mean_all, fname="07_isolation_db_mean_allroi.png",
            ylabel="Isolation mean all-ROI (dB)",
            title="Mode isolation — mean, all ROIs (dB) vs. layers",
            annotate_best=True, best_mode="max")

plot_metric(M_iso_wc_all, fname="08_isolation_db_worst_allroi.png",
            ylabel="Isolation worst all-ROI (dB)",
            title="Mode isolation — worst-case, all ROIs (dB) vs. layers",
            marker="s", linestyle="--",
            annotate_best=True, best_mode="max")

plot_metric(M_target_wl, fname="09_target_wl_ratio.png",
            ylabel="TargetWL / AllWL (ROI)",
            title="Wavelength-demux ratio vs. layers",
            annotate_best=True, best_mode="max")

# ============================================================
# 2) 汇总 panel 图(一张图看全部)
# ============================================================
print("\n📊 [2/3] Generating summary panel ...")

panels = [
    (M_amp_err,      "Avg amp error",          "lower is better", "min"),
    (M_rel_err,      "Avg relative amp error", "lower is better", "min"),
    (M_cc_mean,      "Correlation (cc)",       "higher is better","max"),
    (M_snr_db,       "SNR_full (dB)",          "higher is better","max"),
    (M_iso_mean,     "Isolation mean (dB)",    "higher is better","max"),
    (M_iso_wc,       "Isolation worst (dB)",   "higher is better","max"),
    (M_iso_mean_all, "Iso mean all-ROI (dB)",  "higher is better","max"),
    (M_iso_wc_all,   "Iso worst all-ROI (dB)", "higher is better","max"),
    (M_target_wl,    "Wavelength-demux ratio", "higher is better","max"),
]

fig, axes = plt.subplots(3, 3, figsize=(16, 12))
for ax, (mat, ylab, hint, best_mode) in zip(axes.flat, panels):
    for li in range(L):
        ax.plot(layers, mat[:, li], marker="o", color=cmap_wl[li], label=wl_labels[li])

    agg = np.nanmean(mat, axis=1)
    idx_best = np.nanargmax(agg) if best_mode == "max" else np.nanargmin(agg)
    ax.axvline(layers[idx_best], color="red", linestyle=":", alpha=0.6)
    ax.set_xticks(layers)
    ax.set_xlabel("layers")
    ax.set_ylabel(ylab)
    ax.set_title(f"{ylab}  ({hint})\nbest @ {layers[idx_best]} layers", fontsize=10)
    ax.legend(fontsize=8, loc="best")

fig.suptitle("All metrics vs. number of layers — summary panel",
             fontsize=14, fontweight="bold", y=1.00)
fig.tight_layout()
out = out_dir / "00_summary_panel.png"
fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
plt.close(fig)
print(f"  ✔ Summary panel -> {out.name}")

# ============================================================
# 3) 归一化对比图(把所有指标放在同一坐标系)
# ============================================================
print("\n📊 [3/3] Generating normalized overlay plot ...")

def _normalize(mat, higher_better=True):
    """归一化到 [0,1],higher_better=False 时取反。"""
    v = np.nanmean(mat, axis=1)  # 跨波长平均
    vmin, vmax = np.nanmin(v), np.nanmax(v)
    if vmax - vmin < 1e-12:
        return np.zeros_like(v)
    n = (v - vmin) / (vmax - vmin)
    return n if higher_better else 1.0 - n

overlay_specs = [
    ("Amp error",         M_amp_err,      False),
    ("Rel amp error",     M_rel_err,      False),
    ("Correlation cc",    M_cc_mean,      True),
    ("SNR (dB)",          M_snr_db,       True),
    ("Iso mean (dB)",     M_iso_mean,     True),
    ("Iso worst (dB)",    M_iso_wc,       True),
    ("Demux ratio",       M_target_wl,    True),
]

fig, ax = plt.subplots(figsize=(10, 5.5))
cmap_metric = plt.cm.tab10(np.linspace(0, 1, len(overlay_specs)))
for (name, mat, hb), color in zip(overlay_specs, cmap_metric):
    n = _normalize(mat, higher_better=hb)
    ax.plot(layers, n, marker="o", label=name, color=color)

ax.set_xlabel("Number of layers")
ax.set_ylabel("Normalized score (1 = best, 0 = worst)")
ax.set_title("All metrics normalized to [0,1] — higher = better\n"
             "(已对 'lower-is-better' 指标取反)")
ax.set_xticks(layers)
ax.legend(loc="best", ncol=2, fontsize=9)
ax.set_ylim(-0.05, 1.05)
fig.tight_layout()
out = out_dir / "00_normalized_overlay.png"
fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
plt.close(fig)
print(f"  ✔ Normalized overlay -> {out.name}")

# ============================================================
# 4) 自动生成一份 markdown 摘要
# ============================================================
def best_layer(mat, mode="max"):
    agg = np.nanmean(mat, axis=1)
    idx = np.nanargmax(agg) if mode == "max" else np.nanargmin(agg)
    return int(layers[idx]), float(agg[idx])

summary_md = out_dir / "summary.md"
with open(summary_md, "w", encoding="utf-8") as f:
    f.write(f"# Metrics Summary\n\n")
    f.write(f"- MAT file: `{mat_path}`\n")
    f.write(f"- Layers tested: {layers.tolist()}\n")
    f.write(f"- Wavelengths : {wavelengths_nm.tolist()} nm\n\n")
    f.write(f"## Best layer count per metric (cross-λ averaged)\n\n")
    f.write(f"| Metric | Best layer | Value |\n|---|---|---|\n")
    for name, mat, mode in [
        ("Amp error",        M_amp_err,      "min"),
        ("Rel amp error",    M_rel_err,      "min"),
        ("Correlation (cc)", M_cc_mean,      "max"),
        ("SNR (dB)",         M_snr_db,       "max"),
        ("Iso mean (dB)",    M_iso_mean,     "max"),
        ("Iso worst (dB)",   M_iso_wc,       "max"),
        ("Iso mean allROI",  M_iso_mean_all, "max"),
        ("Iso worst allROI", M_iso_wc_all,   "max"),
        ("Demux ratio",      M_target_wl,    "max"),
    ]:
        bl, bv = best_layer(mat, mode)
        f.write(f"| {name} | **{bl}** | {bv:.4f} |\n")

print(f"\n📝 Summary -> {summary_md}")
print(f"\n✅ All plots saved to: {out_dir}\n")
