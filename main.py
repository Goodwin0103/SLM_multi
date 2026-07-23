"""
WL Isolation analysis — INTERACTIVE manual ROI calibration.
需要 GUI 环境才能弹出选点窗口。
"""

# ============================================================
# ★ 强制使用 GUI 后端（必须在 pyplot 之前）
# ============================================================
import matplotlib
_GUI_OK = False
for backend in ["TkAgg", "Qt5Agg", "QtAgg", "MacOSX"]:
    try:
        matplotlib.use(backend, force=True)
        print(f"✔ Using interactive backend: {backend}")
        _GUI_OK = True
        break
    except Exception:
        continue
if not _GUI_OK:
    raise RuntimeError(
        "❌ 没有可用的 GUI 后端！请安装 PyQt5:  pip install PyQt5\n"
        "   或者在本地电脑(非远程服务器)上运行。"
    )

import matplotlib.pyplot as plt
print(f"   Active backend: {matplotlib.get_backend()}")

import numpy as np
from PIL import Image
from pathlib import Path
import json

# ============================================================
# 配置
# ============================================================
IMG_PATHS = {
    (654, 1): "654_1_layer.png",
    (654, 2): "654_2_layer.png",
    (852, 1): "852_1_layer.png",
    (852, 2): "852_2_layer.png",
}
N_MODES_PER_WL = {654: 6, 852: 3}
INITIAL_RADIUS = {654: 14, 852: 26}
CALIB_FILE = Path("wl_iso_manual_rois.json")

# ============================================================
# 交互式选点工具
# ============================================================
class InteractiveROIPicker:
    """
    操作:
      L-click  : 添加 ROI
      R-click  : 删除最近的 ROI
      'g'      : 切换 GREEN(target) <-> RED(leak) 模式
      '+'/'-'  : 调当前模式下的圈半径
      Enter    : 完成本张图,关闭窗口
      'q'      : 取消并退出全部
    """
    def __init__(self, img, wl, n_layer, init_radius_t, init_radius_l):
        self.img = img
        self.wl = wl
        self.n_layer = n_layer
        self.r_t = init_radius_t      # 绿圈半径 (target)
        self.r_l = init_radius_l      # 红圈半径 (leak)
        self.targets = []             # [(cy, cx)]
        self.leaks = []
        self.mode = "target"
        self.cancelled = False

        self.fig, self.ax = plt.subplots(figsize=(11, 8))
        vmax = np.percentile(img, 99.5)
        self.ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax)
        self.fig.canvas.mpl_connect("button_press_event", self.on_click)
        self.fig.canvas.mpl_connect("key_press_event", self.on_key)
        self._redraw()

    def _redraw(self):
        for p in list(self.ax.patches):
            p.remove()
        for (cy, cx) in self.targets:
            self.ax.add_patch(plt.Circle((cx, cy), self.r_t, fill=False,
                                          edgecolor="lime", linewidth=2.5))
        for (cy, cx) in self.leaks:
            self.ax.add_patch(plt.Circle((cx, cy), self.r_l, fill=False,
                                          edgecolor="red", linewidth=2, linestyle="--"))
        mode_color = "lime" if self.mode == "target" else "red"
        mode_label = "GREEN (target)" if self.mode == "target" else "RED (leakage)"
        nm = N_MODES_PER_WL[self.wl]
        self.ax.set_title(
            f"λ={self.wl}nm, L={self.n_layer}  |  ~{nm} modes expected\n"
            f"Current click mode: {mode_label}   "
            f"[{len(self.targets)} green / {len(self.leaks)} red]\n"
            f"L-click=add  R-click=del  'g'=toggle  +/-=radius  "
            f"Enter=done  q=cancel",
            fontsize=11, color=mode_color, fontweight="bold")
        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None:
            return
        cx, cy = int(round(event.xdata)), int(round(event.ydata))
        target_list = self.targets if self.mode == "target" else self.leaks
        if event.button == 1:         # 左键 add
            target_list.append((cy, cx))
        elif event.button == 3:       # 右键 delete nearest
            if target_list:
                d = [(p[0]-cy)**2 + (p[1]-cx)**2 for p in target_list]
                target_list.pop(int(np.argmin(d)))
        self._redraw()

    def on_key(self, event):
        if event.key == "g":
            self.mode = "leak" if self.mode == "target" else "target"
            self._redraw()
        elif event.key in ("+", "="):
            if self.mode == "target": self.r_t += 2
            else: self.r_l += 2
            self._redraw()
        elif event.key in ("-", "_"):
            if self.mode == "target": self.r_t = max(3, self.r_t - 2)
            else: self.r_l = max(3, self.r_l - 2)
            self._redraw()
        elif event.key == "enter":
            plt.close(self.fig)
        elif event.key == "q":
            self.cancelled = True
            plt.close(self.fig)

    def run(self):
        plt.show(block=True)   # 关键: block=True 才会等用户操作
        return {
            "targets": self.targets,
            "leaks": self.leaks,
            "radius_target": self.r_t,
            "radius_leak": self.r_l,
            "cancelled": self.cancelled,
        }


# ============================================================
# 加载 4 张图
# ============================================================
imgs = {k: np.array(Image.open(v).convert("L"), dtype=np.float32)
        for k, v in IMG_PATHS.items()}
H, W = next(iter(imgs.values())).shape
print(f"\nImage size: {H}×{W}")

# ============================================================
# 逐图交互式标定
# ============================================================
print("\n" + "=" * 60)
print(" 交互式 ROI 标定")
print("=" * 60)
print("""
操作:
  - 左键: 在亮斑上添加 ROI
  - 右键: 删除最近的 ROI
  - 'g' : 切换 GREEN(target本波长目标位置) / RED(leak另一波长泄漏位置)
  - '+' / '-': 调当前圈的半径
  - Enter: 完成本张图,进入下一张
  - 'q' : 取消并退出全部

含义:
  🟢 GREEN  = 本波长应该聚焦的位置 (用于计算 E_target)
  🔴 RED    = 另一波长应该出现的位置,这里有信号 = 泄漏 (用于计算 E_leak)

WL Isolation (dB) = 10·log10( E_target / E_leak )
越大表示该波长的信号越干净,没漏到另一波长去。
""")
input("准备好了按 Enter 开始标定...")

# 从最干净的开始标(最容易看清亮斑)
order = [(852, 2), (852, 1), (654, 2), (654, 1)]
calibration = {}

for (wl, nl) in order:
    img = imgs[(wl, nl)]
    print(f"\n>>> 当前: λ={wl} nm, L={nl} ({N_MODES_PER_WL[wl]} modes)")
    other_wl = 852 if wl == 654 else 654
    picker = InteractiveROIPicker(
        img, wl, nl,
        init_radius_t=INITIAL_RADIUS[wl],
        init_radius_l=INITIAL_RADIUS[other_wl],
    )
    result = picker.run()
    if result["cancelled"]:
        print("❌ 已取消")
        exit(0)
    calibration[(wl, nl)] = result
    print(f"   ✓ green={len(result['targets'])}, red={len(result['leaks'])}, "
          f"r_t={result['radius_target']}, r_l={result['radius_leak']}")

# ============================================================
# 保存标定结果
# ============================================================
calib_dump = {
    f"{wl}_{nl}": {
        "targets": v["targets"],
        "leaks": v["leaks"],
        "radius_target": v["radius_target"],
        "radius_leak": v["radius_leak"],
    } for (wl, nl), v in calibration.items()
}
CALIB_FILE.write_text(json.dumps(calib_dump, indent=2))
print(f"\n✔ 标定结果已保存 -> {CALIB_FILE}")

# ============================================================
# 计算 WL Isolation
# ============================================================
def roi_sum(img, cy, cx, r):
    yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
    mask = (yy - cy)**2 + (xx - cx)**2 <= r**2
    return float(img[mask].sum()), int(mask.sum())

def total_energy(img, rois, r):
    E, N = 0.0, 0
    for (cy, cx) in rois:
        e, n = roi_sum(img, cy, cx, r); E += e; N += n
    return E, N

# 切回 Agg 出图,避免最终结果图也弹窗
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt   # 重新 import 让后端生效

results = []
print(f"\n{'L':<4}{'λ(nm)':<8}{'#m':<5}{'#tgt':<6}{'#lk':<6}"
      f"{'E_tgt':<12}{'E_leak':<12}"
      f"{'E_t/(px·m·N)':<15}{'E_o/(px·m·N)':<15}{'WL Iso (dB)':<13}")
print("-" * 100)
for nl in (1, 2):
    for wl in (654, 852):
        img = imgs[(wl, nl)]
        nm = N_MODES_PER_WL[wl]
        nm_o = N_MODES_PER_WL[852 if wl == 654 else 654]
        c = calibration[(wl, nl)]
        E_t, n_t = total_energy(img, c["targets"], c["radius_target"])
        E_l, n_l = total_energy(img, c["leaks"],   c["radius_leak"])
        Nt, Nl = max(len(c["targets"]), 1), max(len(c["leaks"]), 1)
        Et_n = E_t / max(n_t * nm   * Nt, 1)
        El_n = E_l / max(n_l * nm_o * Nl, 1)
        wl_iso = 10 * np.log10(max(Et_n, 1e-12) / max(El_n, 1e-12))
        results.append({"L": nl, "wl": wl, "nm": nm, "Nt": Nt, "Nl": Nl,
                        "E_t": E_t, "E_l": E_l, "wl_iso": wl_iso})
        print(f"{nl:<4}{wl:<8}{nm:<5}{Nt:<6}{Nl:<6}"
              f"{E_t:<12.3e}{E_l:<12.3e}"
              f"{Et_n:<15.5f}{El_n:<15.5f}{wl_iso:<+13.2f}")

print(f"\n=== Mean WL Iso per layer ===")
for nl in (1, 2):
    arr = [r["wl_iso"] for r in results if r["L"] == nl]
    s654 = next(r for r in results if r["L"]==nl and r["wl"]==654)["wl_iso"]
    s852 = next(r for r in results if r["L"]==nl and r["wl"]==852)["wl_iso"]
    print(f"  L={nl}: mean={np.mean(arr):+.2f} dB | 654={s654:+.2f} dB  852={s852:+.2f} dB")

# ============================================================
# 出最终图
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
pos = {(654, 1): axes[0,0], (852, 1): axes[0,1],
       (654, 2): axes[1,0], (852, 2): axes[1,1]}
for (wl, nl), img in imgs.items():
    ax = pos[(wl, nl)]
    vmax = np.percentile(img, 99.5)
    ax.imshow(img, cmap="inferno", vmin=0, vmax=vmax)
    c = calibration[(wl, nl)]
    for i, (cy, cx) in enumerate(c["targets"]):
        ax.add_patch(plt.Circle((cx, cy), c["radius_target"], fill=False,
                                edgecolor="lime", linewidth=2.5))
        ax.text(cx, cy, f"{i+1}", ha="center", va="center",
                color="lime", fontsize=9, fontweight="bold")
    for (cy, cx) in c["leaks"]:
        ax.add_patch(plt.Circle((cx, cy), c["radius_leak"], fill=False,
                                edgecolor="red", linewidth=2, linestyle="--"))
    rec = next(r for r in results if r["L"]==nl and r["wl"]==wl)
    ax.set_title(f"λ={wl} nm, L={nl} ({rec['nm']} modes)\n"
                 f"green={rec['Nt']}, red={rec['Nl']}  |  "
                 f"WL Iso = {rec['wl_iso']:+.2f} dB",
                 fontsize=11, fontweight="bold")
    ax.axis("off")
fig.suptitle("WL Isolation — Manual ROI Calibration\n"
             "🟢 GREEN = target | 🔴 RED = leakage",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig("wl_isolation_manual.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# 柱状图
fig, ax = plt.subplots(figsize=(8, 5))
layers = [1, 2]; wls = [654, 852]; width = 0.35
x = np.arange(len(layers))
for i, wl in enumerate(wls):
    ys = [next(r for r in results if r["L"]==l and r["wl"]==wl)["wl_iso"] for l in layers]
    color = "tab:blue" if wl == 654 else "tab:orange"
    bars = ax.bar(x + (i-0.5)*width, ys, width,
                  label=f"λ={wl} nm ({N_MODES_PER_WL[wl]} modes)", color=color)
    for b, y in zip(bars, ys):
        ax.text(b.get_x()+b.get_width()/2,
                y + (0.4 if y >= 0 else -1.0),
                f"{y:+.1f}", ha="center", fontsize=10, fontweight="bold")
mean_per_layer = [np.mean([r["wl_iso"] for r in results if r["L"]==l]) for l in layers]
ax.plot(x, mean_per_layer, "k--D", label="mean", linewidth=2, markersize=10)
for xi, my in zip(x, mean_per_layer):
    ax.text(xi, my + 0.4, f"{my:+.1f}", ha="center", color="k", fontsize=10)
ax.set_xticks(x); ax.set_xticklabels([f"L={l}" for l in layers], fontsize=12)
ax.set_ylabel("WL Isolation (dB)", fontsize=12)
ax.set_title("Manually Calibrated WL Iso vs Layer Count", fontsize=12, fontweight="bold")
ax.axhline(0, color="k", linewidth=0.5)
ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=10)
fig.tight_layout()
fig.savefig("wl_iso_manual_bar.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("\n✔ Saved: wl_isolation_manual.png, wl_iso_manual_bar.png")
