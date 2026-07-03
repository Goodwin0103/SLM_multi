"""批量多波长训练 + 自动画图（单脚本）

对一组 mode 数量（默认 6,10,20,30,40,50）顺序调用 mainfor6_wl.py 训练，
每次内部自动循环 layers 1-5；训练完读取每层四个指标，最后画四张折线图：
  - Mode Isolation (dB)      vs Layers
  - Target/All ROI (ratio)   vs Layers
  - SNR (dB)                 vs Layers
  - Throughput               vs Layers
每张图横轴是层数，每个 mode 数量画一条彩色折线。

用法（在服务器上 conda activate odnn 之后）：
  前台：  python batch_train_wl.py
  后台：  nohup python batch_train_wl.py > batch.log 2>&1 &
可选覆盖： python batch_train_wl.py --config xx.json --output_dir yy
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent

# ============================================================
# 配置：用户只需改这一段
# ============================================================
# 注意：MAT_FILE 必须包含 >= max(MODE_COUNTS) 个模式；
#       field_size 必须匹配该 .mat 模式场的空间尺寸（文件名里的数字，如 _100 / _176）。
MAT_FILE = "mmf_103modes_100_PD_1.15.mat"      # 模式 .mat 路径（>=50 modes）
MODE_COUNTS = [6, 10, 20, 30, 40, 50]           # 要扫的 mode 数量（每个一条线）
OUTPUT_DIR = "results/batch_sweep"              # 所有输出的根目录

# 固定的基础训练参数（键名与 mainfor6_wl.py 的 --config 一致）
BASE_CONFIG: Dict = {
    "num_layers_list": [1, 2, 3, 4, 5],   # 每次训练内部循环的层数
    "field_size": 100,                    # 须匹配 MAT_FILE 的空间尺寸
    "layer_size": 110,
    "out_size": 600,
    "padding_ratio_out": 0.5,
    "padding_ratio": 0.5,
    "pixel_size_um": 1.0,
    "wl_start_nm": 1550,
    "wl_spacing_nm": 0.5,
    "wl_count": 1,                        # 单波长
    "base_wavelength_idx": 0,
    "z_layers_um": 45,
    "z_prop_um": 200000,
    "z_input_to_first_um": 0,
    "epochs": 1000,
    "lr": 1.99,
    "batch_size": 16,
    "circle_focus_radius": 5,
    "margin_ratio": 0.2,
    "phase_option": 4,
    "evaluation_mode": "eigenmode",
    "training_dataset_mode": "eigenmode",
    "label_pattern_mode": "circle",
}

# 四个要画的指标：(summary jsonl 里的键, 文件名, 纵轴标题)
_METRICS = [
    ("mode_isolation_db_mean",    "mode_isolation_vs_layers.png", "Mode Isolation (dB)"),
    ("target_all_roi_ratio_mean", "target_all_roi_vs_layers.png", "Target/All ROI (ratio)"),
    ("snr_db_mean",               "snr_vs_layers.png",            "SNR (dB)"),
    ("throughput_mean",           "throughput_vs_layers.png",     "Throughput"),
]


def pick_best_gpu() -> Optional[int]:
    """返回剩余显存最大的 GPU 编号；没有 GPU 则返回 None（用 CPU）。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        )
        if out.returncode != 0:
            return None
        best_idx, best_free = None, -1
        for line in out.stdout.strip().splitlines():
            idx_s, free_s = line.split(",")
            idx, free = int(idx_s), int(free_s)
            if free > best_free:
                best_idx, best_free = idx, free
        return best_idx
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        return None


def run_one(num_modes: int, base_cfg: Dict, mat_file: str, run_dir: Path) -> bool:
    """对单个 mode 数量跑一次完整训练（内部循环 layers）。成功返回 True。"""
    run_dir.mkdir(parents=True, exist_ok=True)

    # 清掉可能残留的旧汇总文件，保证 mainfor6_wl.py 的追加从空开始
    summary_file = run_dir / "logs" / "summary_metrics_wl.jsonl"
    if summary_file.exists():
        summary_file.unlink()

    cfg = {k: v for k, v in base_cfg.items()}
    cfg["num_modes"] = int(num_modes)
    cfg_path = run_dir / "train_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))

    gpu = pick_best_gpu()
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    gpu_label = f"GPU {gpu}" if gpu is not None else "CPU"

    print(f"  -> launching mainfor6_wl.py (num_modes={num_modes}) on {gpu_label}", flush=True)
    proc = subprocess.run(
        [sys.executable, "mainfor6_wl.py",
         "--config", str(cfg_path),
         "--mat_file", mat_file,
         "--output_dir", str(run_dir)],
        env=env, cwd=str(PROJECT_ROOT),
    )
    if proc.returncode != 0:
        print(f"  !! training for num_modes={num_modes} exited with code {proc.returncode}",
              flush=True)
        return False
    return True


def collect_rows(run_dir: Path) -> List[Dict]:
    """读取某次训练的每层汇总（每层一行 JSON）。"""
    summary_file = run_dir / "logs" / "summary_metrics_wl.jsonl"
    if not summary_file.exists():
        return []
    rows: List[Dict] = []
    for line in summary_file.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def plot_all(summary_path: Path, out_dir: Path) -> None:
    """按 num_modes 分组，对四个指标各画一张 Layers-vs-metric 折线图。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict] = []
    for line in summary_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    # 按 mode 数量分组：{num_modes: {num_layers: row}}
    by_modes: Dict[int, Dict[int, Dict]] = {}
    for r in rows:
        if "num_modes" not in r or "num_layers" not in r:
            continue
        by_modes.setdefault(int(r["num_modes"]), {})[int(r["num_layers"])] = r
    if not by_modes:
        print("!! no valid data to plot", flush=True)
        return

    sorted_modes = sorted(by_modes.keys())
    colors = plt.cm.tab10.colors
    markers = ["o", "s", "^", "D", "v", "P", "*", "X"]
    linestyles = ["-", "--", "-.", ":"]

    for key, fname, ylabel in _METRICS:
        fig, ax = plt.subplots(figsize=(7.5, 5.0))
        for i, M in enumerate(sorted_modes):
            layer_map = by_modes[M]
            xs = sorted(layer_map.keys())
            ys = [layer_map[x].get(key, float("nan")) for x in xs]
            ax.plot(
                xs, ys,
                color=colors[i % len(colors)],
                marker=markers[i % len(markers)],
                linestyle=linestyles[i % len(linestyles)],
                linewidth=2, markersize=7,
                label=f"M={M}",
            )
        ax.set_xlabel("Number of Layers", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} vs Layers", fontsize=13)
        # 横轴只显示整数层数
        all_layers = sorted({x for m in by_modes.values() for x in m.keys()})
        ax.set_xticks(all_layers)
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.legend(title="Modes", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {out_dir / fname}", flush=True)


def run_batch(mat_file: str, mode_counts: List[int], base_cfg: Dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "batch_summary.jsonl"
    summary_path.write_text("")   # 清空，重新开始

    total = len(mode_counts)
    for i, M in enumerate(mode_counts):
        print(f"\n{'=' * 70}\n[{i + 1}/{total}] num_modes = {M}\n{'=' * 70}", flush=True)
        run_dir = output_dir / f"modes_{M}"
        ok = run_one(M, base_cfg, mat_file, run_dir)
        if not ok:
            with open(summary_path, "a") as f:
                f.write(json.dumps({"num_modes": int(M), "status": "error"}) + "\n")
            print(f"[{i + 1}/{total}] num_modes={M} FAILED, continue.", flush=True)
            continue
        rows = collect_rows(run_dir)
        if not rows:
            print(f"  !! no summary_metrics_wl.jsonl rows for num_modes={M}", flush=True)
        with open(summary_path, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[{i + 1}/{total}] num_modes={M} done ({len(rows)} layers collected).", flush=True)

    print(f"\n{'=' * 70}\nAll trainings finished. Plotting...\n{'=' * 70}", flush=True)
    plot_all(summary_path, output_dir / "plots")
    print("\nDone. Plots are in:", output_dir / "plots", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Batch multi-wavelength training + auto plot")
    ap.add_argument("--config", type=str, default=None,
                    help="可选 JSON，覆盖顶部配置（可含 mat_file / mode_counts / 其它训练参数）")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="可选，覆盖输出根目录")
    args = ap.parse_args()

    mat_file = MAT_FILE
    mode_counts = list(MODE_COUNTS)
    base_cfg = dict(BASE_CONFIG)
    output_dir = Path(OUTPUT_DIR)

    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        mat_file = cfg.pop("mat_file", mat_file)
        if "mode_counts" in cfg:
            mode_counts = [int(x) for x in cfg.pop("mode_counts")]
        cfg.pop("output_dir", None)
        base_cfg.update(cfg)   # 其余键覆盖训练参数
    if args.output_dir:
        output_dir = Path(args.output_dir)

    print("MAT_FILE   :", mat_file, flush=True)
    print("MODE_COUNTS:", mode_counts, flush=True)
    print("LAYERS     :", base_cfg.get("num_layers_list"), flush=True)
    print("OUTPUT_DIR :", output_dir, flush=True)

    run_batch(mat_file, mode_counts, base_cfg, output_dir)


if __name__ == "__main__":
    main()
