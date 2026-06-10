import math, os, random, time
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import savemat
from torch.utils.data import DataLoader

from odnn_io import load_complex_modes_from_mat
from ODNN_functions import generate_complex_weights, generate_fields_ts
from odnn_wavelength_analysis import ModelGeometry
from odnn_model_factory import build_d2nn
from odnn_train_unified import train_d2nn_unified
from odnn_label_builder import build_labels_and_regions, build_train_dataset
from odnn_training_eval import evaluate_spot_metrics
from odnn_eval_multiwl import (
    evaluate_spot_metrics_multiwl,
    evaluate_target_wl_over_all_wl_roi_ratio,
)
from odnn_training_visualization import capture_eigenmode_propagation, build_uniform_fractions
from odnn_training_eval import save_prediction_diagnostics



# ============================================================
# Reproducibility / device
# ============================================================
SEED = 424242
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Using Device:", device)

# ============================================================
# ★ Wavelength configuration — single or multi-wavelength
# ============================================================
wavelengths_list = (654e-9,)                                      # SINGLE
# wavelengths_list = (1530e-9, 1540e-9, 1550e-9, 1560e-9)             # MULTI

# ============================================================
# Geometry / training params
# ============================================================
field_size = 176
layer_size = 300
out_size   = 500
num_modes  = 6
batch_size = 16
num_layer_option = [1, 2, 3]

z_layers         = 45.744e-3
z_prop           = 130e-3
z_input_to_first = z_layers
pixel_size       = 12.5e-6
padding_ratio    = 0.5
padding_ratio_out = 0.5

phase_option         = 4
label_pattern_mode   = "eigenmode"
eigenmode_focus_radius = 30
eigenmode_detectsize   = 35
circle_focus_radius    = 20
circle_detectsize      = 40

epochs           = 1000
lr_single        = 1.99
lr_multi         = 1e-3

perturb_cfg = dict(
    z_sigma             = 0.0,
    mask_shift_sigma_px = 0.0,
    input_tilt_sigma    = 0.0,
    input_scale_sigma   = 0.0,
)

GEOM = ModelGeometry(
    layer_size          = layer_size,
    z_layers            = z_layers,
    z_prop              = z_prop,
    pixel_size          = pixel_size,
    wavelength          = wavelengths_list[0],
    wavelengths         = wavelengths_list,
    base_wavelength_idx = len(wavelengths_list) // 2,
    z_input_to_first    = z_input_to_first,
    padding_ratio       = padding_ratio,
    out_size            = out_size,
    padding_ratio_out   = padding_ratio_out,
)
L = GEOM.L
print(f"[CONFIG] {'Multi' if L > 1 else 'Single'}-wavelength run, L={L}, "
      f"λ={[f'{w*1e9:.1f}nm' for w in GEOM.wavelength_list]}")

# ============================================================
# Output root
# ============================================================
RUN_TS  = datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_TAG = f"{'multi' if L > 1 else 'single'}WL_L{L}_m{num_modes}_ls{layer_size}_{RUN_TS}"
RUN_ROOT = Path("results") / RUN_TAG
RUN_ROOT.mkdir(parents=True, exist_ok=True)
DIR_TRAIN  = RUN_ROOT / "training_analysis";  DIR_TRAIN.mkdir(parents=True, exist_ok=True)
DIR_CKPT   = RUN_ROOT / "checkpoints";        DIR_CKPT.mkdir (parents=True, exist_ok=True)
DIR_MASK   = RUN_ROOT / "phase_masks";        DIR_MASK.mkdir (parents=True, exist_ok=True)
DIR_METRIC = RUN_ROOT / "metrics_analysis";   DIR_METRIC.mkdir(parents=True, exist_ok=True)
print(f"[OUTPUT] {RUN_ROOT.resolve()}")

DIR_PROPAGATION = RUN_ROOT / "propagation_slices"; DIR_PROPAGATION.mkdir(parents=True, exist_ok=True)
DIR_PRED_VIZ    = RUN_ROOT / "prediction_viz";     DIR_PRED_VIZ.mkdir(parents=True, exist_ok=True)

# ============================================================
# Eigenmodes
# ============================================================
eigenmodes = load_complex_modes_from_mat(
    "data/mmf_10modes_GRIN_176_PD1.2.mat", key="modes_field"
)
H_m, W_m = eigenmodes.shape[:2]
if H_m != W_m:
    M = max(H_m, W_m)
    pad = np.zeros((M, M, eigenmodes.shape[2]), dtype=eigenmodes.dtype)
    pad[(M - H_m) // 2:(M - H_m) // 2 + H_m,
        (M - W_m) // 2:(M - W_m) // 2 + W_m, :] = eigenmodes
    eigenmodes = pad
mmf_data_np = eigenmodes[:, :, :num_modes].transpose(2, 0, 1)
amp_norm = (np.abs(mmf_data_np) - np.abs(mmf_data_np).min()) / \
           (np.abs(mmf_data_np).max() - np.abs(mmf_data_np).min() + 1e-12)
mmf_data_np = amp_norm * np.exp(1j * np.angle(mmf_data_np))
MMF_data_ts = torch.from_numpy(mmf_data_np)

if phase_option == 4:
    base_amp = np.eye(num_modes, dtype=np.float32)
    base_ph  = np.eye(num_modes, dtype=np.float32)
else:
    base_amp, base_ph = generate_complex_weights(1000, num_modes, phase_option)

# ============================================================
# Build labels + ROIs + datasets
# ============================================================
lc = build_labels_and_regions(
    geom                   = GEOM,
    num_modes              = num_modes,
    out_size               = out_size,
    label_pattern_mode     = label_pattern_mode,
    eigenmode_detectsize   = eigenmode_detectsize,
    eigenmode_focus_radius = eigenmode_focus_radius,
    circle_focus_radius    = circle_focus_radius,
    circle_detectsize      = circle_detectsize,
)
MMF_Label_data     = lc["MMF_Label_data"]
evaluation_regions = lc["evaluation_regions"]
detectsize         = lc["detectsize"]
focus_radius       = lc["focus_radius"]

train_data = build_train_dataset(
    geom            = GEOM,
    num_modes       = num_modes,
    field_size      = field_size,
    layer_size      = layer_size,
    out_size        = out_size,
    mmf_modes       = MMF_data_ts,
    MMF_Label_data  = MMF_Label_data,
    phase_option    = phase_option,
    base_amplitudes = base_amp,
    base_phases     = base_ph,
)

# Test loader (eigenmode mode = same as train)
if L > 1:
    test_loader = DataLoader(train_data[0], batch_size=batch_size, shuffle=False)
else:
    test_loader = DataLoader(train_data,    batch_size=batch_size, shuffle=False)

# ============================================================
# Train + evaluate per layer count
# ============================================================
all_losses          = []
all_metrics_layers  = []   # single-wl: list of dict; multi-wl: list of dict[wl_idx -> metrics]

for num_layer in num_layer_option:
    print(f"\n{'='*70}\nTraining D2NN ({'multi' if L>1 else 'single'}-WL) "
          f"with {num_layer} layers\n{'='*70}")

    model = build_d2nn(GEOM, num_layers=num_layer, device=device, perturb_cfg=perturb_cfg)
    print(model)

    result = train_d2nn_unified(
        model, train_data,
        geom       = GEOM,
        epochs     = epochs,
        batch_size = batch_size,
        lr         = (lr_multi if L > 1 else lr_single),
        device     = device,
        seed       = SEED,
    )
    losses = result["losses"]
    all_losses.append(losses)
    total_t = result["total_time"]
    print(f"✔ Training done in {total_t:.1f}s ({total_t/60:.2f} min)  "
          f"final loss={result['final_loss']:.4e}")

    # ---------- training curves ----------
    epochs_arr = np.arange(1, len(losses) + 1)
    fig, ax = plt.subplots(); ax.plot(epochs_arr, losses); ax.set_yscale("log")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.grid(True, alpha=0.3)
    ax.set_title(f"Loss curve — {num_layer} layers (L={L})")
    fig.savefig(DIR_TRAIN / f"loss_L{num_layer}_{RUN_TS}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ---------- save phase masks ----------
    pm_dir = DIR_MASK / f"L{num_layer}"; pm_dir.mkdir(parents=True, exist_ok=True)
    masks = [np.mod(layer.phase.detach().cpu().numpy(), 2 * np.pi) for layer in model.layers]
    savemat(str(pm_dir / "phase_masks.mat"), {"phase_masks": np.stack(masks, 0).astype(np.float32)})

    # ---------- checkpoint ----------
    torch.save(
        {"state_dict": model.state_dict(),
         "meta": {
             "num_layers": int(num_layer), "num_modes": int(num_modes),
             "wavelengths": np.asarray(GEOM.wavelength_list, dtype=np.float64),
             "layer_size": int(layer_size), "out_size": int(out_size),
         }},
        DIR_CKPT / f"d2nn_L{num_layer}.pth",
    )

    # ---------- evaluation ----------
    if L == 1:
        # ★ Reconstruct image_test_data for the evaluator (eigenmode test set)
        if phase_option == 4:
            _eval_amp = base_amp[:num_modes]
            _eval_ph  = base_ph[:num_modes]
        else:
            _eval_amp, _eval_ph = base_amp, base_ph

        _cw = (_eval_amp * np.exp(1j * _eval_ph)).astype(np.complex64)
        image_test_data = generate_fields_ts(
            torch.from_numpy(_cw),
            MMF_data_ts,
            _eval_amp.shape[0],
            num_modes,
            field_size,
        ).to(torch.complex64)

        # amplitudes_phases shape: (N, num_modes + (num_modes-1))
        if num_modes > 1:
            amplitudes_phases = np.hstack(
                (_eval_amp, _eval_ph[:, 1:] / (2 * np.pi))
            )
        else:
            amplitudes_phases = _eval_amp

        metrics = evaluate_spot_metrics(
            model, test_loader, evaluation_regions,
            detect_radius      = detectsize,
            device             = device,
            pred_case          = 1,
            num_modes          = num_modes,
            phase_option       = phase_option,
            amplitudes         = _eval_amp,
            amplitudes_phases  = amplitudes_phases,
            phases             = _eval_ph,
            mmf_modes          = MMF_data_ts,
            field_size         = field_size,
            image_test_data    = image_test_data,        # ★ 真实张量，不再 None
        )
        print(f"  iso_dB = {metrics.get('isolation_db_mean', float('nan')):.2f}, "
              f"SNR_dB = {metrics.get('snr_db_full',         float('nan')):.2f}, "
              f"rel_err= {metrics.get('avg_relative_amp_err',float('nan')):.4e}")
        all_metrics_layers.append(metrics)
    else:
        # ---------- Multi-wavelength branch (unchanged) ----------
        from odnn_eval_multiwl import (
            evaluate_spot_metrics_multiwl,
            evaluate_target_wl_over_all_wl_roi_ratio,
            evaluate_snr_isolation_crosstalk_multiwl,
        )

        per_wl_metrics: dict = {}
        for li in range(L):
            test_loader_wl = DataLoader(train_data[li], batch_size=batch_size, shuffle=False)

            m_amp = evaluate_spot_metrics_multiwl(
                model, test_loader_wl,
                device             = device,
                evaluation_regions = evaluation_regions,
                detect_radius      = detectsize // 2,
                wl_idx=li, L=L, num_modes=num_modes,
            )
            m_snr = evaluate_snr_isolation_crosstalk_multiwl(
                model, test_loader_wl,
                device             = device,
                evaluation_regions = evaluation_regions,
                detect_radius      = detectsize // 2,
                wl_idx=li, L=L, num_modes=num_modes,
            )
            per_wl_metrics[li] = {**m_amp, **m_snr}
            print(f"  λ={GEOM.wavelength_list[li]*1e9:.0f}nm  "
                  f"iso(sameWL)={m_snr['isolation_db_mean']:.2f}dB  "
                  f"iso(allROI)={m_snr['isolation_db_mean_allroi']:.2f}dB  "
                  f"SNR={m_snr['snr_db_full']:.2f}dB  "
                  f"rel_err={m_amp['avg_relative_amp_err']:.4e}")

        wl_ratio = evaluate_target_wl_over_all_wl_roi_ratio(
            model, DataLoader(train_data[0], batch_size=batch_size, shuffle=False),
            device             = device,
            evaluation_regions = evaluation_regions,
            detect_radius      = detectsize // 2,
            L=L, num_modes=num_modes,
        )
        per_wl_metrics["target_wl_ratio_per_wl"] = wl_ratio["ratio_per_wl"]
        per_wl_metrics["target_wl_ratio_mean"]   = wl_ratio["ratio_mean"]
        print(f"  TargetWL/AllWL ratio mean = {wl_ratio['ratio_mean']:.4f}, "
              f"per-WL = {[f'{x:.3f}' for x in wl_ratio['ratio_per_wl']]}")

        all_metrics_layers.append(per_wl_metrics)
    # ============================================================
    # ★ Propagation slices (per eigenmode)
    # ============================================================
    timestamp_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    prop_dir_layer = DIR_PROPAGATION / f"L{num_layer}"
    prop_dir_layer.mkdir(parents=True, exist_ok=True)

    # Visualize one representative eigenmode (mode index 2 by default)
    eigenmode_index = min(2, MMF_data_ts.shape[0] - 1)
    layer_fractions = [build_uniform_fractions(3) for _ in range(num_layer)]
    output_fractions = build_uniform_fractions(3)

    if L == 1:
        # ---- Single wavelength ----
        prop_summary = capture_eigenmode_propagation(
            model              = model,
            eigenmode_field    = MMF_data_ts[eigenmode_index],
            mode_index         = eigenmode_index,
            layer_size         = layer_size,
            z_input_to_first   = z_input_to_first,
            z_layers           = z_layers,
            z_prop             = z_prop,
            pixel_size         = pixel_size,
            wavelength         = GEOM.wavelength,
            output_dir         = prop_dir_layer,
            tag                = f"L{num_layer}_{timestamp_tag}",
            fractions_between_layers = layer_fractions,
            output_fractions   = output_fractions,
        )
        print(f"  ✔ propagation slices -> {prop_summary['fig_path']}")
    else:
        # ---- Multi-wavelength: one figure per λ ----
        for wl_idx, wl in enumerate(GEOM.wavelength_list):
            # 用一个临时单波长包装：把 wl_idx 那一通道的输出拿出来可视化
            prop_dir_wl = prop_dir_layer / f"wl{wl_idx}_{wl*1e9:.0f}nm"
            prop_dir_wl.mkdir(parents=True, exist_ok=True)
            try:
                prop_summary = capture_eigenmode_propagation(
                    model              = model,
                    eigenmode_field    = MMF_data_ts[eigenmode_index],
                    mode_index         = eigenmode_index,
                    layer_size         = layer_size,
                    z_input_to_first   = z_input_to_first,
                    z_layers           = z_layers,
                    z_prop             = z_prop,
                    pixel_size         = pixel_size,
                    wavelength         = float(wl),
                    output_dir         = prop_dir_wl,
                    tag                = f"L{num_layer}_wl{wl_idx}_{timestamp_tag}",
                    fractions_between_layers = layer_fractions,
                    output_fractions   = output_fractions,
                    wavelength_idx     = wl_idx,    # ★ 如 capture_... 支持该参数
                )
                print(f"  ✔ propagation slices λ={wl*1e9:.0f}nm -> {prop_summary['fig_path']}")
            except TypeError:
                # capture_eigenmode_propagation 不支持 wavelength_idx 时跳过
                print(f"  ⚠ capture_eigenmode_propagation does not support multi-WL "
                      f"(wavelength_idx). Skipped λ={wl*1e9:.0f}nm.")
                break

        # ============================================================
        # ★ Prediction diagnostics (per-sample plots)
        # ============================================================
        pred_dir_layer = DIR_PRED_VIZ / f"L{num_layer}"
        if L == 1:
            # 单波长：直接喂入 train_data（TensorDataset）
            diag_dataset = train_data
        else:
            # 多波长：取基准波长的 dataset
            base_idx = GEOM.base_wavelength_idx if GEOM.base_wavelength_idx is not None else L // 2
            diag_dataset = train_data[base_idx]

        try:
            diag_paths = save_prediction_diagnostics(
                model            = model,
                test_dataset     = diag_dataset,
                evaluation_regions = evaluation_regions if L == 1 else
                                    [evaluation_regions[mk * L + base_idx] for mk in range(num_modes)],
                layer_size       = GEOM.out_size,
                detect_radius    = detectsize,
                num_samples      = min(3, num_modes),
                output_dir       = pred_dir_layer,
                device           = device,
                tag              = f"L{num_layer}_{timestamp_tag}",
            )
            if diag_paths:
                print(f"  ✔ prediction viz ({len(diag_paths)} samples) -> {diag_paths[0].parent}")
            else:
                print("  ⚠ no prediction diagnostics produced (empty dataset?)")
        except Exception as e:
            print(f"  ⚠ prediction_viz failed: {type(e).__name__}: {e}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n" + "=" * 70 + "\nAll training done.\n" + "=" * 70)

# ============================================================
# Save metrics .mat (auto-tag for plot_from_mat.py)
# ============================================================
layer_counts = np.asarray(num_layer_option, dtype=np.int32)
NL = len(all_metrics_layers)

if L == 1:
    iso_mean = np.array([m.get("isolation_db_mean", np.nan) for m in all_metrics_layers])
    snr_db   = np.array([m.get("snr_db_full",       np.nan) for m in all_metrics_layers])
    payload = {
        "layers":           layer_counts.astype(np.float64),
        "isolation_db_mean": iso_mean,
        "snr_db_full":       snr_db,
    }
else:
    iso_mean   = np.full((NL, L), np.nan)
    iso_allroi = np.full((NL, L), np.nan)
    snr_db     = np.full((NL, L), np.nan)
    for i, mlist in enumerate(all_metrics_layers):
        for li, m in mlist.items():
            iso_mean[i, li]   = m.get("isolation_db_mean",        np.nan)
            iso_allroi[i, li] = m.get("isolation_db_mean_allroi", np.nan)
            snr_db[i, li]     = m.get("snr_db_full",              np.nan)
    payload = {
        "layers":                   layer_counts.astype(np.float64),
        "wavelengths_m":            np.asarray(GEOM.wavelength_list, dtype=np.float64),
        "wavelengths_nm":           np.asarray(GEOM.wavelength_list, dtype=np.float64) * 1e9,
        "isolation_db_mean":        iso_mean,
        "isolation_db_mean_allroi": iso_allroi,
        "snr_db_full":              snr_db,
    }

mat_out = DIR_METRIC / f"metrics_vs_layers_{RUN_TS}.mat"
savemat(str(mat_out), payload)
print(f"✔ Metrics .mat -> {mat_out}")
print(f"\n✅ All outputs saved under {RUN_ROOT}")
