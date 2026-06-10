"""
Unified factory: single- or multi-wavelength D2NN model.
"""
from __future__ import annotations
import inspect
from typing import Optional

import numpy as np
import torch

from odnn_model import D2NNModel
from odnn_multiwl_model import D2NNModelMultiWL
from odnn_wavelength_analysis import ModelGeometry


def _apply_z_jitter(z_value: float, sigma: float) -> float:
    """Add Gaussian jitter to a z-distance if sigma > 0."""
    if sigma > 0:
        return float(z_value + np.random.randn() * sigma)
    return float(z_value)


def _build_single_wl(
    geom: ModelGeometry,
    *,
    num_layers: int,
    device: torch.device,
    perturb_cfg: Optional[dict] = None,
) -> D2NNModel:
    """Build single-wavelength D2NNModel directly via __init__.

    perturb_cfg keys handled here:
        z_sigma                : Gaussian σ on z distances (m)
        mask_shift_sigma_px    : (forwarded if D2NNModel accepts it)
        input_tilt_sigma       : (forwarded if D2NNModel accepts it)
        input_scale_sigma      : (forwarded if D2NNModel accepts it)
    """
    pcfg = perturb_cfg or {}
    z_sigma = float(pcfg.get("z_sigma", 0.0))

    # Apply z-jitter if requested
    z_input = _apply_z_jitter(geom.z_input_to_first, z_sigma)
    z_lay   = _apply_z_jitter(geom.z_layers,         z_sigma)
    z_pp    = _apply_z_jitter(geom.z_prop,           z_sigma)

    # Probe D2NNModel signature so we only pass parameters it actually supports
    init_params = inspect.signature(D2NNModel.__init__).parameters

    kwargs = dict(
        num_layers       = num_layers,
        layer_size       = geom.layer_size,
        z_layers         = z_lay,
        z_prop           = z_pp,
        pixel_size       = geom.pixel_size,
        wavelength       = geom.wavelength,
        device           = device,
        padding_ratio    = geom.padding_ratio,
        z_input_to_first = z_input,
    )
    # Optional kwargs only if the model declares them
    if "out_size"          in init_params: kwargs["out_size"]          = geom.out_size
    if "padding_ratio_out" in init_params: kwargs["padding_ratio_out"] = geom.padding_ratio_out
    if "perturb_cfg"       in init_params: kwargs["perturb_cfg"]       = perturb_cfg

    model = D2NNModel(**kwargs).to(device)

    # If D2NNModel doesn't accept perturb_cfg natively, store it as attribute
    # so other code (eval / analysis) can still inspect it.
    if "perturb_cfg" not in init_params and perturb_cfg:
        model.perturb_cfg = dict(perturb_cfg)

    return model


def build_d2nn(
    geom: ModelGeometry,
    *,
    num_layers: int,
    device: torch.device,
    perturb_cfg: Optional[dict] = None,
):
    """Single source of truth — L=1 -> D2NNModel, L>=2 -> D2NNModelMultiWL."""
    # ---------------- Multi-wavelength ----------------
    if geom.is_multiwavelength:
        base_idx = (geom.base_wavelength_idx
                    if geom.base_wavelength_idx is not None
                    else len(geom.wavelength_list) // 2)
        return D2NNModelMultiWL(
            num_layers          = num_layers,
            layer_size          = geom.layer_size,
            z_layers            = geom.z_layers,
            z_prop              = geom.z_prop,
            pixel_size          = geom.pixel_size,
            wavelengths         = list(geom.wavelength_list),
            device              = device,
            padding_ratio       = geom.padding_ratio,
            z_input_to_first    = geom.z_input_to_first,
            base_wavelength_idx = base_idx,
            perturb_cfg         = perturb_cfg,
            out_size            = geom.out_size,
            padding_ratio_out   = geom.padding_ratio_out,
        ).to(device)

    # ---------------- Single-wavelength ----------------
    return _build_single_wl(
        geom,
        num_layers  = num_layers,
        device      = device,
        perturb_cfg = perturb_cfg,
    )
