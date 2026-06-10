"""
D2NN single-wavelength model.

Geometry:
    Input -> pre_propagation (layer_size grid)
          -> N x DiffractionLayer (layer_size grid)
          -> embed to out_size canvas
          -> propagation (out_size grid)
          -> RegressionDetector (|·|^2)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def propagation(E, z, lam, layer_size, pixel_size, device):
    """
    Legacy free-space propagation (kept for backward compatibility with
    odnn_training_visualization.py and other older modules).

    Note: prefer using the Propagation class for new code.
    """
    E = E.clone().detach().to(dtype=torch.complex64, device=device)

    fft_c = torch.fft.fft2(E)
    c = torch.fft.fftshift(fft_c)

    fx = torch.fft.fftshift(torch.fft.fftfreq(layer_size, d=pixel_size)).to(device)
    fxx, fyy = torch.meshgrid(fx, fx, indexing="ij")

    argument = (2 * torch.pi) ** 2 * ((1.0 / lam) ** 2 - fxx ** 2 - fyy ** 2)
    tmp = torch.sqrt(torch.abs(argument))
    kz = torch.where(argument >= 0, tmp, 1j * tmp)

    return torch.fft.ifft2(torch.fft.ifftshift(c * torch.exp(1j * kz * z)))

# ============================================================
# Complex padding helpers
# ============================================================
def complex_pad(E, pad_h, pad_w):
    """Zero-pad a complex tensor on the last 2 dims by (pad_h, pad_w) on each side."""
    Er = torch.view_as_real(E)                                       # (..., H, W, 2)
    Er_pad = F.pad(Er, (0, 0, pad_w, pad_w, pad_h, pad_h),
                   mode="constant", value=0)
    return torch.view_as_complex(Er_pad.contiguous())


def complex_crop(E_pad, H, W, pad_h, pad_w):
    """Center-crop a complex tensor back from a (H+2*pad_h, W+2*pad_w) canvas."""
    return E_pad[..., pad_h:pad_h + H, pad_w:pad_w + W].contiguous()

def complex_pad_asymm(E, pad_top, pad_bottom, pad_left, pad_right):
    """Asymmetric zero-pad on last 2 dims for a complex tensor."""
    Er = torch.view_as_real(E)                                          # (..., H, W, 2)
    Er_pad = F.pad(
        Er,
        (0, 0, pad_left, pad_right, pad_top, pad_bottom),
        mode="constant", value=0,
    )
    return torch.view_as_complex(Er_pad.contiguous())


def make_pad_slices(H, W, padding_ratio=None, pad_px=None):
    """Return (pad_h, pad_w, (slice_h, slice_w)) for centering after padding."""
    if pad_px is None:
        pad_h = int(round(H * padding_ratio))
        pad_w = int(round(W * padding_ratio))
    else:
        pad_h = pad_w = int(pad_px)
    sl_h = slice(pad_h, pad_h + H)
    sl_w = slice(pad_w, pad_w + W)
    return pad_h, pad_w, (sl_h, sl_w)

# ============================================================
# Free-space propagation (angular spectrum)
# ============================================================
class Propagation(nn.Module):
    """Free-space propagation by distance z, with optional zero-padding."""

    def __init__(self, units, dx, lam, z, device, pad_px=0):
        super().__init__()
        self.units  = int(units)
        self.dx     = float(dx)
        self.lam    = float(lam)
        self.z      = float(z)
        self.pad_px = int(pad_px)

        self.register_buffer("kz_base", self._make_kz(self.units, self.dx, self.lam, device))
        if self.pad_px > 0:
            units_pad = self.units + 2 * self.pad_px
            self.register_buffer("kz_pad", self._make_kz(units_pad, self.dx, self.lam, device))
        else:
            self.kz_pad = None

    @staticmethod
    def _make_kz(N, dx, lam, device):
        fx = torch.fft.fftshift(torch.fft.fftfreq(N, d=dx)).to(device)
        fxx, fyy = torch.meshgrid(fx, fx, indexing="ij")
        argument = (2 * torch.pi) ** 2 * ((1.0 / lam) ** 2 - fxx ** 2 - fyy ** 2)
        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j * tmp).to(torch.complex64)
        return kz

    @staticmethod
    def _propagate(E, kz, z):
        E = E.to(torch.complex64)
        C = torch.fft.fftshift(torch.fft.fft2(E), dim=(-2, -1))
        return torch.fft.ifft2(
            torch.fft.ifftshift(C * torch.exp(1j * kz * z), dim=(-2, -1))
        )

    def forward(self, inputs):
        assert inputs.is_complex(), "Propagation expects complex64 inputs."
        # inputs: (B, 1, H, W) or (B, L, H, W)  — propagate per-channel
        B, C, H, W = inputs.shape
        if H != self.units or W != self.units:
            raise RuntimeError(
                f"Propagation got input ({H}x{W}) but was built for units={self.units}"
            )

        if self.pad_px > 0:
            p = self.pad_px
            # Pad on the spatial dims (keep B and C intact)
            Ein = complex_pad(inputs, p, p)                # (B, C, H+2p, W+2p)
            Eout = self._propagate(Ein, self.kz_pad, self.z)
            return complex_crop(Eout, H, W, p, p)          # (B, C, H, W)
        return self._propagate(inputs, self.kz_base, self.z)


# ============================================================
# Diffraction layer (phase mask + propagation)
# ============================================================
class DiffractionLayer(nn.Module):
    """Phase-only mask of shape (units, units), then free-space propagation by z."""

    def __init__(self, units, dx, lam, z, device, pad_px=0):
        super().__init__()
        self.units  = int(units)
        self.dx     = float(dx)
        self.lam    = float(lam)
        self.z      = float(z)
        self.pad_px = int(pad_px)

        self.phase = nn.Parameter(torch.randn(self.units, self.units, dtype=torch.float32))

        self.register_buffer("kz_base", Propagation._make_kz(self.units, self.dx, self.lam, device))
        if self.pad_px > 0:
            units_pad = self.units + 2 * self.pad_px
            self.register_buffer("kz_pad", Propagation._make_kz(units_pad, self.dx, self.lam, device))
        else:
            self.kz_pad = None

    @staticmethod
    def _propagate(E, kz, z):
        return Propagation._propagate(E, kz, z)

    def forward(self, inputs):
        assert inputs.is_complex(), "DiffractionLayer expects complex64 inputs."
        B, C, H, W = inputs.shape
        if H != self.units or W != self.units:
            raise RuntimeError(
                f"DiffractionLayer got input ({H}x{W}) but was built for units={self.units}"
            )

        phase_c = torch.exp(
            1j * self.phase.to(inputs.device, dtype=torch.float32)
        ).to(torch.complex64)                              # (H, W)

        if self.pad_px > 0:
            p = self.pad_px
            Ein = complex_pad(inputs, p, p)                # (B, C, H+2p, W+2p)

            # phase canvas: 1 outside the mask area, mask phase inside
            phase_big = torch.ones(
                H + 2 * p, W + 2 * p,
                dtype=torch.complex64, device=inputs.device,
            )
            phase_big[p:p + H, p:p + W] = phase_c
            Ein = Ein * phase_big                          # broadcast over B, C

            Eout = self._propagate(Ein, self.kz_pad, self.z)
            return complex_crop(Eout, H, W, p, p)          # (B, C, H, W)

        Ein = inputs * phase_c                             # broadcast over B, C
        return self._propagate(Ein, self.kz_base, self.z)


# ============================================================
# Detector (intensity)
# ============================================================
class RegressionDetector(nn.Module):
    def forward(self, inputs):
        return torch.abs(inputs) ** 2


# ============================================================
# D2NN single-wavelength model
# ============================================================
class D2NNModel(nn.Module):
    """
    Inputs : (B, 1, layer_size, layer_size) complex64
    Outputs: (B, 1, out_size,   out_size  ) float32 (intensity)
    """

    def __init__(
        self,
        num_layers,
        layer_size,
        z_layers,
        z_prop,
        pixel_size,
        wavelength,
        device,
        padding_ratio: float = 0.5,
        z_input_to_first: float = 0.0,
        out_size: int | None = None,
        padding_ratio_out: float | None = None,
        perturb_cfg: dict | None = None,
    ):
        super().__init__()
        self.layer_size = int(layer_size)
        self.out_size   = int(out_size) if out_size is not None else int(layer_size)
        if self.out_size < self.layer_size:
            raise ValueError(
                f"out_size ({self.out_size}) must be >= layer_size ({self.layer_size})"
            )

        # ---- perturb_cfg: alignment-error injection ----
        self.perturb_cfg = perturb_cfg or {}
        z_sigma = float(self.perturb_cfg.get("z_sigma", 0.0))

        def _zj(z_nominal: float) -> float:
            if z_sigma > 0:
                return float(z_nominal + np.random.randn() * z_sigma)
            return float(z_nominal)

        z_layers         = _zj(z_layers)
        z_prop           = _zj(z_prop)
        z_input_to_first = _zj(z_input_to_first)

        pad_px  = int(round(self.layer_size * padding_ratio))
        out_pad = (int(round(self.out_size * padding_ratio_out))
                   if padding_ratio_out is not None else 0)

        # input -> first layer
        self.pre_propagation = Propagation(
            self.layer_size, pixel_size, wavelength, z_input_to_first,
            device, pad_px=pad_px,
        )
        # diffraction layers (all on layer_size grid)
        self.layers = nn.ModuleList([
            DiffractionLayer(
                self.layer_size, pixel_size, wavelength, z_layers,
                device, pad_px=pad_px,
            )
            for _ in range(int(num_layers))
        ])
        # final free-space propagation: lives on the OUT_SIZE grid
        self.propagation = Propagation(
            self.out_size, pixel_size, wavelength, z_prop,
            device, pad_px=out_pad,
        )
        self.regression = RegressionDetector()

    # ----- helper: embed (B,C,layer_size,layer_size) into out_size canvas -----
    def _embed_to_out_canvas(self, x: torch.Tensor) -> torch.Tensor:
        if self.out_size == self.layer_size:
            return x
        B, C, H, W = x.shape
        out = torch.zeros(
            (B, C, self.out_size, self.out_size),
            dtype=x.dtype, device=x.device,
        )
        oy = (self.out_size - H) // 2
        ox = (self.out_size - W) // 2
        out[:, :, oy:oy + H, ox:ox + W] = x
        return out

    # ----- forward -----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.is_complex(), "D2NNModel expects complex inputs."
        x = self.pre_propagation(x)                 # (B,1,layer_size,layer_size)
        for layer in self.layers:
            x = layer(x)                            # still on layer_size grid
        x = self._embed_to_out_canvas(x)            # (B,1,out_size,out_size)
        x = self.propagation(x)                     # final propagation
        return self.regression(x)                   # (B,1,out_size,out_size) intensity
