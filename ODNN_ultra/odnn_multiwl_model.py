"""
Multi-wavelength D2NN model with optional perturbations.
Phase masks share a single physical SLM phi0 across all wavelengths,
scaled by phi_l = phi0 * (lam0/lam_l).  perturb_cfg works just like
the single-wavelength D2NNModel.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Complex padding helpers
# -------------------------
def complex_pad(E, pad_h, pad_w):
    Er = torch.view_as_real(E)
    Er_pad = F.pad(Er, (0, 0, pad_w, pad_w, pad_h, pad_h), mode="constant", value=0)
    return torch.view_as_complex(Er_pad.contiguous())


def complex_crop(E_pad, H, W, pad_h, pad_w):
    return E_pad[..., pad_h:pad_h + H, pad_w:pad_w + W].contiguous()


def _roll_complex_2d(E: torch.Tensor, sy: int, sx: int) -> torch.Tensor:
    """Shift along last 2 dims with zero-fill, complex-safe."""
    if sy == 0 and sx == 0:
        return E
    Er = torch.view_as_real(E)
    out = torch.zeros_like(Er)
    H, W = Er.shape[-3], Er.shape[-2]
    if abs(sy) >= H or abs(sx) >= W:
        return torch.view_as_complex(out)
    src_y = slice(0, H - sy) if sy >= 0 else slice(-sy, H)
    dst_y = slice(sy, H)     if sy >= 0 else slice(0, H + sy)
    src_x = slice(0, W - sx) if sx >= 0 else slice(-sx, W)
    dst_x = slice(sx, W)     if sx >= 0 else slice(0, W + sx)
    out[..., dst_y, dst_x, :] = Er[..., src_y, src_x, :]
    return torch.view_as_complex(out.contiguous())


# -------------------------
# Multi-wavelength propagation
# -------------------------
class PropagationMultiWL(nn.Module):
    """inputs/outputs: (B, L, H, W) complex64."""

    def __init__(self, units, dx, wavelengths, z, device, pad_px=0):
        super().__init__()
        self.units = int(units)
        self.dx = float(dx)
        self.z = float(z)
        self.pad_px = int(pad_px)

        wl = torch.tensor(wavelengths, dtype=torch.float32, device=device)
        self.register_buffer("wavelengths", wl)
        self.register_buffer("kz_base", self._make_kz_stack(self.units, self.dx, wl, device))
        if self.pad_px > 0:
            units_pad = self.units + 2 * self.pad_px
            self.register_buffer("kz_pad", self._make_kz_stack(units_pad, self.dx, wl, device))
        else:
            self.kz_pad = None

    @staticmethod
    def _make_kz_stack(N, dx, wavelengths, device):
        fx = torch.fft.fftshift(torch.fft.fftfreq(N, d=dx)).to(device)
        fxx, fyy = torch.meshgrid(fx, fx, indexing="ij")
        inv_lam2 = (1.0 / wavelengths)[:, None, None] ** 2
        argument = (2 * torch.pi) ** 2 * (inv_lam2 - fxx[None] ** 2 - fyy[None] ** 2)
        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j * tmp).to(torch.complex64)
        return kz

    @staticmethod
    def _propagate(E, kz, z):
        E = E.to(torch.complex64)
        C = torch.fft.fftshift(torch.fft.fft2(E), dim=(-2, -1))
        return torch.fft.ifft2(
            torch.fft.ifftshift(C * torch.exp(1j * kz[None] * z), dim=(-2, -1))
        )

    def forward(self, inputs):
        assert inputs.is_complex(), "PropagationMultiWL expects complex inputs."
        B, L, H, W = inputs.shape
        if L != int(self.wavelengths.numel()):
            raise ValueError(f"Input L={L} mismatches wavelengths={int(self.wavelengths.numel())}")
        if self.pad_px > 0:
            p = self.pad_px
            Ein = complex_pad(inputs, p, p)
            Eout = self._propagate(Ein, self.kz_pad, self.z)
            return complex_crop(Eout, H, W, p, p)
        return self._propagate(inputs, self.kz_base, self.z)


# -------------------------
# Multi-wavelength diffraction layer
# -------------------------
class DiffractionLayerMultiWL(nn.Module):
    """
    Single SLM physical phase phi0(x,y) shared across wavelengths.
    Effective phase at λ_l: phi_l = phi0 * (lam0/lam_l).
    Optional mask_shift_px shifts the phase plate (same shift for all λ).
    """

    def __init__(self, units, dx, wavelengths, z, device, pad_px=0,
                 base_wavelength_idx=None, mask_shift_px=(0, 0)):
        super().__init__()
        self.units = int(units)
        self.dx = float(dx)
        self.z = float(z)
        self.pad_px = int(pad_px)
        self.mask_shift_px = (int(mask_shift_px[0]), int(mask_shift_px[1]))

        wl = torch.tensor(wavelengths, dtype=torch.float32, device=device)
        self.register_buffer("wavelengths", wl)
        if base_wavelength_idx is None:
            base_wavelength_idx = int(len(wavelengths) // 2)
        self.base_wavelength_idx = int(base_wavelength_idx)
        self.register_buffer("lam0", wl[self.base_wavelength_idx].clone())

        self.phase = nn.Parameter(torch.randn(self.units, self.units, dtype=torch.float32))

        self.register_buffer("kz_base", PropagationMultiWL._make_kz_stack(self.units, self.dx, wl, device))
        if self.pad_px > 0:
            units_pad = self.units + 2 * self.pad_px
            self.register_buffer("kz_pad", PropagationMultiWL._make_kz_stack(units_pad, self.dx, wl, device))
        else:
            self.kz_pad = None

    @staticmethod
    def _propagate(E, kz, z):
        return PropagationMultiWL._propagate(E, kz, z)

    def forward(self, inputs):
        assert inputs.is_complex(), "DiffractionLayerMultiWL expects complex inputs."
        B, L, H, W = inputs.shape

        # phi_l = phi0 * (lam0/lam_l)
        scale = (self.lam0 / self.wavelengths).to(inputs.device)
        phi = self.phase[None, :, :] * scale[:, None, None]              # (L,H,W)
        phase_c = torch.exp(1j * phi).to(torch.complex64)                # (L,H,W)

        if self.pad_px > 0:
            p = self.pad_px
            Ein = complex_pad(inputs, p, p)                              # (B,L,H+2p,W+2p)
            phase_big = torch.ones(L, H + 2 * p, W + 2 * p,
                                   dtype=torch.complex64, device=inputs.device)
            phase_big[:, p:p + H, p:p + W] = phase_c
            # mask shift (same for all L)
            sy, sx = self.mask_shift_px
            if sy != 0 or sx != 0:
                phase_big = _roll_complex_2d(phase_big, sy, sx)
            Ein = Ein * phase_big[None]
            Eout = self._propagate(Ein, self.kz_pad, self.z)
            return complex_crop(Eout, H, W, p, p)

        # no padding
        sy, sx = self.mask_shift_px
        if sy != 0 or sx != 0:
            phase_c = _roll_complex_2d(phase_c, sy, sx)
        Ein = inputs * phase_c[None]
        return self._propagate(Ein, self.kz_base, self.z)


# -------------------------
# Detector
# -------------------------
class RegressionDetector(nn.Module):
    def forward(self, inputs):
        return torch.abs(inputs) ** 2


# -------------------------
# D2NN Multi-wavelength model
# -------------------------
class D2NNModelMultiWL(nn.Module):
    """
    inputs : (B, 1 | L, layer_size, layer_size) complex
    outputs: (B, L, out_size, out_size) intensity
    
    Single-wavelength input is auto-broadcast to L channels.
    The field is embedded into an out_size canvas before the final
    free-space propagation to the camera plane.
    """

    def __init__(
        self,
        num_layers,
        layer_size,
        z_layers,
        z_prop,
        pixel_size,
        wavelengths,
        device,
        padding_ratio=0.5,
        z_input_to_first=0.0,
        base_wavelength_idx=None,
        perturb_cfg: dict | None = None,
        out_size: int | None = None,                    # ★ NEW
        padding_ratio_out: float | None = None,         # ★ NEW
    ):
        super().__init__()
        self.layer_size = int(layer_size)
        self.out_size   = int(out_size) if out_size is not None else int(layer_size)
        if self.out_size < self.layer_size:
            raise ValueError(
                f"out_size ({self.out_size}) must be >= layer_size ({self.layer_size})"
            )

        pad_px = int(round(layer_size * padding_ratio))
        out_pad = (int(round(self.out_size * padding_ratio_out))
                   if padding_ratio_out is not None else 0)

        self.perturb_cfg = perturb_cfg or {}

        # ---- z jitter (same draw shared across all L) ----
        z_sigma = float(self.perturb_cfg.get("z_sigma", 0.0))
        def _zj(z_nominal: float) -> float:
            if z_sigma > 0:
                return float(z_nominal + np.random.randn() * z_sigma)
            return float(z_nominal)

        # ---- per-layer mask shift (px) ----
        shift_sigma = float(self.perturb_cfg.get("mask_shift_sigma_px", 0.0))
        def _shift_draw():
            if shift_sigma > 0:
                return (int(round(np.random.randn() * shift_sigma)),
                        int(round(np.random.randn() * shift_sigma)))
            return (0, 0)

        # pre-propagation on layer_size grid
        self.pre_propagation = PropagationMultiWL(
            self.layer_size, pixel_size, wavelengths,
            _zj(z_input_to_first), device, pad_px=pad_px,
        )
        # diffraction layers on layer_size grid
        self.layers = nn.ModuleList([
            DiffractionLayerMultiWL(
                self.layer_size, pixel_size, wavelengths,
                _zj(z_layers), device, pad_px=pad_px,
                base_wavelength_idx=base_wavelength_idx,
                mask_shift_px=_shift_draw(),
            )
            for _ in range(int(num_layers))
        ])
        # ★ FINAL propagation lives on out_size grid
        self.propagation = PropagationMultiWL(
            self.out_size, pixel_size, wavelengths,
            _zj(z_prop), device, pad_px=out_pad,
        )
        self.regression = RegressionDetector()

    @property
    def L(self) -> int:
        return int(self.propagation.wavelengths.numel())

    def _embed_to_out_canvas(self, x: torch.Tensor) -> torch.Tensor:
        """Center-embed (B,L,layer_size,layer_size) into (B,L,out_size,out_size)."""
        if self.out_size == self.layer_size:
            return x
        B, L, H, W = x.shape
        out = torch.zeros(
            (B, L, self.out_size, self.out_size),
            dtype=x.dtype, device=x.device,
        )
        oy = (self.out_size - H) // 2
        ox = (self.out_size - W) // 2
        out[:, :, oy:oy + H, ox:ox + W] = x
        return out

    def forward(self, x):
        assert x.is_complex(), "D2NNModelMultiWL expects complex inputs."
        # auto-broadcast (B,1,H,W) -> (B,L,H,W) on multi-WL models
        if x.ndim == 4 and x.shape[1] == 1 and self.L > 1:
            x = x.repeat(1, self.L, 1, 1).contiguous()

        x = self.pre_propagation(x)            # (B, L, layer_size, layer_size)
        for layer in self.layers:
            x = layer(x)                       # still on layer_size grid
        x = self._embed_to_out_canvas(x)       # ★ -> (B, L, out_size, out_size)
        x = self.propagation(x)                # final free-space propagation
        return self.regression(x)              # (B, L, out_size, out_size) intensity
