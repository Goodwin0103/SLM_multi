import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Complex padding helpers
# -------------------------
def complex_pad(E, pad_h, pad_w):
    """
    E: (..., H, W) complex64/complex128
    return: (..., H+2pad_h, W+2pad_w) complex
    """
    Er = torch.view_as_real(E)  # (..., H, W, 2)
    Er_pad = F.pad(Er, (0, 0, pad_w, pad_w, pad_h, pad_h), mode="constant", value=0)
    return torch.view_as_complex(Er_pad.contiguous())


def complex_crop(E_pad, H, W, pad_h, pad_w):
    """
    E_pad: (..., H+2pad_h, W+2pad_w)
    """
    return E_pad[..., pad_h : pad_h + H, pad_w : pad_w + W].contiguous()


# -------------------------
# Multi-wavelength propagation
# -------------------------
class PropagationMultiWL(nn.Module):
    """
    多波长自由传播层
    inputs: (B, L, H, W) complex
    outputs: (B, L, H, W) complex
    """

    def __init__(self, units, dx, wavelengths, z, device, pad_px=0):
        super().__init__()
        self.units = int(units)
        self.dx = float(dx)
        self.z = float(z)
        self.pad_px = int(pad_px)

        wl = torch.tensor(wavelengths, dtype=torch.float32, device=device)  # (L,)
        self.register_buffer("wavelengths", wl)

        self.register_buffer("kz_base", self._make_kz_stack(self.units, self.dx, wl, device))

        if self.pad_px > 0:
            units_pad = self.units + 2 * self.pad_px
            self.register_buffer("kz_pad", self._make_kz_stack(units_pad, self.dx, wl, device))
        else:
            self.kz_pad = None

    @staticmethod
    def _make_kz_stack(N, dx, wavelengths, device):
        """
        return kz: (L, N, N) complex64
        """
        fx = torch.fft.fftshift(torch.fft.fftfreq(N, d=dx)).to(device)
        fxx, fyy = torch.meshgrid(fx, fx, indexing="ij")  # (N,N)

        inv_lam2 = (1.0 / wavelengths)[:, None, None] ** 2  # (L,1,1)
        argument = (2 * torch.pi) ** 2 * (inv_lam2 - fxx[None] ** 2 - fyy[None] ** 2)  # (L,N,N)

        tmp = torch.sqrt(torch.abs(argument))
        kz = torch.where(argument >= 0, tmp, 1j * tmp).to(torch.complex64)
        return kz

    @staticmethod
    def _propagate(E, kz, z):
        """
        E: (B,L,N,N) complex
        kz: (L,N,N) complex
        """
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
        else:
            return self._propagate(inputs, self.kz_base, self.z)


# -------------------------
# Multi-wavelength diffraction layer (phase mask + propagation)
# -------------------------
class DiffractionLayerMultiWL(nn.Module):
    """
    多波长衍射层：相位掩膜对不同 λ 按 lam0/lam 缩放 + 多波长传播
    inputs:  (B, L, H, W) complex
    outputs: (B, L, H, W) complex
    """

    def __init__(self, units, dx, wavelengths, z, device, pad_px=0, base_wavelength_idx=None):
        super().__init__()
        self.units = int(units)
        self.dx = float(dx)
        self.z = float(z)
        self.pad_px = int(pad_px)

        wl = torch.tensor(wavelengths, dtype=torch.float32, device=device)  # (L,)
        self.register_buffer("wavelengths", wl)

        if base_wavelength_idx is None:
            base_wavelength_idx = int(len(wavelengths) // 2)
        self.base_wavelength_idx = int(base_wavelength_idx)
        self.register_buffer("lam0", wl[self.base_wavelength_idx].clone())

        # trainable base mask phase for lam0
        self.phase = nn.Parameter(torch.randn(self.units, self.units, dtype=torch.float32))

        # kz stacks
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

        # build wavelength-scaled phase: phi_l = phi0 * (lam0/lam_l)
        scale = (self.lam0 / self.wavelengths).to(inputs.device)
        phi = self.phase[None, :, :] * scale[:, None, None]
        phase_c = torch.exp(1j * phi).to(torch.complex64)

        if self.pad_px > 0:
            p = self.pad_px
            Ein = complex_pad(inputs, p, p)

            phase_big = torch.ones(
                L, H + 2 * p, W + 2 * p, dtype=torch.complex64, device=inputs.device
            )
            phase_big[:, p : p + H, p : p + W] = phase_c
            Ein = Ein * phase_big[None]

            Eout = self._propagate(Ein, self.kz_pad, self.z)
            return complex_crop(Eout, H, W, p, p)
        else:
            Ein = inputs * phase_c[None]
            return self._propagate(Ein, self.kz_base, self.z)


# -------------------------
# Detector
# -------------------------
class RegressionDetector(nn.Module):
    """
    输出强度图（不做 ROI 聚合）
    inputs:  (B, L, H, W) complex
    outputs: (B, L, H, W) float
    """
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return torch.abs(inputs) ** 2


# -------------------------
# D2NN Multi-wavelength model (★ 支持 out_size)
# -------------------------
class D2NNModelMultiWL(nn.Module):
    """
    inputs:  (B, L, layer_size, layer_size) complex
    outputs: (B, L, out_size, out_size) intensity
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
        out_size=None,              # ★ 新增
        padding_ratio_out=None,     # ★ 新增
    ):
        super().__init__()
        self.layer_size = int(layer_size)
        self.out_size = int(out_size) if out_size is not None else int(layer_size)

        pad_px = int(round(layer_size * padding_ratio))

        if padding_ratio_out is None:
            padding_ratio_out = padding_ratio
        self.padding_ratio_out = float(padding_ratio_out)
        pad_px_out = int(round(self.out_size * self.padding_ratio_out))

        self.pre_propagation = PropagationMultiWL(
            layer_size, pixel_size, wavelengths, z_input_to_first, device, pad_px=pad_px
        )

        self.layers = nn.ModuleList(
            [
                DiffractionLayerMultiWL(
                    layer_size,
                    pixel_size,
                    wavelengths,
                    z_layers,
                    device,
                    pad_px=pad_px,
                    base_wavelength_idx=base_wavelength_idx,
                )
                for _ in range(int(num_layers))
            ]
        )

        # ★ 最后传播层使用 out_size
        self.propagation = PropagationMultiWL(
            self.out_size, pixel_size, wavelengths, z_prop, device, pad_px=pad_px_out
        )

        self.regression = RegressionDetector()

    def _embed_to_out_canvas(self, x):
        """
        将 (B, L, layer_size, layer_size) 嵌入到 (B, L, out_size, out_size)
        out_size > layer_size: zero-pad
        out_size < layer_size: center-crop
        out_size == layer_size: 直接返回
        """
        if self.out_size == self.layer_size:
            return x

        B, L, H, W = x.shape
        diff_h = self.out_size - H
        diff_w = self.out_size - W

        if diff_h >= 0 and diff_w >= 0:
            # zero-pad
            pt = diff_h // 2
            pb = diff_h - pt
            pl = diff_w // 2
            pr = diff_w - pl
            Er = torch.view_as_real(x)  # (B, L, H, W, 2)
            Er_pad = F.pad(Er, (0, 0, pl, pr, pt, pb), mode='constant', value=0)
            return torch.view_as_complex(Er_pad.contiguous())
        else:
            # center-crop
            crop_h = (-diff_h) // 2
            crop_w = (-diff_w) // 2
            return x[:, :, crop_h:crop_h + self.out_size, crop_w:crop_w + self.out_size].contiguous()

    def forward(self, x):
        x = self.pre_propagation(x)
        for layer in self.layers:
            x = layer(x)
        x = self._embed_to_out_canvas(x)   # ★ 嵌入到 out_size 画布
        x = self.propagation(x)
        x = self.regression(x)             # (B, L, out_size, out_size)
        return x
