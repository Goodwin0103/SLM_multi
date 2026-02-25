import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ========================= 新增代码1：轻量级调制网络 =========================
class WavelengthModulationNetwork(nn.Module):
    """
    轻量级MLP：波长 → 相位调制
    参数量：仅约10k
    """
    def __init__(self, units, hidden_dim=32):
        super().__init__()
        self.units = units
        
        # 小型MLP
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),  # 输出低分辨率 16×16
            nn.Tanh()
        )
    
    def forward(self, wavelength):
        """
        wavelength: scalar (e.g., 1530.0)
        return: (units, units) 相位调制矩阵
        """
        # 归一化波长到 [-1, 1]
        wl_normalized = (wavelength - 1547.5) / 17.5
        wl_input = torch.tensor([[wl_normalized]], 
                               dtype=torch.float32,
                               device=self.mlp[0].weight.device)
        
        # 通过MLP生成低分辨率调制
        modulation_flat = self.mlp(wl_input)  # (1, 256)
        modulation_small = modulation_flat.view(1, 1, 16, 16)  # (1, 1, 16, 16)
        
        # 上采样到目标尺寸
        modulation = F.interpolate(
            modulation_small,
            size=(self.units, self.units),
            mode='bilinear',
            align_corners=True
        ).squeeze()  # (units, units)
        
        return modulation


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

        self.register_buffer("kz_base", self._make_kz_stack(self.units, self.dx, wl, device))  # (L,N,N)

        if self.pad_px > 0:
            units_pad = self.units + 2 * self.pad_px
            self.register_buffer("kz_pad", self._make_kz_stack(units_pad, self.dx, wl, device))  # (L,Np,Np)
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
            Ein = complex_pad(inputs, p, p)                  # (B,L,H+2p,W+2p)
            Eout = self._propagate(Ein, self.kz_pad, self.z) # (B,L,H+2p,W+2p)
            return complex_crop(Eout, H, W, p, p)            # (B,L,H,W)
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
    
    ===== 改进版：添加了波长特定调制 =====
    """

    def __init__(self, units, dx, wavelengths, z, device, pad_px=0, base_wavelength_idx=None, 
                 modulation_network=None):  # ← 新增参数
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

        # ===== 新增：波长调制网络 =====
        self.modulation_network = modulation_network
        if self.modulation_network is not None:
            self.modulation_strength = nn.Parameter(torch.tensor(0.3))
        # ================================

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
        scale = (self.lam0 / self.wavelengths).to(inputs.device)          # (L,)
        
        # ===== 修改：添加波长特定调制 =====
        phi_list = []
        for i in range(L):
            # 原有的波长缩放
            phi_scaled = self.phase * scale[i]
            
            # 如果有调制网络，添加波长特定调制
            if self.modulation_network is not None:
                wl_value = self.wavelengths[i].item()
                modulation = self.modulation_network(wl_value)  # (units, units)
                phi_i = phi_scaled + self.modulation_strength * modulation
            else:
                phi_i = phi_scaled
            
            phi_list.append(phi_i)
        
        phi = torch.stack(phi_list, dim=0)  # (L, H, W)
        # ====================================
        
        phase_c = torch.exp(1j * phi).to(torch.complex64)                 # (L,H,W)

        if self.pad_px > 0:
            p = self.pad_px
            Ein = complex_pad(inputs, p, p)                                # (B,L,H+2p,W+2p)

            # phase outside is 1 (no phase)
            phase_big = torch.ones(
                L, H + 2 * p, W + 2 * p, dtype=torch.complex64, device=inputs.device
            )
            phase_big[:, p : p + H, p : p + W] = phase_c
            Ein = Ein * phase_big[None]                                    # (B,L,H+2p,W+2p)

            Eout = self._propagate(Ein, self.kz_pad, self.z)
            return complex_crop(Eout, H, W, p, p)                           # (B,L,H,W)
        else:
            Ein = inputs * phase_c[None]                                    # (B,L,H,W)
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
# D2NN Multi-wavelength model
# -------------------------
class D2NNModelMultiWL(nn.Module):
    """
    inputs:  (B, L, H, W) complex
    outputs: (B, L, H, W) intensity
    
    ===== 改进版：添加了波长调制网络 =====
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
        use_wavelength_modulation=True,  # ← 新增参数：是否启用调制
    ):
        super().__init__()
        pad_px = int(round(layer_size * padding_ratio))

        # ===== 新增：创建共享的波长调制网络 =====
        if use_wavelength_modulation:
            self.shared_modulation_network = WavelengthModulationNetwork(
                units=layer_size,
                hidden_dim=32
            ).to(device)
        else:
            self.shared_modulation_network = None
        # =========================================

        self.pre_propagation = PropagationMultiWL(
            layer_size, pixel_size, wavelengths, z_input_to_first, device, pad_px=pad_px
        )

        # ===== 修改：传入调制网络 =====
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
                    modulation_network=self.shared_modulation_network,  # ← 传入调制网络
                )
                for _ in range(int(num_layers))
            ]
        )
        # ================================

        self.propagation = PropagationMultiWL(
            layer_size, pixel_size, wavelengths, z_prop, device, pad_px=pad_px
        )

        self.regression = RegressionDetector()

    def forward(self, x):
        x = self.pre_propagation(x)
        for layer in self.layers:
            x = layer(x)
        x = self.propagation(x)
        x = self.regression(x)  # (B,L,H,W)
        return x
