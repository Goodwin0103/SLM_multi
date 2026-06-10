"""
Training IO utilities for D2NN.

Includes:
  - save_masks_one_file_per_layer
  - save_to_mat_light_plus
  - train_multiwl  (multi-wavelength standard trainer, no staging)
"""
from __future__ import annotations

import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.io import savemat
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset


def train_multiwl(
    model: torch.nn.Module,
    train_datasets_per_wl: List[torch.utils.data.TensorDataset],
    *,
    wavelengths: np.ndarray,
    base_wavelength_idx: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 424242,
    scheduler_gamma: float = 0.99,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    多波长 D2NN 标准训练（无分阶段）。

    每个 epoch:
        - 对每个波长 li 取出对应 dataset 的一个 batch
        - 对该波长在模型输出 (B,L,H,W) 中只取 [:, li] 与 label 计算 MSE
        - 把 L 个波长的 loss 等权求和后一次 backward
    """
    import torch.optim as optim
    from torch.optim.lr_scheduler import ExponentialLR
    from torch.utils.data import DataLoader

    L = len(wavelengths)
    if not (0 <= base_wavelength_idx < L):
        raise ValueError(f"base_wavelength_idx={base_wavelength_idx} out of range [0,{L})")
    if len(train_datasets_per_wl) != L:
        raise ValueError(f"Expected {L} datasets, got {len(train_datasets_per_wl)}")

    # 每个波长一个 DataLoader
    g = torch.Generator(); g.manual_seed(seed)
    loaders = [
        DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g, drop_last=False)
        for ds in train_datasets_per_wl
    ]

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=scheduler_gamma)

    if verbose:
        print(f"\n{'='*70}")
        print(f"Multi-Wavelength Training (standard, all-λ simultaneous)")
        print(f"{'='*70}")
        print(f"  L         = {L}")
        print(f"  λ list    = {[f'{w*1e9:.1f}nm' for w in wavelengths]}")
        print(f"  base λ    = {wavelengths[base_wavelength_idx]*1e9:.1f}nm "
              f"(idx={base_wavelength_idx})")
        print(f"  epochs    = {epochs}")
        print(f"  batch     = {batch_size}")
        print(f"  lr        = {lr}")
        print(f"{'='*70}\n")

    losses: List[float] = []
    epoch_durations: List[float] = []

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.train()

        # 每个波长各自迭代它的 loader；以最短 loader 长度对齐
        iters = [iter(dl) for dl in loaders]
        n_iter = min(len(dl) for dl in loaders)
        epoch_loss = 0.0

        for _ in range(n_iter):
            optimizer.zero_grad(set_to_none=True)

            # ★ 累积所有波长的 loss(等权)
            loss_total = torch.zeros((), device=device, dtype=torch.float32)
            for li in range(L):
                batch = next(iters[li])
                if len(batch) == 3:
                    images, label_img, _amp = batch
                else:
                    images, label_img = batch

                images    = images.to(device, dtype=torch.complex64,  non_blocking=True)
                label_img = label_img.to(device, dtype=torch.float32, non_blocking=True)
                if images.ndim == 3:
                    images = images.unsqueeze(1)

                # 单通道输入 -> L 通道
                x = images if images.shape[1] == L else images.repeat(1, L, 1, 1).contiguous()
                I_blhw = model(x)                    # (B, L, H, W)
                loss_li = F.mse_loss(I_blhw[:, li], label_img[:, 0])
                loss_total = loss_total + loss_li

            loss_total = loss_total / L              # 取均值
            loss_total.backward()
            optimizer.step()
            epoch_loss += float(loss_total.item())

        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        epoch_loss /= max(1, n_iter)
        losses.append(epoch_loss)
        epoch_durations.append(time.time() - epoch_t0)

        if verbose and (epoch % 100 == 0 or epoch in (1, epochs)):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch [{epoch}/{epochs}]  loss={epoch_loss:.10f}  "
                  f"lr={current_lr:.6f}  time={epoch_durations[-1]:.2f}s")

    total_time = time.time() - t_start
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ Training completed!")
        print(f"  Total time : {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"  Final loss : {losses[-1]:.10f}")
        print(f"{'='*70}")

    return {
        "losses":          losses,
        "epoch_durations": epoch_durations,
        "total_time":      total_time,
        "final_loss":      losses[-1] if losses else float("nan"),
        "stage_info":      None,                      # 保留 key 以兼容
    }
