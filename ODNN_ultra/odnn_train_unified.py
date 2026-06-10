"""Unified training entry point — dispatches single/multi-wavelength."""
from __future__ import annotations
import time
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, TensorDataset


def train_d2nn_unified(
    model: nn.Module,
    train_data,
    *,
    geom,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 424242,
    scheduler_gamma: float = 0.99,
    verbose: bool = True,
):
    # ---------- Multi-wavelength path ----------
    if geom.is_multiwavelength:
        from odnn_training_io import train_multiwl          # ★ 改这里
        if not isinstance(train_data, (list, tuple)):
            raise TypeError("Multi-wl training expects list[TensorDataset] (one per λ).")
        return train_multiwl(                                # ★ 改这里
            model                 = model,
            train_datasets_per_wl = list(train_data),
            wavelengths           = np.asarray(geom.wavelength_list, dtype=np.float64),
            base_wavelength_idx   = (geom.base_wavelength_idx
                                     if geom.base_wavelength_idx is not None
                                     else geom.L // 2),
            epochs                = epochs,
            batch_size            = batch_size,
            lr                    = lr,
            device                = device,
            seed                  = seed,
            scheduler_gamma       = scheduler_gamma,
            verbose               = verbose,
        )


    # ---------- Single-wavelength path ----------
    if isinstance(train_data, DataLoader):
        train_loader = train_data
    else:
        if isinstance(train_data, (list, tuple)):
            train_data = TensorDataset(*[torch.stack(t) for t in zip(*train_data)])
        g = torch.Generator(); g.manual_seed(seed)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=g)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=scheduler_gamma)

    losses, durations = [], []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        te = time.time()
        model.train()
        eloss = 0.0
        for images, labels in train_loader:
            images = images.to(device, dtype=torch.complex64, non_blocking=True)
            labels = labels.to(device, dtype=torch.float32,  non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            out = model(images)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            eloss += loss.item()
        scheduler.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        losses.append(eloss / max(1, len(train_loader)))
        durations.append(time.time() - te)
        if verbose and (epoch % 100 == 0 or epoch in (1, epochs)):
            current_lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch [{epoch}/{epochs}]  loss={losses[-1]:.10f}  "
                  f"lr={current_lr:.6f}  time={durations[-1]:.2f}s")

    total_time = time.time() - t0
    return {
        "losses":          losses,
        "epoch_durations": durations,
        "total_time":      total_time,
        "final_loss":      losses[-1] if losses else float("nan"),
        "stage_info":      None,
    }
