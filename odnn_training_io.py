from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from scipy.io import savemat

from odnn_training_eval import sample_tensor_slices


def save_masks_one_file_per_layer(
    phase_layers: list[np.ndarray] | list[torch.Tensor],
    out_dir: str | Path,
    *,
    base_name: str = "mask",
    save_degree: bool = False,
    use_xlsx: bool = True,
) -> None:
    """
    Persist per-layer phase masks to disk, one file per layer.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for index, mask in enumerate(phase_layers, start=1):
        array = np.asarray(mask, dtype=np.float32)
        if save_degree:
            array = np.degrees(array)

        file_name = f"{base_name}_layer{index}"
        if use_xlsx:
            df = pd.DataFrame(array)
            df.to_excel(out_dir / f"{file_name}.xlsx", index=False, header=False, engine="openpyxl")
        else:
            np.savetxt(out_dir / f"{file_name}.csv", array, delimiter=",")


def save_to_mat_light(
    filepath: str | Path,
    phase_stack: np.ndarray,
    input_field: torch.Tensor,
    propagated_fields: list[torch.Tensor],
    *,
    kmax: int = 16,
    save_amplitude_only: bool = True,
) -> None:
    """
    Lightweight MAT writer used for quick inspection of ODNN snapshots.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    masks = np.asarray(phase_stack, dtype=np.float32)
    input_field = input_field.detach().cpu()
    if save_amplitude_only:
        input_payload = torch.abs(input_field).to(torch.float32).numpy()
    else:
        input_payload = input_field.numpy()

    prop_slices = None
    if propagated_fields:
        tensor = propagated_fields[0].detach().cpu()
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        sampled = sample_tensor_slices(tensor, kmax)
        if save_amplitude_only:
            sampled = torch.abs(sampled).to(torch.float32)
        prop_slices = sampled.numpy()

    mdict = {"temp_model": masks, "temp_E": input_payload}
    if prop_slices is not None:
        mdict["propagated_slices"] = prop_slices

    savemat(filepath, mdict, do_compression=True)
    print(f"Saved (v5 light): {filepath}")


def save_to_mat_light_plus(
    filepath: str | Path,
    *,
    phase_stack: np.ndarray,
    input_field: np.ndarray,
    scans: Dict[str, Dict[str, np.ndarray]],
    camera_field: np.ndarray | None = None,
    sample_stacks_kmax: int = 20,
    save_amplitude_only: bool = False,
    meta: dict | None = None,
) -> None:
    """
    Extended MAT writer that preserves complex stacks and optional metadata.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {"temp_model": np.asarray(phase_stack, dtype=np.float32)}

    input_field = np.asarray(input_field)
    if save_amplitude_only:
        payload["E0_amp"] = np.abs(input_field)
    else:
        payload["E0"] = input_field

    for name, pack in scans.items():
        stack = np.asarray(pack["stack"])
        z_values = np.asarray(pack["z"])

        if sample_stacks_kmax is not None and stack.ndim == 3 and stack.shape[0] > sample_stacks_kmax:
            indices = np.linspace(0, stack.shape[0] - 1, sample_stacks_kmax, dtype=int)
            stack = stack[indices]
            z_values = z_values[indices]

        if save_amplitude_only:
            payload[f"{name}_amp"] = np.abs(stack)
        else:
            payload[name] = stack
        payload[f"{name}_z"] = z_values

    if camera_field is not None:
        camera_array = np.asarray(camera_field)
        if save_amplitude_only:
            payload["E_camera_amp"] = np.abs(camera_array)
        else:
            payload["E_camera"] = camera_array

    if meta is not None:
        payload["meta_json"] = json.dumps(meta, ensure_ascii=False)

    savemat(filepath, payload, do_compression=True)
    print(f"Saved (v5 plus): {filepath}")


# ============================================================
# 🆕 分阶段训练函数（Multi-Wavelength Staged Training）
# ============================================================

from typing import List, Optional
import time


def train_multiwl_staged(
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
    stage_ratios: Optional[List[float]] = None,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    分阶段训练多波长D2NN模型
    
    训练策略：
        - 阶段1: 只训练中心波长（base_wavelength）
        - 阶段2: 训练相邻波长（base ± 1）
        - 阶段3: 训练扩展范围（base ± 2）
        - 阶段4: 训练所有波长
    
    Args:
        model: D2NNModelMultiWL 模型实例
        train_datasets_per_wl: 每个波长的训练数据集列表，长度为 L
        wavelengths: 波长数组 (L,)，单位：米
        base_wavelength_idx: 基准波长索引（通常是中心波长）
        epochs: 总训练轮数
        batch_size: 批次大小
        lr: 初始学习率
        device: 训练设备
        seed: 随机种子
        scheduler_gamma: 学习率衰减因子
        stage_ratios: 每个阶段的epoch比例，默认 [0.25, 0.25, 0.25, 0.25]
        verbose: 是否打印详细信息
    
    Returns:
        包含训练历史的字典：
            - losses: 每个epoch的损失列表
            - epoch_durations: 每个epoch的训练时间列表
            - stage_info: 每个阶段的信息
            - total_time: 总训练时间
            - final_loss: 最终损失
    
    Example:
        >>> result = train_multiwl_staged(
        ...     model=model,
        ...     train_datasets_per_wl=train_datasets,
        ...     wavelengths=np.array([1530e-9, 1540e-9, 1550e-9]),
        ...     base_wavelength_idx=1,
        ...     epochs=1000,
        ...     batch_size=16,
        ...     lr=1e-3,
        ...     device=torch.device('cuda'),
        ... )
        >>> print(f"Final loss: {result['final_loss']:.6f}")
    """
    import torch.optim as optim
    from torch.optim.lr_scheduler import ExponentialLR
    from torch.utils.data import DataLoader
    
    L = len(wavelengths)
    base_idx = base_wavelength_idx
    
    # 验证输入
    if not (0 <= base_idx < L):
        raise ValueError(f"base_wavelength_idx={base_idx} out of range [0, {L})")
    
    if len(train_datasets_per_wl) != L:
        raise ValueError(f"Expected {L} datasets, got {len(train_datasets_per_wl)}")
    
    # 默认阶段比例
    if stage_ratios is None:
        stage_ratios = [0.25, 0.25, 0.25, 0.25]
    
    if not np.isclose(sum(stage_ratios), 1.0):
        raise ValueError(f"stage_ratios must sum to 1.0, got {sum(stage_ratios)}")
    
    # 优化器和调度器
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer, gamma=scheduler_gamma)
    
    # 定义训练阶段
    training_stages = _define_training_stages(
        L=L,
        base_idx=base_idx,
        epochs=epochs,
        stage_ratios=stage_ratios,
        wavelengths=wavelengths,
    )
    
    # 训练历史
    losses: List[float] = []
    epoch_durations: List[float] = []
    stage_info: List[Dict] = []
    
    # 随机数生成器
    g = torch.Generator()
    g.manual_seed(seed)
    
    # 开始训练
    t_start = time.time()
    total_epoch_count = 0
    
    for stage_idx, stage in enumerate(training_stages):
        stage_name = stage['name']
        active_wl_indices = stage['wl_indices']
        stage_epochs = stage['epochs']
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"{stage_name} (Stage {stage_idx+1}/{len(training_stages)})")
            print(f"Training wavelengths: {[wavelengths[i]*1e9 for i in active_wl_indices]} nm")
            print(f"Epochs: {total_epoch_count+1} - {total_epoch_count+stage_epochs}")
            print(f"{'='*70}")
        
        stage_t_start = time.time()
        stage_losses = []
        
        for epoch in range(1, stage_epochs + 1):
            epoch_t0 = time.time()
            
            # 训练一个epoch
            epoch_loss = _train_one_epoch_multiwl(
                model=model,
                train_datasets_per_wl=train_datasets_per_wl,
                active_wl_indices=active_wl_indices,
                optimizer=optimizer,
                device=device,
                batch_size=batch_size,
                L=L,
                generator=g,
            )
            
            scheduler.step()
            losses.append(epoch_loss)
            stage_losses.append(epoch_loss)
            
            # 同步GPU
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            
            epoch_duration = time.time() - epoch_t0
            epoch_durations.append(epoch_duration)
            
            # 打印进度
            if verbose and (epoch % 50 == 0 or epoch == 1 or epoch == stage_epochs):
                print(f"  Epoch [{total_epoch_count+epoch}/{total_epoch_count+stage_epochs}] "
                      f"loss={epoch_loss:.10f} time={epoch_duration:.2f}s")
        
        stage_duration = time.time() - stage_t_start
        total_epoch_count += stage_epochs
        
        # 记录阶段信息
        stage_info.append({
            'stage_idx': stage_idx,
            'name': stage_name,
            'wl_indices': active_wl_indices,
            'wavelengths_nm': [wavelengths[i]*1e9 for i in active_wl_indices],
            'epochs': stage_epochs,
            'avg_loss': float(np.mean(stage_losses)),
            'final_loss': stage_losses[-1],
            'duration_sec': stage_duration,
        })
        
        if verbose:
            print(f"✔ {stage_name} completed: avg_loss={np.mean(stage_losses):.10f}, "
                  f"time={stage_duration:.2f}s")
    
    total_time = time.time() - t_start
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ Staged training completed!")
        print(f"Total time: {total_time:.2f}s ({total_time/60:.2f} min)")
        print(f"Final loss: {losses[-1]:.10f}")
        print(f"{'='*70}")
    
    return {
        'losses': losses,
        'epoch_durations': epoch_durations,
        'stage_info': stage_info,
        'total_time': total_time,
        'final_loss': losses[-1],
    }


def _define_training_stages(
    L: int,
    base_idx: int,
    epochs: int,
    stage_ratios: List[float],
    wavelengths: np.ndarray,
) -> List[Dict]:
    """
    定义训练阶段
    
    Args:
        L: 波长总数
        base_idx: 基准波长索引
        epochs: 总epoch数
        stage_ratios: 每个阶段的epoch比例
        wavelengths: 波长数组
    
    Returns:
        训练阶段列表
    """
    # 计算每个阶段的epoch数
    stage_epochs = [int(epochs * ratio) for ratio in stage_ratios[:-1]]
    stage_epochs.append(epochs - sum(stage_epochs))  # 最后一个阶段取剩余
    
    # 定义每个阶段的波长索引
    stages = []
    
    # 阶段1: 只有中心波长
    stages.append({
        'name': 'Stage 1: Center wavelength',
        'wl_indices': [base_idx],
        'epochs': stage_epochs[0],
    })
    
    # 阶段2: 相邻波长 (base ± 1)
    wl_stage2 = [base_idx]
    if base_idx > 0:
        wl_stage2.insert(0, base_idx - 1)
    if base_idx < L - 1:
        wl_stage2.append(base_idx + 1)
    
    stages.append({
        'name': 'Stage 2: Adjacent wavelengths',
        'wl_indices': wl_stage2,
        'epochs': stage_epochs[1],
    })
    
    # 阶段3: 扩展范围 (base ± 2)
    wl_stage3 = list(range(max(0, base_idx - 2), min(L, base_idx + 3)))
    
    stages.append({
        'name': 'Stage 3: Extended range',
        'wl_indices': wl_stage3,
        'epochs': stage_epochs[2],
    })
    
    # 阶段4: 所有波长
    stages.append({
        'name': 'Stage 4: All wavelengths',
        'wl_indices': list(range(L)),
        'epochs': stage_epochs[3],
    })
    
    return stages


def _train_one_epoch_multiwl(
    model: torch.nn.Module,
    train_datasets_per_wl: List[torch.utils.data.TensorDataset],
    active_wl_indices: List[int],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    batch_size: int,
    L: int,
    generator: torch.Generator,
) -> float:
    """
    训练一个epoch（内部辅助函数）
    
    Args:
        model: 模型
        train_datasets_per_wl: 每个波长的数据集
        active_wl_indices: 当前激活的波长索引
        optimizer: 优化器
        device: 设备
        batch_size: 批次大小
        L: 总波长数
        generator: 随机数生成器
    
    Returns:
        该epoch的平均损失
    """
    from torch.utils.data import DataLoader
    import torch.nn.functional as F
    
    model.train()
    epoch_loss = 0.0
    batch_count = 0
    
    for wl_idx in active_wl_indices:
        train_loader_wl = DataLoader(
            train_datasets_per_wl[wl_idx],
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
        )
        
        for batch in train_loader_wl:
            if len(batch) == 3:
                images, label_img, amp = batch
            else:
                images, label_img = batch
            
            images = images.to(device, dtype=torch.complex64, non_blocking=True)
            label_img = label_img.to(device, dtype=torch.float32, non_blocking=True)
            
            # 确保输入维度正确
            if images.ndim == 3:
                images = images.unsqueeze(1)  # (B, 1, H, W)
            
            # 复制到所有波长通道
            x = images.repeat(1, L, 1, 1).contiguous()  # (B, L, H, W)
            
            # 前向传播
            optimizer.zero_grad(set_to_none=True)
            I_blhw = model(x)  # (B, L, H, W)
            
            # 只计算当前波长的损失
            loss = F.mse_loss(I_blhw[:, wl_idx], label_img[:, 0])
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_loss += float(loss.item())
            batch_count += 1
    
    return epoch_loss / max(1, batch_count)


def print_stage_summary(stage_info: List[Dict]) -> None:
    """
    打印训练阶段摘要
    
    Args:
        stage_info: 阶段信息列表（来自 train_multiwl_staged 的返回值）
    
    Example:
        >>> result = train_multiwl_staged(...)
        >>> print_stage_summary(result['stage_info'])
    """
    print(f"\n{'='*70}")
    print("Training Stage Summary")
    print(f"{'='*70}")
    print(f"{'Stage':<8} {'Wavelengths (nm)':<30} {'Epochs':<8} {'Avg Loss':<12} {'Time (s)':<10}")
    print(f"{'-'*70}")
    
    for info in stage_info:
        wl_str = ', '.join([f"{wl:.0f}" for wl in info['wavelengths_nm']])
        if len(wl_str) > 28:
            wl_str = wl_str[:25] + "..."
        
        print(f"{info['stage_idx']+1:<8} {wl_str:<30} {info['epochs']:<8} "
              f"{info['avg_loss']:<12.6e} {info['duration_sec']:<10.2f}")
    
    print(f"{'='*70}\n")


def save_staged_training_info(
    stage_info: List[Dict],
    output_path: str | Path,
) -> None:
    """
    保存训练阶段信息到文本文件
    
    Args:
        stage_info: 阶段信息列表（来自 train_multiwl_staged 的返回值）
        output_path: 输出文件路径（.txt）
    
    Example:
        >>> result = train_multiwl_staged(...)
        >>> save_staged_training_info(result['stage_info'], 'stage_info.txt')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("Staged Training Information\n")
        f.write("="*70 + "\n\n")
        
        for info in stage_info:
            f.write(f"Stage {info['stage_idx']+1}: {info['name']}\n")
            f.write(f"  Wavelengths: {info['wavelengths_nm']} nm\n")
            f.write(f"  Epochs: {info['epochs']}\n")
            f.write(f"  Average Loss: {info['avg_loss']:.10f}\n")
            f.write(f"  Final Loss: {info['final_loss']:.10f}\n")
            f.write(f"  Duration: {info['duration_sec']:.2f} seconds\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
    
    print(f"✔ Stage info saved -> {output_path}")


def save_staged_training_to_mat(
    stage_info: List[Dict],
    losses: List[float],
    epoch_durations: List[float],
    output_path: str | Path,
    *,
    num_layers: int,
    wavelengths: np.ndarray,
) -> None:
    """
    保存分阶段训练信息到MAT文件
    
    Args:
        stage_info: 阶段信息列表
        losses: 每个epoch的损失
        epoch_durations: 每个epoch的训练时间
        output_path: 输出MAT文件路径
        num_layers: 模型层数
        wavelengths: 波长数组
    
    Example:
        >>> result = train_multiwl_staged(...)
        >>> save_staged_training_to_mat(
        ...     stage_info=result['stage_info'],
        ...     losses=result['losses'],
        ...     epoch_durations=result['epoch_durations'],
        ...     output_path='training_info.mat',
        ...     num_layers=10,
        ...     wavelengths=wavelengths,
        ... )
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 准备MAT数据
    mat_data = {
        'losses': np.array(losses, dtype=np.float64),
        'epoch_durations': np.array(epoch_durations, dtype=np.float64),
        'cumulative_time': np.cumsum(epoch_durations),
        'total_time': np.sum(epoch_durations),
        'num_layers': np.array([num_layers], dtype=np.int32),
        'wavelengths_nm': wavelengths * 1e9,
        'num_stages': np.array([len(stage_info)], dtype=np.int32),
    }
    
    # 添加每个阶段的信息
    for i, info in enumerate(stage_info, start=1):
        prefix = f'stage{i}_'
        mat_data[f'{prefix}name'] = info['name']
        mat_data[f'{prefix}wl_indices'] = np.array(info['wl_indices'], dtype=np.int32)
        mat_data[f'{prefix}wavelengths_nm'] = np.array(info['wavelengths_nm'], dtype=np.float32)
        mat_data[f'{prefix}epochs'] = np.array([info['epochs']], dtype=np.int32)
        mat_data[f'{prefix}avg_loss'] = np.array([info['avg_loss']], dtype=np.float64)
        mat_data[f'{prefix}final_loss'] = np.array([info['final_loss']], dtype=np.float64)
        mat_data[f'{prefix}duration_sec'] = np.array([info['duration_sec']], dtype=np.float64)
    
    savemat(str(output_path), mat_data, do_compression=True)
    print(f"✔ Staged training MAT saved -> {output_path}")
