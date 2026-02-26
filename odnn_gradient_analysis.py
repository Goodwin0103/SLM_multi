"""
梯度冲突分析模块
用于诊断多波长训练中的梯度冲突问题
"""

import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader


# ============================================================
# 核心函数1：训练前诊断
# ============================================================
def diagnose_gradient_conflict_before_training(
    model: nn.Module,
    wavelengths: np.ndarray,
    train_loader: DataLoader,
    device: torch.device,
    num_batches: int = 10,
    save_dir: Path = None,
) -> Tuple[np.ndarray, Dict]:
    """
    训练前快速诊断梯度冲突
    
    Args:
        model: D2NN模型
        wavelengths: 波长数组 (L,) 单位：米
        train_loader: 训练数据加载器
        device: 设备
        num_batches: 采样的batch数量
        save_dir: 保存目录
    
    Returns:
        conflict_matrix: (L, L) 余弦相似度矩阵
        metrics: 统计指标字典
    """
    print("\n" + "="*70)
    print("🔍 训练前梯度冲突诊断")
    print("="*70)
    
    model.train()
    model.to(device)
    
    L = len(wavelengths)
    
    # 收集每个波长的梯度
    wavelength_gradients = {float(wl): [] for wl in wavelengths}
    
    batch_count = 0
    for batch_idx, batch in enumerate(train_loader):
        if batch_count >= num_batches:
            break
        
        # 解包数据
        if len(batch) == 3:
            images, label_img, amp = batch
        else:
            images, label_img = batch
            amp = None
        
        images = images.to(device, dtype=torch.complex64)
        label_img = label_img.to(device, dtype=torch.float32)
        
        if images.ndim == 3:
            images = images.unsqueeze(1)
        
        # 为每个波长单独计算梯度
        for wl_idx, wl in enumerate(wavelengths):
            model.zero_grad()
            
            # 复制输入到所有波长通道
            x = images.repeat(1, L, 1, 1).contiguous()
            
            # 前向传播
            I_blhw = model(x)  # (B, L, H, W)
            
            # 只计算当前波长的损失
            loss = F.mse_loss(I_blhw[:, wl_idx], label_img[:, 0])
            
            # 反向传播
            loss.backward()
            
            # 收集第一层的梯度
            if hasattr(model, 'layers') and len(model.layers) > 0:
                if hasattr(model.layers[0], 'phase'):
                    grad = model.layers[0].phase.grad.clone().cpu()
                    wavelength_gradients[float(wl)].append(grad)
        
        batch_count += 1
    
    print(f"✔ 已采样 {batch_count} 个批次")
    
    # 计算平均梯度
    avg_grads = {wl: torch.stack(wavelength_gradients[wl]).mean(0) 
                 for wl in wavelength_gradients.keys()}
    
    # 计算冲突矩阵
    conflict_matrix = np.zeros((L, L))
    cos_sims = []
    
    for i in range(L):
        for j in range(L):
            wl_i = float(wavelengths[i])
            wl_j = float(wavelengths[j])
            
            grad_i = avg_grads[wl_i].flatten()
            grad_j = avg_grads[wl_j].flatten()
            
            cos_sim = F.cosine_similarity(grad_i, grad_j, dim=0).item()
            conflict_matrix[i, j] = cos_sim
            
            if i < j:
                cos_sims.append(cos_sim)
    
    # 统计指标
    negative_ratio = sum(1 for x in cos_sims if x < 0) / len(cos_sims) if cos_sims else 0.0
    strong_conflict_ratio = sum(1 for x in cos_sims if x < -0.3) / len(cos_sims) if cos_sims else 0.0
    
    metrics = {
        'mean_similarity': float(np.mean(cos_sims)) if cos_sims else 0.0,
        'std_similarity': float(np.std(cos_sims)) if cos_sims else 0.0,
        'negative_ratio': float(negative_ratio),
        'strong_conflict_ratio': float(strong_conflict_ratio),
        'min_similarity': float(min(cos_sims)) if cos_sims else 0.0,
        'max_similarity': float(max(cos_sims)) if cos_sims else 0.0,
    }
    
    # 打印报告
    print(f"\n📊 诊断结果:")
    print(f"  波长数量: {L}")
    print(f"  波长对数: {len(cos_sims)}")
    print(f"  平均余弦相似度: {metrics['mean_similarity']:.3f}")
    print(f"  标准差: {metrics['std_similarity']:.3f}")
    print(f"  负相似度比例: {metrics['negative_ratio']:.1%}")
    print(f"  强冲突比例 (cos<-0.3): {metrics['strong_conflict_ratio']:.1%}")
    print(f"  最小相似度: {metrics['min_similarity']:.3f}")
    print(f"  最大相似度: {metrics['max_similarity']:.3f}")
    
    # 判断严重程度
    if metrics['negative_ratio'] > 0.3:
        print(f"\n⚠️  警告: 检测到严重梯度冲突！")
        print(f"     建议使用分阶段训练或加权损失")
    elif metrics['negative_ratio'] > 0.1:
        print(f"\n⚠️  注意: 存在中等程度梯度冲突")
        print(f"     建议监控训练过程")
    else:
        print(f"\n✅ 梯度冲突程度较低，可以正常训练")
    
    # 可视化
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 子图1：冲突矩阵热图
        im = axes[0].imshow(conflict_matrix, cmap='RdYlGn', vmin=-1, vmax=1)
        axes[0].set_xticks(range(L))
        axes[0].set_yticks(range(L))
        axes[0].set_xticklabels([f'{wl*1e9:.0f}' for wl in wavelengths], rotation=45)
        axes[0].set_yticklabels([f'{wl*1e9:.0f}' for wl in wavelengths])
        axes[0].set_title('Gradient Conflict Matrix\n(Cosine Similarity)', 
                         fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Wavelength (nm)')
        axes[0].set_ylabel('Wavelength (nm)')
        
        # 添加数值标注
        for i in range(L):
            for j in range(L):
                if i != j:
                    color = 'white' if abs(conflict_matrix[i, j]) > 0.5 else 'black'
                    axes[0].text(j, i, f'{conflict_matrix[i, j]:.2f}',
                               ha="center", va="center", color=color, fontsize=9)
        
        plt.colorbar(im, ax=axes[0], label='Cosine Similarity')
        
        # 子图2：相似度分布
        axes[1].scatter(range(len(cos_sims)), cos_sims, s=100, alpha=0.6)
        axes[1].axhline(0, color='red', linestyle='--', linewidth=2, 
                       label='Zero (Orthogonal)')
        axes[1].axhline(-0.3, color='orange', linestyle=':', linewidth=2, 
                       label='Conflict Threshold')
        axes[1].set_xlabel('Wavelength Pair Index', fontsize=12)
        axes[1].set_ylabel('Cosine Similarity', fontsize=12)
        axes[1].set_title('Gradient Alignment Between Wavelength Pairs', 
                         fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        fig_path = save_dir / 'gradient_conflict_diagnosis.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"\n✔ 诊断图已保存: {fig_path}")
    
    print("="*70 + "\n")
    
    return conflict_matrix, metrics


# ============================================================
# 核心函数2：训练中监控
# ============================================================
@torch.no_grad()
def monitor_gradient_conflict_during_training(
    model: nn.Module,
    wavelengths: np.ndarray,
    train_loader: DataLoader,
    device: torch.device,
    num_batches: int = 5,
) -> Dict:
    """
    训练中快速监控梯度冲突
    """
    model.eval()
    L = len(wavelengths)
    
    wavelength_gradients = {float(wl): [] for wl in wavelengths}
    
    with torch.enable_grad():
        batch_count = 0
        for batch in train_loader:
            if batch_count >= num_batches:
                break
            
            if len(batch) == 3:
                images, label_img, amp = batch
            else:
                images, label_img = batch
            
            images = images.to(device, dtype=torch.complex64)
            label_img = label_img.to(device, dtype=torch.float32)
            
            if images.ndim == 3:
                images = images.unsqueeze(1)
            
            for wl_idx, wl in enumerate(wavelengths):
                model.zero_grad()
                x = images.repeat(1, L, 1, 1).contiguous()
                I_blhw = model(x)
                loss = F.mse_loss(I_blhw[:, wl_idx], label_img[:, 0])
                loss.backward()
                
                if hasattr(model, 'layers') and len(model.layers) > 0:
                    if hasattr(model.layers[0], 'phase'):
                        grad = model.layers[0].phase.grad.clone().cpu()
                        wavelength_gradients[float(wl)].append(grad)
            
            batch_count += 1
    
    # 计算指标
    avg_grads = {wl: torch.stack(wavelength_gradients[wl]).mean(0) 
                 for wl in wavelength_gradients.keys()}
    
    cos_sims = []
    for i in range(L):
        for j in range(i+1, L):
            grad_i = avg_grads[float(wavelengths[i])].flatten()
            grad_j = avg_grads[float(wavelengths[j])].flatten()
            cos_sim = F.cosine_similarity(grad_i, grad_j, dim=0).item()
            cos_sims.append(cos_sim)
    
    metrics = {
        'mean_similarity': float(np.mean(cos_sims)) if cos_sims else 0.0,
        'std_similarity': float(np.std(cos_sims)) if cos_sims else 0.0,
        'negative_ratio': float(sum(1 for x in cos_sims if x < 0) / len(cos_sims)) if cos_sims else 0.0,
        'strong_conflict_ratio': float(sum(1 for x in cos_sims if x < -0.3) / len(cos_sims)) if cos_sims else 0.0,
    }
    
    model.train()
    return metrics


# ============================================================
# 核心函数3：训练后详细分析
# ============================================================
@torch.no_grad()
def detailed_gradient_analysis_after_training(
    model: nn.Module,
    wavelengths: np.ndarray,
    train_loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    num_batches: int = 20,
) -> List[Dict]:
    """
    训练后详细分析每层的梯度冲突
    """
    print("\n" + "="*70)
    print("📊 训练后详细梯度分析")
    print("="*70)
    
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.eval()
    L = len(wavelengths)
    
    if not hasattr(model, 'layers') or len(model.layers) == 0:
        print("⚠️  模型没有 layers 属性，跳过分析")
        return []
    
    num_layers = len(model.layers)
    
    # 收集所有层的梯度
    layer_gradients = {
        layer_idx: {float(wl): [] for wl in wavelengths}
        for layer_idx in range(num_layers)
    }
    
    with torch.enable_grad():
        batch_count = 0
        for batch in train_loader:
            if batch_count >= num_batches:
                break
            
            if len(batch) == 3:
                images, label_img, amp = batch
            else:
                images, label_img = batch
            
            images = images.to(device, dtype=torch.complex64)
            label_img = label_img.to(device, dtype=torch.float32)
            
            if images.ndim == 3:
                images = images.unsqueeze(1)
            
            for wl_idx, wl in enumerate(wavelengths):
                model.zero_grad()
                x = images.repeat(1, L, 1, 1).contiguous()
                I_blhw = model(x)
                loss = F.mse_loss(I_blhw[:, wl_idx], label_img[:, 0])
                loss.backward()
                
                # 收集每层的梯度
                for layer_idx, layer in enumerate(model.layers):
                    if hasattr(layer, 'phase'):
                        grad = layer.phase.grad.clone().cpu()
                        layer_gradients[layer_idx][float(wl)].append(grad)
            
            batch_count += 1
    
    print(f"✔ 已采样 {batch_count} 个批次")
    
    # 分析每层
    conflict_per_layer = []
    
    for layer_idx in range(num_layers):
        avg_grads = {
            wl: torch.stack(layer_gradients[layer_idx][wl]).mean(0)
            for wl in layer_gradients[layer_idx].keys()
        }
        
        cos_sims = []
        for i in range(L):
            for j in range(i+1, L):
                grad_i = avg_grads[float(wavelengths[i])].flatten()
                grad_j = avg_grads[float(wavelengths[j])].flatten()
                cos_sim = F.cosine_similarity(grad_i, grad_j, dim=0).item()
                cos_sims.append(cos_sim)
        
        conflict_per_layer.append({
            'layer': layer_idx,
            'mean_similarity': float(np.mean(cos_sims)) if cos_sims else 0.0,
            'negative_ratio': float(sum(1 for x in cos_sims if x < 0) / len(cos_sims)) if cos_sims else 0.0,
            'strong_conflict_ratio': float(sum(1 for x in cos_sims if x < -0.3) / len(cos_sims)) if cos_sims else 0.0,
        })
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    layers = [x['layer'] for x in conflict_per_layer]
    mean_sims = [x['mean_similarity'] for x in conflict_per_layer]
    neg_ratios = [x['negative_ratio'] for x in conflict_per_layer]
    strong_ratios = [x['strong_conflict_ratio'] for x in conflict_per_layer]
    
    axes[0].plot(layers, mean_sims, 'o-', linewidth=2, markersize=8)
    axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Layer Index')
    axes[0].set_ylabel('Mean Cosine Similarity')
    axes[0].set_title('Gradient Similarity Across Layers')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(layers, neg_ratios, alpha=0.7, color='orange')
    axes[1].set_xlabel('Layer Index')
    axes[1].set_ylabel('Negative Ratio')
    axes[1].set_title('Negative Gradient Ratio per Layer')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    axes[2].bar(layers, strong_ratios, alpha=0.7, color='red')
    axes[2].set_xlabel('Layer Index')
    axes[2].set_ylabel('Strong Conflict Ratio')
    axes[2].set_title('Strong Conflict Ratio per Layer')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig_path = save_dir / 'conflict_per_layer.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # 打印报告
    print(f"\n📋 各层梯度冲突报告:")
    print(f"{'Layer':<8} {'Mean Sim':<12} {'Neg Ratio':<12} {'Strong Conflict':<15}")
    print("-" * 50)
    for data in conflict_per_layer:
        print(f"{data['layer']:<8} "
              f"{data['mean_similarity']:<12.3f} "
              f"{data['negative_ratio']:<12.1%} "
              f"{data['strong_conflict_ratio']:<15.1%}")
    
    print("="*70 + "\n")
    print(f"✔ 详细分析结果已保存: {fig_path}")
    
    return conflict_per_layer
