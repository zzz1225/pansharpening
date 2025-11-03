import os
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from net import (
    NPPCPansharpening,
    WV3H5Dataset,
    compute_uncertainty_map,
    luminance,
    plot_training_log,
    sam_loss,
    sam_metric,
    ergas_metric,
    q8_metric,
    nppc_supervision_loss,
    uncertainty_calibration_loss,
    complementarity_loss,
    set_seed,
    visualize_results,
)

# =============================
# Metrics & Loss helpers
# =============================

def l1(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return (x - y).abs().mean()


def perceptual_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """简单的感知损失: 梯度域相似性"""
    def grad_loss(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        gx_a = a[:, :, :, :-1] - a[:, :, :, 1:]
        gx_b = b[:, :, :, :-1] - b[:, :, :, 1:]
        gy_a = a[:, :, :-1, :] - a[:, :, 1:, :]
        gy_b = b[:, :, :-1, :] - b[:, :, 1:, :]
        return (gx_a - gx_b).abs().mean() + (gy_a - gy_b).abs().mean()
    return grad_loss(x, y)

# =============================
# Training routine (增强版)
# =============================

def train_nppc_pansharp_wv3(
    train_h5: str = "/root/nas-public-linkdata/datasets/pansharpening/WV3/training_wv3/train_wv3.h5",
    test_h5: str = "/root/nas-public-linkdata/datasets/pansharpening/WV3/reduced_examples/test_wv3_multiExm1.h5",
    ratio: float = 2047.0,
    bands_ms: int = 8,
    device: str = "cuda",
    epochs: Tuple[int, int, int] = (5, 5, 10),
    batch_size: int = 2,
    n_dirs: int = 3,
    base_channels: int = 64,
    num_workers: int = 4,
    subset_ratio: float = 0.2,
    accumulation_steps: int = 2,
    out_dir: str = "./outputs",
    show_fig: bool = True,
) -> Tuple[NPPCPansharpening, List[Dict[str, float]], Dict[str, List[float]], DataLoader]:
    set_seed(123)

    train_set_full = WV3H5Dataset(train_h5, ratio=ratio)
    test_set_full = WV3H5Dataset(test_h5, ratio=ratio)

    train_size = max(1, int(len(train_set_full) * subset_ratio))
    test_size = max(1, min(int(len(test_set_full) * subset_ratio), len(test_set_full)))
    train_indices = list(range(train_size))
    test_indices = list(range(test_size))

    train_set = Subset(train_set_full, train_indices)
    test_set = Subset(test_set_full, test_indices)

    print(f"使用训练样本: {len(train_set)}/{len(train_set_full)}")
    print(f"使用测试样本: {len(test_set)}/{len(test_set_full)}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(1, num_workers // 2),
        pin_memory=True,
    )

    model = NPPCPansharpening(
        bands_ms=bands_ms, base_channels=base_channels, n_dirs=n_dirs
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    log_list: List[Dict[str, float]] = []
    global_step = 0

    # =============================
    # Stage 1: Baseline网络预训练
    # =============================
    print("=" * 80)
    print(" Stage 1: Training Baseline Fusion Network ".center(80))
    print("=" * 80)
    model.train()
    for epoch in range(epochs[0]):
        epoch_loss = 0.0
        for batch_idx, (pan, gt, ms, lms) in enumerate(train_loader):
            pan = pan.to(device)
            gt = gt.to(device)
            ms_up = lms.to(device)

            x_base = model.baseline(torch.cat([pan, ms_up], dim=1)).clamp(0, 1)
            
            # 组合损失: L1 + SAM + 感知损失
            loss_l1 = l1(x_base, gt)
            loss_sam = sam_loss(x_base, gt)
            loss_percept = perceptual_loss(x_base, gt)
            loss = loss_l1 + 0.2 * loss_sam + 0.1 * loss_percept
            
            loss = loss / accumulation_steps
            loss.backward()
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % 20 == 0:
                    with torch.no_grad():
                        sam_val = sam_metric(x_base, gt)
                        ergas_val = ergas_metric(x_base, gt)
                        q8_val = q8_metric(x_base, gt)
                    print(
                        f"[Stage1 | Epoch {epoch+1}/{epochs[0]}] step {global_step} | "
                        f"loss={loss.item() * accumulation_steps:.4f} | "
                        f"SAM={sam_val:.4f}° | ERGAS={ergas_val:.4f} | Q8={q8_val:.4f}"
                    )

            del pan, gt, ms_up, x_base, loss
            if device == "cuda":
                torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"[Stage1] Epoch {epoch+1}/{epochs[0]} completed. Avg Loss: {avg_loss:.4f}")

    # =============================
    # Stage 2: 不确定性估计模块训练
    # =============================
    print("\n" + "=" * 80)
    print(" Stage 2: Training NPPC Uncertainty Estimation ".center(80))
    print("=" * 80)

    # 冻结baseline
    for param in model.baseline.parameters():
        param.requires_grad = False
    
    # 只优化不确定性模块
    optimizer = torch.optim.Adam(
        list(model.nppc_pan.parameters()) + list(model.nppc_ms.parameters()), 
        lr=1e-3
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs[1] * len(train_loader), eta_min=1e-5
    )

    for epoch in range(epochs[1]):
        epoch_loss = 0.0
        for batch_idx, (pan, gt, ms, lms) in enumerate(train_loader):
            pan = pan.to(device)
            gt = gt.to(device)
            ms_up = lms.to(device)

            with torch.no_grad():
                x_base = model.baseline(torch.cat([pan, ms_up], dim=1)).clamp(0, 1)

            # 计算残差 (作为不确定性的监督信号)
            res_pan = luminance(gt) - luminance(x_base)
            res_ms = gt - x_base

            # 前向传播不确定性模块
            w_pan, s2_pan, s2_pan_map, ctx_pan = model.nppc_pan(pan)
            w_ms, s2_ms, s2_ms_map, ctx_ms = model.nppc_ms(ms_up)

            res_pan_flat = res_pan.flatten(1)
            res_ms_flat = res_ms.flatten(1)

            # NPPC监督损失 (方向和方差)
            ld_pan, lv_pan = nppc_supervision_loss(w_pan, s2_pan, res_pan_flat)
            ld_ms, lv_ms = nppc_supervision_loss(w_ms, s2_ms, res_ms_flat)

            # 计算不确定性图
            u_pan = compute_uncertainty_map(w_pan, s2_pan_map, pan.shape)
            u_ms = compute_uncertainty_map(w_ms, s2_ms_map, ms_up.shape)

            # 不确定性校准损失 (预测的不确定性应匹配实际误差)
            l_cal_pan = uncertainty_calibration_loss(u_pan, res_pan)
            l_cal_ms = uncertainty_calibration_loss(u_ms, res_ms.mean(1, keepdim=True))

            # 组合损失
            loss = (ld_pan + ld_ms) + 0.5 * (lv_pan + lv_ms) + 0.2 * (l_cal_pan + l_cal_ms)
            loss = loss / accumulation_steps
            loss.backward()
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    list(model.nppc_pan.parameters()) + list(model.nppc_ms.parameters()), 
                    max_norm=1.0
                )
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                if global_step % 20 == 0:
                    print(
                        f"[Stage2 | Epoch {epoch+1}/{epochs[1]}] step {global_step} | "
                        f"loss={loss.item() * accumulation_steps:.4f} | "
                        f"u_pan={u_pan.mean().item():.4f} | u_ms={u_ms.mean().item():.4f}"
                    )

            del (
                pan, gt, ms_up, x_base, res_pan, res_ms,
                w_pan, w_ms, s2_pan, s2_ms, u_pan, u_ms,
                res_pan_flat, res_ms_flat, loss,
            )
            if device == "cuda":
                torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"[Stage2] Epoch {epoch+1}/{epochs[1]} completed. Avg Loss: {avg_loss:.4f}")

    # =============================
    # Stage 3: 端到端微调 (不确定性引导融合)
    # =============================
    print("\n" + "=" * 80)
    print(" Stage 3: End-to-End Fusion Training with Uncertainty Guidance ".center(80))
    print("=" * 80)

    # 解冻所有参数
    for param in model.parameters():
        param.requires_grad = True
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs[2] * len(train_loader), eta_min=1e-6
    )

    for epoch in range(epochs[2]):
        epoch_loss = 0.0
        for batch_idx, (pan, gt, ms, lms) in enumerate(train_loader):
            pan = pan.to(device)
            gt = gt.to(device)
            ms_up = lms.to(device)

            # 前向传播完整模型
            x_out, info = model(pan, ms_up, return_uncertainty=True)

            # 主要重建损失
            loss_rec = l1(x_out, gt) + 0.2 * sam_loss(x_out, gt) + 0.1 * perceptual_loss(x_out, gt)
            
            # 统计一致性损失 (保持色彩分布)
            loss_stat = 0.05 * l1(x_out.mean((2, 3)), info["x_baseline"].mean((2, 3)))
            
            # 互补性损失 (鼓励在不同区域选择可靠模态)
            if "comp_pan" in info and "comp_ms" in info:
                loss_comp = 0.1 * complementarity_loss(
                    info["comp_pan"], info["comp_ms"], 
                    info["r_pan"], info["r_ms"]
                )
            else:
                loss_comp = torch.tensor(0.0, device=device)
            
            # 组合损失
            loss = (loss_rec + loss_stat + loss_comp) / accumulation_steps
            loss.backward()
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                
                with torch.no_grad():
                    sam_val = sam_metric(x_out, gt)
                    ergas_val = ergas_metric(x_out, gt)
                    q8_val = q8_metric(x_out, gt)
                    
                    metrics = {
                        "step": global_step,
                        "l1": float(l1(x_out, gt)),
                        "sam": float(sam_val),
                        "ergas": float(ergas_val),
                        "q8": float(q8_val),
                        "x_baseline_l1": float(l1(info["x_baseline"], gt)),
                        "u_pan": float(info["u_pan"].mean()),
                        "u_ms": float(info["u_ms"].mean()),
                        "r_pan": float(info["r_pan"].mean()),
                        "r_ms": float(info["r_ms"].mean()),
                    }
                    
                    # 记录互补性指标
                    if "comp_pan" in info:
                        metrics["comp_pan"] = float(info["comp_pan"].mean())
                        metrics["comp_ms"] = float(info["comp_ms"].mean())
                    
                    log_list.append(metrics)
                    
                    if global_step % 20 == 0:
                        print(
                            f"[Stage3 | Epoch {epoch+1}/{epochs[2]}] step {global_step} | "
                            f"L1={metrics['l1']:.4f} | SAM={metrics['sam']:.4f}° | "
                            f"ERGAS={metrics['ergas']:.4f} | Q8={metrics['q8']:.4f}"
                        )

            del pan, gt, ms_up, x_out, info, loss
            if device == "cuda":
                torch.cuda.empty_cache()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"[Stage3] Epoch {epoch+1}/{epochs[2]} completed. Avg Loss: {avg_loss:.4f}")

    print("\n" + "=" * 80)
    print(" Training Completed! ".center(80))
    print("=" * 80)

    # =============================
    # 测试集评估
    # =============================
    print("\nEvaluating on test set...")
    model.eval()
    test_metrics: Dict[str, List[float]] = {
        "l1": [], "sam": [], "ergas": [], "q8": [],
        "baseline_l1": [], "baseline_sam": [], "baseline_ergas": []
    }
    
    with torch.no_grad():
        for pan, gt, ms, lms in test_loader:
            pan = pan.to(device)
            gt = gt.to(device)
            ms_up = lms.to(device)
            
            x_out, info = model(pan, ms_up, return_uncertainty=True)
            x_base = info["x_baseline"]
            
            # 评估融合输出
            test_metrics["l1"].append(float(l1(x_out, gt)))
            test_metrics["sam"].append(float(sam_metric(x_out, gt)))
            test_metrics["ergas"].append(float(ergas_metric(x_out, gt)))
            test_metrics["q8"].append(float(q8_metric(x_out, gt)))
            
            # 评估baseline (用于对比)
            test_metrics["baseline_l1"].append(float(l1(x_base, gt)))
            test_metrics["baseline_sam"].append(float(sam_metric(x_base, gt)))
            test_metrics["baseline_ergas"].append(float(ergas_metric(x_base, gt)))

    print("\n" + "=" * 80)
    print(" Test Set Results ".center(80))
    print("=" * 80)
    print("\nProposed Method (NPPC):")
    for key in ["l1", "sam", "ergas", "q8"]:
        values = np.array(test_metrics[key])
        if key == "sam":
            print(f"  {key.upper()}: {values.mean():.4f}° ± {values.std():.4f}°")
        else:
            print(f"  {key.upper()}: {values.mean():.4f} ± {values.std():.4f}")
    
    print("\nBaseline Method:")
    for key in ["baseline_l1", "baseline_sam", "baseline_ergas"]:
        values = np.array(test_metrics[key])
        metric_name = key.replace("baseline_", "").upper()
        if "sam" in key:
            print(f"  {metric_name}: {values.mean():.4f}° ± {values.std():.4f}°")
        else:
            print(f"  {metric_name}: {values.mean():.4f} ± {values.std():.4f}")
    
    # 计算改进百分比
    print("\nImprovement over Baseline:")
    l1_improve = (np.mean(test_metrics["baseline_l1"]) - np.mean(test_metrics["l1"])) / np.mean(test_metrics["baseline_l1"]) * 100
    sam_improve = (np.mean(test_metrics["baseline_sam"]) - np.mean(test_metrics["sam"])) / np.mean(test_metrics["baseline_sam"]) * 100
    ergas_improve = (np.mean(test_metrics["baseline_ergas"]) - np.mean(test_metrics["ergas"])) / np.mean(test_metrics["baseline_ergas"]) * 100
    print(f"  L1: {l1_improve:+.2f}% | SAM: {sam_improve:+.2f}% | ERGAS: {ergas_improve:+.2f}%")
    print("=" * 80)

    # 保存模型
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "nppc_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_metrics': test_metrics,
        'config': {
            'bands_ms': bands_ms,
            'base_channels': base_channels,
            'n_dirs': n_dirs,
        }
    }, model_path)
    print(f"\n[Model Saved] {model_path}")

    # 绘制训练曲线
    plot_training_log(log_list, save_dir=out_dir, show=show_fig)

    # 可视化测试样本
    print("\nGenerating visualizations...")
    with torch.no_grad():
        pan, gt, ms, lms = next(iter(test_loader))
        pan = pan.to(device)
        gt = gt.to(device)
        ms_up = lms.to(device)
        x_out, info = model(pan, ms_up, return_uncertainty=True)
        visualize_results(
            pan[0], ms_up[0], gt[0], x_out[0], info,
            device, save_dir=out_dir, prefix="test_sample", show=show_fig
        )

    return model, log_list, test_metrics, test_loader


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    # 加载配置文件
    with open("super_para.yml") as f:
        cfg = yaml.safe_load(f)

    # 解析配置
    train_h5 = cfg.get("train_h5", "/root/nas-public-linkdata/datasets/pansharpening/WV3/training_wv3/train_wv3.h5")
    test_h5 = cfg.get("test_h5", "/root/nas-public-linkdata/datasets/pansharpening/WV3/reduced_examples/test_wv3_multiExm1.h5")
    ratio = cfg.get("ratio", 2047.0)
    bands_ms = cfg.get("bands_ms", cfg.get("ms_target_channel", 8))
    
    epochs_cfg = cfg.get("epochs", (5, 5, 10))
    if isinstance(epochs_cfg, int):
        epochs = (epochs_cfg, epochs_cfg, epochs_cfg)
    elif isinstance(epochs_cfg, (list, tuple)):
        if len(epochs_cfg) == 3:
            epochs = tuple(epochs_cfg)
        else:
            raise ValueError("epochs must be an int or a sequence of length 3")
    else:
        raise TypeError("epochs configuration must be int, list, or tuple")

    batch_size = cfg.get("batch_size", 2)
    n_dirs = cfg.get("n_dirs", 3)
    base_channels = cfg.get("base_channels", 64)
    num_workers = cfg.get("num_workers", 4)
    subset_ratio = cfg.get("subset_ratio", 0.4)
    out_dir = cfg.get("out_dir", "./outputs")
    show_fig = cfg.get("show_fig", True)

    # 开始训练
    train_nppc_pansharp_wv3(
        train_h5=train_h5,
        test_h5=test_h5,
        ratio=ratio,
        bands_ms=bands_ms,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        n_dirs=n_dirs,
        base_channels=base_channels,
        num_workers=num_workers,
        subset_ratio=subset_ratio,
        out_dir=out_dir,
        show_fig=show_fig,
    )
