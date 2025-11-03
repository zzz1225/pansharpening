import os
import random
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

# Ensure headless compatibility for Matplotlib
if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")


# =============================
# Utility helpers
# =============================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sobel_grad(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Sobel gradients and magnitude for single-channel tensors."""
    kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    ky = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=x.dtype, device=x.device).view(1, 1, 3, 3)
    gx = F.conv2d(x, kx, padding=1)
    gy = F.conv2d(x, ky, padding=1)
    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return gx, gy, grad_mag


def luminance(ms: torch.Tensor) -> torch.Tensor:
    return ms.mean(1, keepdim=True)


def norm01(t: torch.Tensor) -> torch.Tensor:
    t_min = t.amin(dim=(2, 3), keepdim=True)
    t_max = t.amax(dim=(2, 3), keepdim=True)
    return (t - t_min) / (t_max - t_min + 1e-6)


def gram_schmidt(x: torch.Tensor) -> torch.Tensor:
    """Orthogonalise basis vectors along the second dimension."""
    original_shape = x.shape
    if x.dim() > 3:
        x = x.flatten(2)
    basis: List[torch.Tensor] = []
    proj_vectors: List[torch.Tensor] = []
    for i in range(x.shape[1]):
        w = x[:, i, :]
        for proj in proj_vectors:
            w = w - proj * torch.sum(w * proj, dim=-1, keepdim=True)
        w_hat = w.detach() / (w.detach().norm(dim=-1, keepdim=True) + 1e-8)
        basis.append(w)
        proj_vectors.append(w_hat)
    x_orth = torch.stack(basis, dim=1)
    if len(original_shape) > 3:
        x_orth = x_orth.view(*original_shape)
    return x_orth


def _ensure_dir(directory: Optional[str]) -> None:
    if directory is not None:
        os.makedirs(directory, exist_ok=True)


def _save_or_show(fig: Figure, fname: Optional[str] = None, save_dir: Optional[str] = None, show: bool = True) -> None:
    backend = plt.get_backend()
    can_show = backend.lower() != "agg" and show
    if save_dir is not None and fname is not None:
        _ensure_dir(save_dir)
        path = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        print(f"[Saved] {path}")
    if can_show:
        plt.show()
    plt.close(fig)


# =============================
# Dataset definition
# =============================
class WV3H5Dataset(Dataset):
    """WV3 dataset loader backed by HDF5 files."""

    def __init__(self, h5_path: str, ratio: float = 2047.0, dtype: np.dtype = np.float32):
        super().__init__()
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"H5 file not found: {h5_path}")
        self.h5_path = h5_path
        self.ratio = ratio
        self.dtype = dtype
        with h5py.File(self.h5_path, "r") as f:
            self.has_gt = "gt" in f
            self.has_pan = "pan" in f
            self.has_ms = "ms" in f
            self.has_lms = "lms" in f
            if not (self.has_pan and self.has_ms and self.has_lms):
                raise KeyError("h5需至少包含键:'pan'、'ms'、'lms'")
            self.length = f["pan"].shape[0]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with h5py.File(self.h5_path, "r") as f:
            pan = f["pan"][idx]
            ms = f["ms"][idx]
            lms = f["lms"][idx]
            if self.has_gt:
                gt = f["gt"][idx]
            else:
                gt = np.zeros_like(lms, dtype=self.dtype)

        pan_t = torch.from_numpy(np.array(pan, dtype=self.dtype)) / self.ratio
        ms_t = torch.from_numpy(np.array(ms, dtype=self.dtype)) / self.ratio
        lms_t = torch.from_numpy(np.array(lms, dtype=self.dtype)) / self.ratio
        gt_t = torch.from_numpy(np.array(gt, dtype=self.dtype)) / self.ratio

        if pan_t.ndim == 2:
            pan_t = pan_t.unsqueeze(0)
        return pan_t, gt_t, ms_t, lms_t


# =============================
# Building blocks
# =============================
class UNet(nn.Module):
    """Compact UNet used for probabilistic component estimation."""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels_list: Tuple[int, ...] = (32, 64, 128, 256),
        bottleneck_channels: int = 512,
        min_channels_decoder: int = 64,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        ch = in_channels
        self.encoder_blocks = nn.ModuleList([])
        layers = [nn.ZeroPad2d(2), nn.Conv2d(ch, channels_list[0], 3, padding=1)]
        ch = channels_list[0]
        self.encoder_blocks.append(nn.Sequential(*layers))
        for i_level, ch_out in enumerate(channels_list):
            layers = []
            if i_level != 0:
                layers.append(nn.MaxPool2d(2))
            layers.extend(
                [
                    nn.Conv2d(ch, ch_out, 3, padding=1),
                    nn.GroupNorm(n_groups, ch_out),
                    nn.LeakyReLU(0.1),
                ]
            )
            ch = ch_out
            self.encoder_blocks.append(nn.Sequential(*layers))

        ch_out = bottleneck_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, ch_out, 3, padding=1),
            nn.GroupNorm(n_groups, ch_out),
            nn.LeakyReLU(0.1),
            nn.Conv2d(ch_out, ch_out, 3, padding=1),
            nn.GroupNorm(n_groups, ch_out),
            nn.LeakyReLU(0.1),
        )
        ch = ch_out

        self.decoder_blocks = nn.ModuleList([])
        for i_level in reversed(range(len(channels_list))):
            ch_skip = channels_list[i_level]
            ch_out = max(channels_list[i_level], min_channels_decoder)
            layers = [
                nn.Conv2d(ch + ch_skip, ch_out, 3, padding=1),
                nn.GroupNorm(n_groups, ch_out),
                nn.LeakyReLU(0.1),
            ]
            if i_level != 0:
                layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            self.decoder_blocks.append(nn.Sequential(*layers))
            ch = ch_out

        ch_skip = channels_list[0]
        self.final_conv = nn.Sequential(
            nn.Conv2d(ch + ch_skip, out_channels, 1),
            nn.ZeroPad2d(-2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcuts: List[torch.Tensor] = []
        for block in self.encoder_blocks:
            x = block(x)
            shortcuts.append(x)
        x = self.bottleneck(x)
        for block in self.decoder_blocks:
            x = torch.cat((x, shortcuts.pop()), dim=1)
            x = block(x)
        x = torch.cat((x, shortcuts.pop()), dim=1)
        return self.final_conv(x)


class EnhancedPCWrapper(nn.Module):
    """增强的不确定性估计模块，支持空间自适应和多尺度不确定性"""

    def __init__(
        self,
        n_dirs: int,
        in_channels: int,
        out_channels: int,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.net = UNet(
            in_channels=in_channels,
            out_channels=out_channels * n_dirs,
            channels_list=(32, 64, 128),
            bottleneck_channels=256,
            n_groups=8,
        )
        self.n_dirs = n_dirs
        self.out_channels = out_channels
        self.mask = mask
        
        # 空间自适应不确定性估计头
        self.sigma_conv = nn.Sequential(
            nn.Conv2d(out_channels * n_dirs, n_dirs * 2, kernel_size=3, padding=1),
            nn.GroupNorm(n_dirs, n_dirs * 2),
            nn.ReLU(),
            nn.Conv2d(n_dirs * 2, n_dirs, kernel_size=1)
        )
        
        # 全局上下文感知模块,捕获长程依赖
        self.context_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * n_dirs, n_dirs, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            w_mat_orth: [B, n_dirs, out_channels * H * W] 正交分解的方向
            sigma2_scalar: [B, n_dirs] 每个方向的标量不确定性
            sigma2_map: [B, n_dirs, H, W] 每个方向的空间不确定性图
            context_weight: [B, n_dirs, 1, 1] 全局上下文权重
        """
        x_norm = (x - 0.5) / 0.2
        w_raw = self.net(x_norm) * 0.2
        
        # 预测空间自适应的不确定性图
        sigma_map = self.sigma_conv(w_raw)
        sigma_map = F.softplus(sigma_map).clamp(min=1e-6)
        
        # 全局上下文权重
        context_weight = self.context_attention(w_raw)
        
        # 正交化处理
        w_mat = w_raw.unflatten(1, (self.n_dirs, self.out_channels))
        if self.mask is not None:
            w_mat = w_mat * self.mask
        batch, n_dirs, out_ch, H, W = w_mat.shape
        w_mat_flat = w_mat.flatten(2)
        w_mat_orth = gram_schmidt(w_mat_flat)
        
        # 标量不确定性 (用于监督损失)
        sigma2_scalar = F.adaptive_avg_pool2d(sigma_map, (1, 1)).view(batch, n_dirs)
        
        return w_mat_orth, sigma2_scalar, sigma_map, context_weight


def compute_uncertainty_map(
    w_flat: torch.Tensor, sigma2: torch.Tensor, original_shape: torch.Size
) -> torch.Tensor:
    """Compute uncertainty map from orthogonal components and sigma2."""
    batch, n_dirs, _ = w_flat.shape
    _, channels, height, width = original_shape
    w_img = w_flat.view(batch, n_dirs, channels, height, width)
    uncertainty = torch.zeros(batch, 1, height, width, device=w_flat.device)

    if sigma2.dim() == 4:
        sigma_map = sigma2
    else:
        sigma_map = sigma2.view(batch, n_dirs, 1, 1).expand(-1, -1, height, width)

    for k in range(n_dirs):
        energy = (w_img[:, k].pow(2)).sum(dim=1, keepdim=True)
        contrib = sigma_map[:, k : k + 1] * energy
        uncertainty = uncertainty + contrib
    return uncertainty


class ComplementarityAwareModule(nn.Module):
    """互补性感知模块: 显式建模PAN和MS的互补区域"""
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # 互补性检测网络
        self.complementarity_net = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels // 2, 3, padding=1),
            nn.BatchNorm2d(channels // 2),
            nn.ReLU(),
            nn.Conv2d(channels // 2, 2, 1),  # 输出2个通道: PAN互补度, MS互补度
            nn.Sigmoid()
        )
        
    def forward(self, f_pan: torch.Tensor, f_ms: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            comp_pan: [B, 1, H, W] PAN的互补性权重
            comp_ms: [B, 1, H, W] MS的互补性权重
        """
        concat = torch.cat([f_pan, f_ms], dim=1)
        comp = self.complementarity_net(concat)
        comp_pan = comp[:, 0:1, :, :]
        comp_ms = comp[:, 1:2, :, :]
        return comp_pan, comp_ms


class MultiScaleSoftAttentionFusion(nn.Module):
    """多尺度软注意力融合模块，结合不确定性和互补性"""

    def __init__(self, channels: int, n_heads: int = 4) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.channels = channels
        
        # 通道注意力 (建模特征重要性)
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels * 2, channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 2, channels * 2, 1),
        )
        
        # 空间注意力 (基于可靠性)
        self.spatial_weight = nn.Sequential(
            nn.Conv2d(4, 16, 7, padding=3),  # 输入: r_pan, r_ms, comp_pan, comp_ms
            nn.ReLU(),
            nn.Conv2d(16, 8, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 互补性感知
        self.comp_module = ComplementarityAwareModule(channels)
        
        # 融合卷积
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * 3, channels * 2, 3, padding=1),
            nn.BatchNorm2d(channels * 2),
            nn.ReLU(),
            nn.Conv2d(channels * 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )

    def forward(
        self,
        f_pan: torch.Tensor,
        f_ms: torch.Tensor,
        r_pan: torch.Tensor,
        r_ms: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch, channels, H, W = f_pan.shape
        
        # 1. 通道注意力
        cat_feat = torch.cat([f_pan, f_ms], dim=1)
        ch_attn = torch.sigmoid(self.channel_attn(cat_feat))
        ch_attn_pan = ch_attn[:, :channels]
        ch_attn_ms = ch_attn[:, channels:]
        f_pan_weighted = f_pan * ch_attn_pan
        f_ms_weighted = f_ms * ch_attn_ms

        # 2. 互补性分析
        comp_pan, comp_ms = self.comp_module(f_pan, f_ms)
        
        # 3. 标准化可靠性
        if r_pan.size(1) != 1:
            r_pan = r_pan.mean(dim=1, keepdim=True)
        if r_ms.size(1) != 1:
            r_ms = r_ms.mean(dim=1, keepdim=True)
        r_pan = norm01(r_pan)
        r_ms = norm01(r_ms)

        # 4. 综合可靠性与互补性的空间权重
        reliability_combined = torch.cat([r_pan, r_ms, comp_pan, comp_ms], dim=1)
        spatial_weight = self.spatial_weight(reliability_combined)
        
        # 5. 软融合
        fused = f_pan_weighted * spatial_weight + f_ms_weighted * (1 - spatial_weight)
        
        # 6. 残差连接原始特征
        fused = torch.cat([fused, cat_feat], dim=1)
        output = self.fusion_conv(fused)
        
        # 返回中间结果用于可视化和分析
        debug_info = {
            'spatial_weight': spatial_weight,
            'comp_pan': comp_pan,
            'comp_ms': comp_ms,
            'ch_attn_pan': ch_attn_pan,
            'ch_attn_ms': ch_attn_ms,
        }
        
        return output, debug_info


class NPPCPansharpening(nn.Module):
    """增强的NPPC全色锐化网络，强化不确定性引导融合"""

    def __init__(self, bands_ms: int = 8, base_channels: int = 64, n_dirs: int = 3) -> None:
        super().__init__()
        self.bands_ms = bands_ms
        self.n_dirs = n_dirs
        
        # Baseline网络
        self.baseline = nn.Sequential(
            nn.Conv2d(1 + bands_ms, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(base_channels, bands_ms, 3, padding=1),
        )
        
        # 特征编码器 (多尺度)
        self.encoder_pan = nn.Sequential(
            nn.Conv2d(1, base_channels // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(),
            nn.Conv2d(base_channels // 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        self.encoder_ms = nn.Sequential(
            nn.Conv2d(bands_ms, base_channels // 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(),
            nn.Conv2d(base_channels // 2, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )
        
        # 增强的不确定性估计
        self.nppc_pan = EnhancedPCWrapper(n_dirs, in_channels=1, out_channels=1)
        self.nppc_ms = EnhancedPCWrapper(n_dirs, in_channels=bands_ms, out_channels=bands_ms)
        
        # 多尺度软注意力融合
        self.fusion = MultiScaleSoftAttentionFusion(base_channels, n_heads=4)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base_channels, base_channels // 2, 3, padding=1),
            nn.BatchNorm2d(base_channels // 2),
            nn.ReLU(),
            nn.Conv2d(base_channels // 2, bands_ms, 3, padding=1),
        )

    def forward(
        self, pan: torch.Tensor, ms_up: torch.Tensor, return_uncertainty: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        _, _, height, width = pan.shape
        
        # 1. Baseline预测
        x_baseline = self.baseline(torch.cat([pan, ms_up], dim=1)).clamp(0, 1)
        
        # 2. 特征编码
        f_pan = self.encoder_pan(pan)
        f_ms = self.encoder_ms(ms_up)
        
        # 3. 不确定性估计 (增强版)
        w_pan, sigma2_pan_scalar, sigma2_pan_map, ctx_pan = self.nppc_pan(pan)
        w_ms, sigma2_ms_scalar, sigma2_ms_map, ctx_ms = self.nppc_ms(ms_up)
        
        # 4. 计算空间不确定性图
        u_pan = compute_uncertainty_map(w_pan, sigma2_pan_map, pan.shape)
        u_ms = compute_uncertainty_map(w_ms, sigma2_ms_map, ms_up.shape)
        
        # 5. 下采样到特征尺度
        u_pan_feat = F.adaptive_avg_pool2d(u_pan, f_pan.shape[-2:])
        u_ms_feat = F.adaptive_avg_pool2d(u_ms, f_ms.shape[-2:])
        
        # 6. 计算可靠性 (不确定性的倒数)
        r_pan = 1.0 / (u_pan_feat + 1e-6)
        r_ms = 1.0 / (u_ms_feat + 1e-6)
        
        # 7. 边缘增强 (PAN在边缘区域更可靠)
        _, _, grad_mag = sobel_grad(pan)
        q = torch.clip(grad_mag.flatten(2).quantile(0.8, dim=2, keepdim=True), min=1e-3)
        edge_weight = (grad_mag / q).clamp(0, 1)
        edge_weight_feat = F.adaptive_avg_pool2d(edge_weight, f_pan.shape[-2:])
        
        # 边缘区域提升PAN可靠性
        r_pan = r_pan * (1.0 + 0.5 * edge_weight_feat)
        
        # 8. 软注意力融合
        f_fused, fusion_debug = self.fusion(f_pan, f_ms, r_pan, r_ms)
        
        # 9. 解码输出
        x_out = self.decoder(f_fused).clamp(0, 1)
        
        # 10. 统计匹配
        x_out = self._match_stats(x_out, x_baseline)
        
        if return_uncertainty:
            info = {
                "x_baseline": x_baseline,
                "u_pan": u_pan,
                "u_ms": u_ms,
                "r_pan": F.interpolate(r_pan, size=(height, width), mode="bilinear", align_corners=False),
                "r_ms": F.interpolate(r_ms, size=(height, width), mode="bilinear", align_corners=False),
                "W_pan": w_pan,
                "W_ms": w_ms,
                "sigma2_pan": sigma2_pan_scalar,
                "sigma2_pan_map": sigma2_pan_map,
                "sigma2_ms": sigma2_ms_scalar,
                "sigma2_ms_map": sigma2_ms_map,
                "ctx_pan": ctx_pan,
                "ctx_ms": ctx_ms,
                "edge_weight": F.interpolate(edge_weight, size=(height, width), mode="bilinear", align_corners=False),
                # 融合过程的可视化信息
                "spatial_weight": F.interpolate(fusion_debug['spatial_weight'], size=(height, width), mode="bilinear", align_corners=False),
                "comp_pan": F.interpolate(fusion_debug['comp_pan'], size=(height, width), mode="bilinear", align_corners=False),
                "comp_ms": F.interpolate(fusion_debug['comp_ms'], size=(height, width), mode="bilinear", align_corners=False),
            }
            return x_out, info
        return x_out, None

    @torch.no_grad()
    def _match_stats(self, x: torch.Tensor, x_ref: torch.Tensor) -> torch.Tensor:
        """统计匹配,保持色彩一致性"""
        mu_x = x.mean(dim=(2, 3), keepdim=True)
        std_x = x.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        mu_ref = x_ref.mean(dim=(2, 3), keepdim=True)
        std_ref = x_ref.std(dim=(2, 3), keepdim=True).clamp(min=1e-6)
        return (mu_ref + (std_ref / std_x) * (x - mu_x)).clamp(0, 1)


# =============================
# Metrics & Losses
# =============================
def sam_metric(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xf, yf = x.flatten(2), y.flatten(2)
    num = (xf * yf).sum(1)
    den = torch.sqrt((xf * xf).sum(1) + eps) * torch.sqrt((yf * yf).sum(1) + eps)
    sam_rad = torch.acos(torch.clamp(num / (den + eps), -1 + 1e-6, 1 - 1e-6))
    return torch.rad2deg(sam_rad).mean()


def ergas_metric(x: torch.Tensor, y: torch.Tensor, ratio: int = 4, eps: float = 1e-8) -> torch.Tensor:
    mean_y = y.mean(dim=(2, 3), keepdim=True)
    mse_per_band = ((x - y) ** 2).mean(dim=(2, 3))
    mean_y_per_band = mean_y.squeeze(2).squeeze(2)
    ergas = 100.0 * ratio * torch.sqrt((mse_per_band / (mean_y_per_band ** 2 + eps)).mean(dim=1))
    return ergas.mean()


def q8_metric(x: torch.Tensor, y: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    _, channels, _, _ = x.shape
    q_values = []
    for c in range(channels):
        x_c = x[:, c : c + 1, :, :]
        y_c = y[:, c : c + 1, :, :]
        mu_x = F.avg_pool2d(x_c, block_size, stride=block_size)
        mu_y = F.avg_pool2d(y_c, block_size, stride=block_size)
        sigma_x = F.avg_pool2d(x_c ** 2, block_size, stride=block_size) - mu_x ** 2
        sigma_y = F.avg_pool2d(y_c ** 2, block_size, stride=block_size) - mu_y ** 2
        sigma_xy = F.avg_pool2d(x_c * y_c, block_size, stride=block_size) - mu_x * mu_y
        numerator = 4 * sigma_xy * mu_x * mu_y
        denominator = (sigma_x + sigma_y) * (mu_x ** 2 + mu_y ** 2) + 1e-8
        q = numerator / denominator
        q_values.append(q.mean())
    return torch.stack(q_values).mean()


def sam_loss(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    xf, yf = x.flatten(2), y.flatten(2)
    num = (xf * yf).sum(1)
    den = torch.sqrt((xf * xf).sum(1) + eps) * torch.sqrt((yf * yf).sum(1) + eps)
    return torch.acos(torch.clamp(num / (den + eps), -1 + 1e-6, 1 - 1e-6)).mean()


def nppc_supervision_loss(
    w_flat: torch.Tensor, sigma2_pred: torch.Tensor, residual_flat: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """NPPC监督损失: 最大化投影能量 + 方差预测准确性"""
    proj = torch.einsum("bkd,bd->bk", w_flat.detach(), residual_flat)
    proj_energy = proj ** 2
    l_dir = -(torch.einsum("bkd,bd->bk", w_flat, residual_flat) ** 2).mean()
    l_var = ((sigma2_pred - proj_energy) ** 2).mean()
    return l_dir, l_var


def uncertainty_calibration_loss(
    u_pred: torch.Tensor, residual: torch.Tensor, reduce: str = 'mean'
) -> torch.Tensor:
    """不确定性校准损失: 预测的不确定性应匹配实际误差"""
    squared_error = residual ** 2
    if residual.size(1) > 1:
        squared_error = squared_error.mean(dim=1, keepdim=True)
    loss = F.mse_loss(u_pred, squared_error, reduction=reduce)
    return loss


def complementarity_loss(
    comp_pan: torch.Tensor, 
    comp_ms: torch.Tensor,
    r_pan: torch.Tensor,
    r_ms: torch.Tensor
) -> torch.Tensor:
    """互补性损失: 鼓励在不同区域选择可靠的模态"""
    # ---- 修复：把 r_pan / r_ms 统一到单通道，避免 [B,1,H,W] vs [B,2,H,W] 广播 ----
    if r_pan.dim() == 4 and r_pan.size(1) != 1:
        r_pan = r_pan.mean(dim=1, keepdim=True)
    if r_ms.dim() == 4 and r_ms.size(1) != 1:
        r_ms = r_ms.mean(dim=1, keepdim=True)

    # 在PAN不可靠的地方，MS互补性应该高
    loss_pan = F.mse_loss(comp_ms, 1.0 - r_pan)
    # 在MS不可靠的地方，PAN互补性应该高
    loss_ms = F.mse_loss(comp_pan, 1.0 - r_ms)
    return (loss_pan + loss_ms) * 0.5


# =============================
# Visualisation helpers
# =============================

# ---- 修复：把 (C,H,W) 或 (1,H,W) 安全转换为 (H,W) 供 imshow 使用 ----
def _to_img2d(arr: np.ndarray) -> np.ndarray:
    """Accepts (H,W), (1,H,W), or (C,H,W). Returns (H,W)"""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        return arr.mean(axis=0)  # 多通道聚合为单通道热力图
    raise ValueError(f"Unexpected shape for image data: {arr.shape}")

def plot_training_log(log_list: List[Dict[str, float]], save_dir: Optional[str] = None, show: bool = True) -> None:
    if not log_list:
        print("No logs to plot")
        return
    steps = [d["step"] for d in log_list]
    
    # 图1: 主要重建指标
    fig1 = plt.figure(figsize=(15, 4))
    ax = fig1.add_subplot(1, 3, 1)
    ax.plot(steps, [d["l1"] for d in log_list], label="Fusion Output", linewidth=2, color='#2E86DE')
    ax.plot(steps, [d["x_baseline_l1"] for d in log_list], label="Baseline", linewidth=2, linestyle="--", color='#EE5A6F')
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("L1 Error", fontsize=11)
    ax.set_title("L1 Reconstruction Error", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax = fig1.add_subplot(1, 3, 2)
    ax.plot(steps, [d["sam"] for d in log_list], label="SAM", linewidth=2, color='#10AC84')
    ax.set_xlabel("Step", fontsize=11)
    ax.set_ylabel("SAM (degrees)", fontsize=11)
    ax.set_title("Spectral Angle Mapper", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax1 = fig1.add_subplot(1, 3, 3)
    color1 = '#F79F1F'
    ax1.plot(steps, [d["ergas"] for d in log_list], label="ERGAS", linewidth=2, color=color1)
    ax1.set_xlabel("Step", fontsize=11)
    ax1.set_ylabel("ERGAS", fontsize=11, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("ERGAS & Q8 Metrics", fontsize=12, fontweight='bold')
    
    ax2 = ax1.twinx()
    color2 = '#5F27CD'
    ax2.plot(steps, [d["q8"] for d in log_list], label="Q8", linewidth=2, color=color2)
    ax2.set_ylabel("Q8", fontsize=11, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    fig1.tight_layout()
    _save_or_show(fig1, "training_metrics.png", save_dir, show)

    # 图2: 不确定性演化
    fig2 = plt.figure(figsize=(12, 5))
    ax = fig2.add_subplot(1, 1, 1)
    ax.plot(steps, [d["u_pan"] for d in log_list], label="PAN Uncertainty", linewidth=2.5, color='#FDA7DF')
    ax.plot(steps, [d["u_ms"] for d in log_list], label="MS Uncertainty", linewidth=2.5, color='#82CCDD')
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Mean Uncertainty", fontsize=12)
    ax.set_title("Uncertainty Evolution During Training", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig2.tight_layout()
    _save_or_show(fig2, "uncertainty_evolution.png", save_dir, show)

    # 图3: 可靠性权重
    fig3 = plt.figure(figsize=(12, 5))
    ax = fig3.add_subplot(1, 1, 1)
    ax.plot(steps, [d["r_pan"] for d in log_list], label="PAN Reliability", linewidth=2.5, color='#FF6B6B')
    ax.plot(steps, [d["r_ms"] for d in log_list], label="MS Reliability", linewidth=2.5, color='#4ECDC4')
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Mean Reliability", fontsize=12)
    ax.set_title("Reliability Weights Evolution", fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    fig3.tight_layout()
    _save_or_show(fig3, "reliability_weights.png", save_dir, show)
    
    # 图4: 互补性分析 (如果有数据)
    if "comp_pan" in log_list[0]:
        fig4 = plt.figure(figsize=(12, 5))
        ax = fig4.add_subplot(1, 1, 1)
        ax.plot(steps, [d["comp_pan"] for d in log_list], label="PAN Complementarity", linewidth=2.5, color='#A29BFE')
        ax.plot(steps, [d["comp_ms"] for d in log_list], label="MS Complementarity", linewidth=2.5, color='#FD79A8')
        ax.set_xlabel("Step", fontsize=12)
        ax.set_ylabel("Complementarity Score", fontsize=12)
        ax.set_title("Modality Complementarity Evolution", fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, linestyle='--')
        fig4.tight_layout()
        _save_or_show(fig4, "complementarity_evolution.png", save_dir, show)


def visualize_results(
    pan: torch.Tensor,
    ms: torch.Tensor,
    gt: torch.Tensor,
    output: torch.Tensor,
    info: Dict[str, torch.Tensor],
    device: str,
    save_dir: Optional[str] = None,
    prefix: str = "sample",
    show: bool = True,
) -> None:
    def to_np(t: torch.Tensor) -> np.ndarray:
        return t.detach().cpu().numpy()

    pan_np = to_np(pan).squeeze()
    ms_np = to_np(ms).transpose(1, 2, 0)
    gt_np = to_np(gt).transpose(1, 2, 0)
    output_np = to_np(output).transpose(1, 2, 0)
    baseline_np = to_np(info["x_baseline"][0]).transpose(1, 2, 0)

    # ---- 使用安全转换，避免 (2,H,W) 直接传入 imshow ----
    u_pan_np = _to_img2d(to_np(info["u_pan"][0]))
    u_ms_np  = _to_img2d(to_np(info["u_ms"][0]))
    r_pan_np = _to_img2d(to_np(info["r_pan"][0]))
    r_ms_np  = _to_img2d(to_np(info["r_ms"][0]))

    # 图1: 主要结果对比
    fig1, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes[0, 0].imshow(pan_np, cmap="gray")
    axes[0, 0].set_title("PAN", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(ms_np[:, :, :3])
    axes[0, 1].set_title("MS-up", fontsize=12, fontweight='bold')
    axes[0, 1].axis("off")
    
    axes[0, 2].imshow(gt_np[:, :, :3])
    axes[0, 2].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[0, 2].axis("off")
    
    axes[1, 0].imshow(baseline_np[:, :, :3])
    axes[1, 0].set_title("Baseline", fontsize=12, fontweight='bold')
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(output_np[:, :, :3])
    axes[1, 1].set_title("NPPC (Ours)", fontsize=12, fontweight='bold')
    axes[1, 1].axis("off")
    
    err = np.abs(output_np - gt_np).mean(axis=2)
    im = axes[1, 2].imshow(err, cmap="hot")
    axes[1, 2].set_title("Absolute Error", fontsize=12, fontweight='bold')
    axes[1, 2].axis("off")
    fig1.colorbar(im, ax=axes[1, 2], fraction=0.046)
    fig1.tight_layout()
    _save_or_show(fig1, f"{prefix}_results.png", save_dir, show)

    # 图2: 不确定性和可靠性
    fig2, axes = plt.subplots(2, 2, figsize=(12, 10))
    im1 = axes[0, 0].imshow(u_pan_np, cmap="viridis")
    axes[0, 0].set_title("PAN Uncertainty", fontsize=12, fontweight='bold')
    axes[0, 0].axis("off")
    fig2.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    im2 = axes[0, 1].imshow(u_ms_np, cmap="viridis")
    axes[0, 1].set_title("MS Uncertainty", fontsize=12, fontweight='bold')
    axes[0, 1].axis("off")
    fig2.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    im3 = axes[1, 0].imshow(r_pan_np, cmap="plasma")
    axes[1, 0].set_title("PAN Reliability", fontsize=12, fontweight='bold')
    axes[1, 0].axis("off")
    fig2.colorbar(im3, ax=axes[1, 0], fraction=0.046)

    im4 = axes[1, 1].imshow(r_ms_np, cmap="plasma")
    axes[1, 1].set_title("MS Reliability", fontsize=12, fontweight='bold')
    axes[1, 1].axis("off")
    fig2.colorbar(im4, ax=axes[1, 1], fraction=0.046)

    fig2.tight_layout()
    _save_or_show(fig2, f"{prefix}_uncertainty.png", save_dir, show)
    
    # 图3: 融合过程可视化
    if "spatial_weight" in info and "comp_pan" in info:
        spatial_w_np = _to_img2d(to_np(info["spatial_weight"][0]))
        comp_pan_np  = _to_img2d(to_np(info["comp_pan"][0]))
        comp_ms_np   = _to_img2d(to_np(info["comp_ms"][0]))
        
        fig3, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = axes[0].imshow(spatial_w_np, cmap="RdYlGn", vmin=0, vmax=1)
        axes[0].set_title("Spatial Fusion Weight\n(1=PAN, 0=MS)", fontsize=12, fontweight='bold')
        axes[0].axis("off")
        fig3.colorbar(im1, ax=axes[0], fraction=0.046)
        
        im2 = axes[1].imshow(comp_pan_np, cmap="YlOrRd", vmin=0, vmax=1)
        axes[1].set_title("PAN Complementarity", fontsize=12, fontweight='bold')
        axes[1].axis("off")
        fig3.colorbar(im2, ax=axes[1], fraction=0.046)
        
        im3 = axes[2].imshow(comp_ms_np, cmap="YlGnBu", vmin=0, vmax=1)
        axes[2].set_title("MS Complementarity", fontsize=12, fontweight='bold')
        axes[2].axis("off")
        fig3.colorbar(im3, ax=axes[2], fraction=0.046)
        
        fig3.tight_layout()
        _save_or_show(fig3, f"{prefix}_fusion_analysis.png", save_dir, show)
    
    # 图4: 边缘和可靠性叠加分析
    if "edge_weight" in info:
        edge_np = _to_img2d(to_np(info["edge_weight"][0]))
        
        fig4, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        im1 = axes[0].imshow(edge_np, cmap="gray")
        axes[0].set_title("Edge Strength", fontsize=12, fontweight='bold')
        axes[0].axis("off")
        fig4.colorbar(im1, ax=axes[0], fraction=0.046)
        
        # 可靠性差异图 (PAN更可靠的区域)
        reliability_diff = r_pan_np - r_ms_np
        im2 = axes[1].imshow(reliability_diff, cmap="RdBu", vmin=-1, vmax=1)
        axes[1].set_title("Reliability Difference\n(Red=PAN better, Blue=MS better)", fontsize=12, fontweight='bold')
        axes[1].axis("off")
        fig4.colorbar(im2, ax=axes[1], fraction=0.046)
        
        # 不确定性差异图
        uncertainty_diff = u_ms_np - u_pan_np
        im3 = axes[2].imshow(uncertainty_diff, cmap="RdYlGn")
        axes[2].set_title("Uncertainty Difference\n(Red=MS more uncertain)", fontsize=12, fontweight='bold')
        axes[2].axis("off")
        fig4.colorbar(im3, ax=axes[2], fraction=0.046)
        
        fig4.tight_layout()
        _save_or_show(fig4, f"{prefix}_diagnostic.png", save_dir, show)
