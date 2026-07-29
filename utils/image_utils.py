#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import numpy as np
import cv2
from PIL import Image
import torchvision
import torch.nn.functional as F


def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def visualize_depth(depth, min_depth=None, max_depth=None, cmap=cv2.COLORMAP_JET):
    """
    将深度图 [1, H, W] 转换为伪彩色可视化图像 [3, H, W]
    """
    # 1. 搬到 CPU 并转为 numpy
    x = depth.detach().cpu().numpy().squeeze() # [H, W]
    
    # 2. 动态计算动态范围 (忽略极值以获得更好的对比度)
    if min_depth is None:
        min_depth = np.percentile(x, 2) # 取第 2 百分位数
    if max_depth is None:
        max_depth = np.percentile(x, 98) # 取第 98 百分位数
        
    # 3. 归一化到 [0, 1] 并裁剪
    x = np.clip(x, min_depth, max_depth)
    x = (x - min_depth) / (max_depth - min_depth + 1e-6)
    
    # 4. 映射到 [0, 255] uint8
    x_u8 = (x * 255).astype(np.uint8)
    
    # 5. 使用 OpenCV 应用伪彩色映射 (默认使用 JET，看起来很酷)
    x_color = cv2.applyColorMap(x_u8, cmap) # 结果是 BGR [H, W, 3]
    x_color = cv2.cvtColor(x_color, cv2.COLOR_BGR2RGB) # 转为 RGB
    
    # 6. 转回 torch Tensor [3, H, W] 并归一化到 [0, 1]
    return torch.from_numpy(x_color).permute(2, 0, 1).float() / 255.0


def visualize_normal(normal):
    """
    将 [-1, 1] 的法线 Tensor [3, H, W] 安全映射到 [0, 1] 的 RGB 空间
    """
    # 1. 强行对通道维 (dim=0) 做归一化，将模长锁死在 1，防止 blending 导致的数值超限
    normal_padded = F.normalize(normal.detach(), p=2, dim=0)

    # 2. 线性映射到 [0, 1]
    normal_vis = (normal_padded + 1.0) * 0.5

    # 3. 此时基本不需要 clamp，但为了安全可以保留
    return torch.clamp(normal_vis, 0.0, 1.0)

def save_tensor_image(tensor, path):
    """将 [3, H, W] 的归一化 Tensor 保存为图片"""
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(path)