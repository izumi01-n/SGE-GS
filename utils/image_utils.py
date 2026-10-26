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
    """Convert a depth map to a normalized pseudocolor image."""
    x = depth.detach().cpu().numpy().squeeze()

    if min_depth is None:
        min_depth = np.percentile(x, 2)
    if max_depth is None:
        max_depth = np.percentile(x, 98)

    x = np.clip(x, min_depth, max_depth)
    x = (x - min_depth) / (max_depth - min_depth + 1e-6)

    x_u8 = (x * 255).astype(np.uint8)

    x_color = cv2.applyColorMap(x_u8, cmap)
    x_color = cv2.cvtColor(x_color, cv2.COLOR_BGR2RGB)

    return torch.from_numpy(x_color).permute(2, 0, 1).float() / 255.0


def visualize_normal(normal):
    """Map surface normals from [-1, 1] to RGB values in [0, 1]."""
    normal_padded = F.normalize(normal.detach(), p=2, dim=0)
    normal_vis = (normal_padded + 1.0) * 0.5
    return torch.clamp(normal_vis, 0.0, 1.0)

def save_tensor_image(tensor, path):
    """Save a normalized image tensor [3, H, W] to disk."""
    ndarr = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(path)