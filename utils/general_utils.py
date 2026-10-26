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
import sys
import os
import random
import numpy as np
from datetime import datetime
import struct
import cv2
import matplotlib as mpl
import matplotlib.cm as cm

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp
    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])
    q = r / norm[:, None]
    R = torch.zeros((q.size(0), 3, 3), device='cuda')
    r, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)
    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]
    return R @ L

def safe_state(silent):
    old_f = sys.stdout
    class F:
        def __init__(self, silent): self.silent = silent
        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else: old_f.write(x)
        def flush(self): old_f.flush()
    sys.stdout = F(silent)
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device("cuda:0"))


def vis_depth(depth, minmax=None, cmap='magma_r'):
    """Convert a depth map into a color visualization."""
    depthmap = np.nan_to_num(depth)
    # Exclude depth values outside the valid visualization range.
    valid_mask = depthmap < 300
    if valid_mask.sum() > 10:
        constant_max = np.percentile(depthmap[valid_mask], 99)
        constant_min = np.percentile(depthmap[valid_mask], 1)
    else:
        constant_max, constant_min = 60, 0
    
    if minmax is not None: # Use the specified depth range.
        constant_min, constant_max = minmax

    normalizer = mpl.colors.Normalize(vmin=constant_min, vmax=constant_max)
    mapper = cm.ScalarMappable(norm=normalizer, cmap=cmap)
    depth_vis_color = (mapper.to_rgba(depthmap)[:, :, :3] * 255).astype(np.uint8)
    return depth_vis_color, [constant_min, constant_max]

def readDepthDmb(file_path):
    with open(file_path, "rb") as f:
        type_id = struct.unpack("i", f.read(4))[0]
        h, w, nb = struct.unpack("iii", f.read(12))
        if type_id != 1: return -1
        depth = np.frombuffer(f.read(h * w * nb * 4), dtype=np.float32).reshape((h, w)).copy()
    return depth

def readNormalDmb(file_path):
    try:
        with open(file_path, 'rb') as f:
            type_id = np.fromfile(f, dtype=np.int32, count=1)[0]
            h, w, nb = np.fromfile(f, dtype=np.int32, count=3)
            if type_id != 1: return -1
            normal = np.fromfile(f, dtype=np.float32, count=h*w*nb).reshape((h, w, nb))
            return normal[:, :, :3]
    except Exception: return -1

def read_propagted_depth(path):    
    cost = readDepthDmb(os.path.join(path, 'costs.dmb'))
    if isinstance(cost, np.ndarray):
        cost[np.isnan(cost)] = 2
        cost[cost < 0] = 2
    
    depth = readDepthDmb(os.path.join(path, 'depths.dmb'))
    if isinstance(depth, np.ndarray):
        depth[np.isnan(depth)] = 300
        depth[depth <= 0] = 300
        depth[depth > 300] = 300
    
    normal = readNormalDmb(os.path.join(path, 'normals.dmb'))
    return depth, cost, normal

def load_pairs_relation(path):
    pairs_relation = []
    try:
        with open(path, 'r') as f:
            num_images = int(f.readline())
            for i in range(num_images):
                ref_id = int(f.readline())
                src_infos = f.readline().split()
                num_src = int(src_infos[0])
                pairs = []
                for j in range(num_src):
                    sid, score = int(src_infos[2*j+1]), float(src_infos[2*j+2])
                    if score <= 0.0 or sid % 8 == 0: continue
                    # Remap image IDs after excluding every eighth image; this assumes the corresponding train/test split.
                    sid_aligned = (sid // 8) * 7 + (sid % 8) - 1
                    pairs.append(sid_aligned)
                    if len(pairs) > 3: break
                if ref_id % 8 != 0: pairs_relation.append(pairs)
    except Exception: pass
    return pairs_relation