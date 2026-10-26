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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C


def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)


def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )


class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        args = (
            raster_settings.bg,
            means3D,
            colors_precomp,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.kernel_size,
            raster_settings.subpixel_offset,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug
        )

        if raster_settings.debug:
            try:
                results = _C.rasterize_gaussians(*args)
                num_rendered, color, depth, alpha, normal, radii, geomBuffer, binningBuffer, imgBuffer = results
            except Exception as ex:
                print("\nAn error occured in forward.")
                raise ex
        else:
            num_rendered, color, depth, alpha, normal, radii, geomBuffer, binningBuffer, imgBuffer = \
                 _C.rasterize_gaussians(*args)

        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered

        ctx.save_for_backward(
           colors_precomp,
           means3D,
           scales,
           rotations,
           cov3Ds_precomp,
           radii,
           sh,
           geomBuffer,
           binningBuffer,
           imgBuffer,
           alpha,
           normal
        )

        # Extend the rasterizer outputs with depth, alpha, and normal maps for geometric supervision.
        return color, radii, depth.float(), alpha.float(), normal.float()

    @staticmethod
    def backward(ctx, grad_out_color, grad_out_radii, grad_out_depth, grad_out_alpha, grad_out_normal):

        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings

        (colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh,
         geomBuffer, binningBuffer, imgBuffer, alpha, normal) = ctx.saved_tensors

        radii_arg = radii.to(torch.int32) if radii is not None else None

        sh_degree_int = int(raster_settings.sh_degree)
        num_rendered_int = int(num_rendered)

        debug_bool = bool(raster_settings.debug)

        # Pass depth and normal gradients to the CUDA backend alongside the color gradient.
        args = (
            raster_settings.bg,           # 1
            means3D,                      # 2
            radii_arg,                    # 3
            colors_precomp,               # 4
            scales,                       # 5
            rotations,                    # 6
            raster_settings.scale_modifier,# 7
            cov3Ds_precomp,               # 8
            raster_settings.viewmatrix,   # 9
            raster_settings.projmatrix,   # 10
            raster_settings.tanfovx,      # 11
            raster_settings.tanfovy,      # 12
            raster_settings.kernel_size,  # 13
            raster_settings.subpixel_offset, # 14
            grad_out_color,               # 15
            grad_out_depth,               # 16 <--- 新增
            grad_out_normal,              # 17 <--- 新增
            sh,                           # 18
            sh_degree_int,                # 19
            raster_settings.campos,       # 20
            geomBuffer,                   # 21
            num_rendered_int,             # 22
            binningBuffer,                # 23
            imgBuffer,                    # 24
            debug_bool                    # 25
        )

        if raster_settings.debug:
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, \
                grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = \
                    _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
            grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, \
            grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = \
                _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads


class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int
    tanfovx: float
    tanfovy: float
    kernel_size: float
    subpixel_offset: torch.Tensor
    bg: torch.Tensor
    scale_modifier: float
    viewmatrix: torch.Tensor
    projmatrix: torch.Tensor
    sh_degree: int
    campos: torch.Tensor
    prefiltered: bool
    debug: bool


class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix
            )
        return visible

    def forward(
        self,
        means3D,
        means2D,
        opacities,
        shs=None,
        colors_precomp=None,
        scales=None,
        rotations=None,
        cov3D_precomp=None
    ):

        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')

        if ((scales is None or rotations is None) and cov3D_precomp is None) or \
           ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')

        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales,
            rotations,
            cov3D_precomp,
            raster_settings,
        )

