/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}


// 1. 确保这里是 9 个元素
std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGaussiansCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
	const float kernel_size,
	const torch::Tensor& subpixel_offset,
    const int image_height, 
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_depth = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_alpha = torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_normal = torch::full({3, H, W}, 0.0, float_opts); // [新增]
  
  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0) M = sh.size(1);

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data_ptr<float>(),
		W, H,
		means3D.contiguous().data_ptr<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data_ptr<float>(), 
		opacity.contiguous().data_ptr<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data_ptr<float>(), 
		viewmatrix.contiguous().data_ptr<float>(), 
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>(),
		tan_fovx,
		tan_fovy,
		kernel_size,
		subpixel_offset.contiguous().data_ptr<float>(),
		prefiltered,
		out_color.data_ptr<float>(),
		out_depth.data_ptr<float>(),
		out_alpha.data_ptr<float>(),
		out_normal.data_ptr<float>(), // [传入指针]
		radii.data_ptr<int>(),        
		debug);
  }

  // 4. 返回 9 个元素的 tuple
  return std::make_tuple(rendered, out_color, out_depth, out_alpha, out_normal, radii, geomBuffer, binningBuffer, imgBuffer);
}



// ... 前面 forward 部分保持你现在的逻辑，已经基本正确 ...

// 修正：Backward 函数签名必须包含 depth 和 normal 的梯度输入
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGaussiansBackwardCUDA(
    const torch::Tensor& background,
    const torch::Tensor& means3D,
    const torch::Tensor& radii,
    const torch::Tensor& colors,
    const torch::Tensor& scales,
    const torch::Tensor& rotations,
    const float scale_modifier,
    const torch::Tensor& cov3D_precomp,
    const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
    const float tan_fovx,
    const float tan_fovy,
    const float kernel_size,
    const torch::Tensor& subpixel_offset,
    const torch::Tensor& dL_dout_color,
    const torch::Tensor& dL_dout_depth,  // <--- 必须加上
    const torch::Tensor& dL_dout_normal, // <--- 必须加上
    const torch::Tensor& sh,
    const int degree,
    const torch::Tensor& campos,
    const torch::Tensor& geomBuffer,
    const int R,
    const torch::Tensor& binningBuffer,
    const torch::Tensor& imageBuffer,
    const bool debug) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0) M = sh.size(1);

  auto float_opts = means3D.options();

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, float_opts);
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, float_opts);
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, float_opts);
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, float_opts);
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, float_opts);
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, float_opts);
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, float_opts);
  torch::Tensor dL_dscales = torch::zeros({P, 3}, float_opts);
  torch::Tensor dL_drotations = torch::zeros({P, 4}, float_opts);
  
  if(P != 0)
  {  
     CudaRasterizer::Rasterizer::backward(P, degree, M, R,
     background.contiguous().data_ptr<float>(),
     W, H, 
     means3D.contiguous().data_ptr<float>(),
     sh.contiguous().data_ptr<float>(),
     colors.contiguous().data_ptr<float>(),
     scales.contiguous().data_ptr<float>(),
     scale_modifier,
     rotations.contiguous().data_ptr<float>(),
     cov3D_precomp.contiguous().data_ptr<float>(),
     viewmatrix.contiguous().data_ptr<float>(),
     projmatrix.contiguous().data_ptr<float>(),
     campos.contiguous().data_ptr<float>(),
     tan_fovx, tan_fovy,
     kernel_size,
     subpixel_offset.contiguous().data_ptr<float>(),
     radii.contiguous().data_ptr<int>(),
     reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
     reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
     reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
     dL_dout_color.contiguous().data_ptr<float>(),
     dL_dmeans2D.contiguous().data_ptr<float>(),
     dL_dconic.contiguous().data_ptr<float>(),  
     dL_dopacity.contiguous().data_ptr<float>(),
     dL_dcolors.contiguous().data_ptr<float>(),
     dL_dmeans3D.contiguous().data_ptr<float>(),
     dL_dcov3D.contiguous().data_ptr<float>(),
     dL_dsh.contiguous().data_ptr<float>(),
     dL_dscales.contiguous().data_ptr<float>(),
     dL_drotations.contiguous().data_ptr<float>(),
     debug);
     // 注意：如果你的底层接口支持法线梯度，也要在这里传入 dL_dout_normal.data_ptr<float>()
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data<float>(),
		viewmatrix.contiguous().data<float>(),
		projmatrix.contiguous().data<float>(),
		present.contiguous().data<bool>());
  }
  
  return present;
}