import torch
import os
from tqdm import tqdm
from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips # 确保环境里有 lpips 库
from gaussian_renderer import render
from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
from argparse import ArgumentParser

def evaluate(model_path, iteration):
    # 1. 环境准备
    parser = ArgumentParser(description="Evaluation script")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=iteration, type=int)
    args = get_combined_args(parser)
    
    dataset = model.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    lpips_fn = lpips.LPIPS(net='vgg').cuda() # 使用 VGG 版本的 LPIPS 是行业标准
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 2. 获取测试相机
    test_cams = scene.getTestCameras()
    print(f"Evaluating {len(test_cams)} images from Test Set...")

    psnrs, ssims, lpips_vals = [], [], []

    # 3. 渲染并计算
    with torch.no_grad():
        for cam in tqdm(test_cams, desc="Rendering"):
            # 关键：必须传入训练时的 kernel_size
            render_pkg = render(cam, gaussians, pipeline.extract(args), background, kernel_size=dataset.kernel_size)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            gt_image = torch.clamp(cam.original_image.cuda(), 0.0, 1.0)

            # 计算指标
            psnrs.append(psnr(image, gt_image).item())
            ssims.append(ssim(image, gt_image).item())
            lpips_vals.append(lpips_fn(image.unsqueeze(0), gt_image.unsqueeze(0)).item())

    # 4. 输出结果
    full_psnr = sum(psnrs) / len(psnrs)
    full_ssim = sum(ssims) / len(ssims)
    full_lpips = sum(lpips_vals) / len(lpips_vals)

    print("\n" + "="*30)
    print(f"Results for Iteration {iteration}:")
    print(f"PSNR:  {full_psnr:.4f}")
    print(f"SSIM:  {full_ssim:.4f}")
    print(f"LPIPS: {full_lpips:.4f}")
    print("="*30)

    # 保存到文件
    with open(os.path.join(model_path, f"results_{iteration}.txt"), "w") as f:
        f.write(f"PSNR: {full_psnr}\nSSIM: {full_ssim}\nLPIPS: {full_lpips}")

if __name__ == "__main__":
    # 修改为你实际的模型路径
    evaluate(model_path="/root/autodl-tmp/output/test_hybrid_v1", iteration=30000)