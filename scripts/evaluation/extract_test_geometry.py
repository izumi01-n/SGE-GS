import os
import sys
import numpy as np

# Handle PLY files without the optional filter_3D property.
import plyfile
_orig_getitem = plyfile.PlyElement.__getitem__
def _patched_getitem(self, key):
    if key == "filter_3D":
        try:
            return _orig_getitem(self, key)
        except ValueError:
            # Return zeros when filter_3D is missing.
            return np.zeros(len(self.data), dtype=np.float32)
    return _orig_getitem(self, key)
plyfile.PlyElement.__getitem__ = _patched_getitem
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "../../submodules"))
sys.path.append(os.path.join(current_dir, "../../submodules", "Propagation"))

import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene

def compute_normal_tv(normal_tensor):
    """Compute the total variation of a rendered normal map."""
    normal_tensor = F.normalize(normal_tensor, dim=0)
    img = normal_tensor.unsqueeze(0) # [1, 3, H, W]
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return ((tv_h + tv_w) / img.numel()).item()

def extract_geometry(dataset, pipeline, ply_path):
    gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, gaussians)

    print(f"[*] Loading trained model PLY from: {ply_path}")
    gaussians.load_ply(ply_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    test_cameras = scene.getTestCameras()
    if len(test_cameras) == 0:
        print("[Warning] No test cameras found! Using train cameras for fallback.")
        test_cameras = scene.getTrainCameras()
    
    tv_scores = []
    print(f"[*] Start processing {len(test_cameras)} test views to compute TV metrics (Pure Data Mode)...")
    
    for idx, viewpoint_cam in enumerate(tqdm(test_cameras)):
        with torch.no_grad():
            render_pkg = render(
                viewpoint_cam, 
                gaussians, 
                pipeline, 
                background, 
                kernel_size=dataset.kernel_size, 
                return_normal=True,      
                return_opacity=True,
                return_depth=True,
                subpixel_offset=None     
            )
            
            rendered_normal = torch.nan_to_num(render_pkg['render_normal'], nan=0.0) 
            tv_val = compute_normal_tv(rendered_normal)
            tv_scores.append(tv_val)

    avg_tv = sum(tv_scores) / len(tv_scores) if tv_scores else 0
    print("\n" + "="*50)
    print(f"测试集平均法线总变分 (Average Normal TV Metric): {avg_tv:.7f}")
    print("="*50)

    txt_path = os.path.join(current_dir, "geometry_eval_results.txt")
    
    with open(txt_path, "a", encoding="utf-8") as f:
        f.write(f"场景数据集: {dataset.source_path}\n")
        f.write(f"点云PLY路径: {ply_path}\n")
        f.write(f"Average Normal TV Metric: {avg_tv:.7f}\n")
        f.write("-" * 60 + "\n\n")
    print(f"[*] 结果已成功追加写入统一文本文件: {txt_path}")

if __name__ == "__main__":
    parser = ArgumentParser(description="Pure Geometry Evaluation for Test Set (No Image Output)")
    lp = ModelParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument("--ply_path", type=str, required=True, help="Path to trained point_cloud.ply file")
    parser.add_argument("--output_dir", type=str, default="./none", help="Placeholder for backward compatibility")
    
    def add_if_not_exists(p, *args, **kwargs):
        dest = kwargs.get('dest') or args[0].lstrip('-').replace('-', '_')
        if not any(arg.dest == dest for arg in p._actions):
            p.add_argument(*args, **kwargs)
            
    add_if_not_exists(parser, '--normal_iter_start', type=int, default=3000)
    add_if_not_exists(parser, '--lambda_l1_normal', type=float, default=0.05)
    add_if_not_exists(parser, '--lambda_depth', type=float, default=0.1)

    args = parser.parse_args()
    
    absolute_ply_path = os.path.abspath(args.ply_path) if not args.ply_path.startswith('/') else args.ply_path
    
    extract_geometry(
        lp.extract(args), 
        pp.extract(args), 
        absolute_ply_path
    )