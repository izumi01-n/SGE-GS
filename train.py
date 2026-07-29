import os
import torch
from random import randint
import sys
import uuid
import random
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
import torch.nn.functional as F
import torchvision

# 核心组件导入
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import check_geometric_consistency, surface_normal_from_depth, get_intrinsic_matrix, depth_propagation
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel
from arguments import ModelParams, OptimizationParams, PipelineParams 
from utils.general_utils import safe_state
from utils.image_utils import psnr, visualize_depth, visualize_normal, save_tensor_image

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    
    # 1. 初始化模型与场景
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    
    all_attrs = dir(gaussians)
    has_filter = "filter_3D" in all_attrs
    has_filter_under = "_filter_3D" in all_attrs
    print(f"--- [Debug] filter_3D: {has_filter}, _filter_3D: {has_filter_under} ---")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # 获取相机列表
    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()

    # 高分辨率偏置采样索引
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    # --- Mip-Splatting 初始化 ---
    if gaussians.get_xyz.shape[0] > 0:
        print(f"[Init] Computing initial 3D filter for {gaussians.get_xyz.shape[0]} points...")
        gaussians.compute_3D_filter(cameras=trainCameras)
        #消融1gaussians.filter_3D.zero_()
    else:
        gaussians.filter_3D = torch.ones((1, 1), device="cuda") * dataset.kernel_size

    # --- GaussianPro 初始化 ---
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    last_densify_iter = 0

    for iteration in range(first_iter, opt.iterations + 1):
        densify_happened = False
        if iteration == first_iter:
            prev_loss = 1e10
        progress = iteration / opt.iterations
        iter_start = torch.cuda.Event(enable_timing=True)
        iter_end = torch.cuda.Event(enable_timing=True)
        iter_start.record()

        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # 2. 采样相机
        if not viewpoint_stack:
            viewpoint_stack = trainCameras.copy()
            
        randidx = randint(0, len(trainCameras) - 1)
        # 高分辨率偏置采样逻辑实现
        if random.random() < 0.3 and len(highresolution_index) > 0:
            randidx = highresolution_index[randint(0, len(highresolution_index)-1)]
            
        viewpoint_cam = trainCameras[randidx]
        gt_image = viewpoint_cam.original_image.cuda()

        # ==============================================================================
        # GaussianPro MVS Propagation Module（几何校验模块）
        # ==============================================================================
        with torch.no_grad():
            # 安全读取命令行传入的生命周期参数
            mvs_begin = getattr(opt, "propagated_iteration_begin", 0)
            mvs_end = getattr(opt, "propagated_iteration_after", 15000)
            mvs_interval = getattr(opt, "propagation_interval", 100) 
            
            is_mvs_iter = (iteration >= mvs_begin) and (iteration <= mvs_end) and (iteration % mvs_interval == 0)
            
            if is_mvs_iter:
                dataset_type = getattr(opt, "dataset", "free")
                src_idxs = []
                # --- 1. 邻居视图选择 ---
                try:
                    cam_centers = torch.stack([cam.camera_center.detach() for cam in viewpoint_stack])
                    current_center = viewpoint_cam.camera_center.detach()
                    dists = torch.norm(cam_centers - current_center.unsqueeze(0), dim=1)
                    closest_idxs = torch.argsort(dists)
                    
                    for idx in closest_idxs:
                        idx_item = idx.item()
                        if idx_item != randidx:  # 排除当前相机
                            src_idxs.append(idx_item)
                        if len(src_idxs) >= 4:   # 4个最近邻视角
                            break
                except Exception as e:
                    print(f"[Warning] 空间邻居搜索失败: {e}。降级使用默认索引间隔。")
                    src_idxs = []
                    intervals = [-3, -2, 2, 3]
                    for itv in intervals:
                        target = randidx + itv
                        if 0 <= target < len(viewpoint_stack) and target != randidx:
                            src_idxs.append(target)
              
                # --- 2. 深度/法线传播 ---
                if len(src_idxs) > 0:
                    if not hasattr(viewpoint_cam, "K"):
                        viewpoint_cam.K = get_intrinsic_matrix(viewpoint_cam).cuda()
                    for src_idx in src_idxs:
                        src_view = viewpoint_stack[src_idx]
                        if not hasattr(src_view, "K"):
                            src_view.K = get_intrinsic_matrix(src_view).cuda()

                    render_pkg_pre = render(
                        viewpoint_cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, 
                        return_normal=False, return_opacity=True, return_depth=True
                    )
                    projected_depth = render_pkg_pre["render_depth"].squeeze(0)

                    try:
                        prop_depth, prop_normal = depth_propagation(
                            viewpoint_cam, projected_depth, viewpoint_stack, 
                            src_idxs, dataset_type, getattr(opt, "patch_size", 40)
                        )
                    except Exception as e:
                        raise e

                    # --- 3. 法线坐标系处理与挂载 ---
                    H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
                    if prop_normal.dim() == 3:
                        if prop_normal.shape[-1] == 3:
                            prop_normal_cam = prop_normal.permute(2, 0, 1)
                        elif prop_normal.shape[0] == 3:
                            prop_normal_cam = prop_normal
                        else:
                            raise ValueError(f"Unknown normal shape: {prop_normal.shape}")
                    elif prop_normal.dim() == 2:
                        if prop_normal.shape[-1] == 3:
                            prop_normal_cam = prop_normal.reshape(H, W, 3).permute(2, 0, 1)
                        else:
                            raise ValueError(f"Unknown normal shape: {prop_normal.shape}")
                    else:
                        raise ValueError(f"Unknown normal dim: {prop_normal.shape}")
                    
                    R_w2c = viewpoint_cam.world_view_transform[:3, :3].T
                    prop_normal_cam = torch.einsum('ij,jhw->ihw', R_w2c, prop_normal_cam)
                    prop_normal_cam = F.normalize(prop_normal_cam, dim=0)

                    viewpoint_cam.normal = prop_normal_cam.detach()
                    viewpoint_cam.depth = prop_depth.detach()

                    # --- 4. 几何一致性校验（重投影验证） ---
                    geometric_counts = None
                    for src_idx in src_idxs:
                        src_view = viewpoint_stack[src_idx]

                        if not hasattr(src_view, "depth") or src_view.depth is None:
                            src_render = render(
                                src_view, gaussians, pipe, background, kernel_size=dataset.kernel_size, 
                                return_normal=False, return_opacity=True, return_depth=True
                            )
                            src_projected_depth = src_render["render_depth"].squeeze(0)

                            src_depth, _ = depth_propagation(
                                src_view, src_projected_depth, viewpoint_stack, 
                                src_idxs, dataset_type, getattr(opt, "patch_size", 40)
                            )
                            src_view.depth = src_depth.detach()

                        mask, _, _, _, _ = check_geometric_consistency(
                            prop_depth.unsqueeze(0), viewpoint_cam.K.unsqueeze(0), 
                            viewpoint_cam.world_view_transform.T.inverse().unsqueeze(0),
                            src_view.depth.unsqueeze(0), src_view.K.unsqueeze(0), 
                            src_view.world_view_transform.T.inverse().unsqueeze(0)
                        )
                        geometric_counts = mask.to(torch.uint8) if geometric_counts is None else geometric_counts + mask.to(torch.uint8)

                    # --- 5. 显式加密渲染 ---
                    if geometric_counts is not None:
                        cost_mask = (geometric_counts.squeeze() >= 2) 
                        render_depth = render_pkg_pre["render_depth"].squeeze()

                        abs_rel_error = torch.abs(prop_depth - render_depth) / (prop_depth + 1e-7)
                        error_mask = abs_rel_error > getattr(opt, "depth_error_threshold", 0.2)

                        final_prop_mask = cost_mask & error_mask
                        
                        sky_mask = getattr(viewpoint_cam, 'sky_mask', None)
                        if sky_mask is not None:
                            final_prop_mask &= sky_mask.to(torch.bool)

                        valid_count = final_prop_mask.sum().item()
                        max_new_pts = 5000

                        if valid_count > max_new_pts:
                            ys, xs = torch.where(final_prop_mask)
                            perm = torch.randperm(valid_count, device=final_prop_mask.device)[:max_new_pts]
                            new_mask = torch.zeros_like(final_prop_mask)
                            new_mask[ys[perm], xs[perm]] = True
                            final_prop_mask = new_mask
                            valid_count = max_new_pts

                        if valid_count > 100:
                            input_depth = prop_depth.clone()
                            input_depth[~final_prop_mask] = prop_depth.max() * 2.0 
                            gaussians.densify_from_depth_propagation(viewpoint_cam, input_depth, final_prop_mask, gt_image)

                            if iteration % 100 == 0:
                                print(f"[MVS] Iter: {iteration} | View: {viewpoint_cam.image_name} | New Pts: {final_prop_mask.sum().item()}")

#                            if iteration % 1000 == 0:
#                                os.makedirs(f"{scene.model_path}/debug_normals", exist_ok=True)
#                                vis_normal = (prop_normal_cam * 0.5 + 0.5).clamp(0, 1)
#                                torchvision.utils.save_image(vis_normal, f"{scene.model_path}/debug_normals/{iteration}_mvs.png")
                              

        # subpixel jitter
        if dataset.ray_jitter:
            subpixel_offset = torch.rand(
                (int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2),
                dtype=torch.float32,
                device="cuda"
            ) - 0.5
        else:
            subpixel_offset = None

        if dataset.resample_gt_image and subpixel_offset is not None:
            gt_image = create_offset_gt(gt_image, subpixel_offset)

        # --- 【核心渲染：Mip-Splatting 模式】 ---
        render_pkg = render(
            viewpoint_cam, 
            gaussians, 
            pipe, 
            background, 
            kernel_size=dataset.kernel_size, 
            return_normal=getattr(opt, 'normal_loss', False),
            return_opacity=True,
            return_depth=True,
            subpixel_offset=subpixel_offset
        )

#        if iteration >= 7000 and iteration % 1000 == 0:
#            if hasattr(viewpoint_cam, 'normal') and viewpoint_cam.normal is not None:
#                os.makedirs(os.path.join(args.model_path, "normal_vis"), exist_ok=True)
#                save_normal_comparison(
#                    iteration,
#                    render_pkg["render_normal"], 
#                    viewpoint_cam.normal,        
#                    torch.ones_like(render_pkg["render_depth"]), # 意见2：修改键名
#                    os.path.join(args.model_path, "normal_vis")
#                )
#            else:
#                print(f"--- [Warning] Iter {iteration}: viewpoint_cam has NO normal! ---")

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        depth, alpha = render_pkg["render_depth"], render_pkg["render_opacity"]

        # ===============================
        # 全局动态权重与空间纹理感知 (Spatial Texture Decoupling)
        # ===============================
        total_iters = getattr(opt, 'iterations', 30000)
        # 时间退火：4000步后开始缓慢降低几何约束，最低降至 20%
        time_progress = max(0.0, min(1.0, (iteration - 4000) / (total_iters - 4000)))
        time_decay_factor = 1.0 - 0.8 * time_progress 
        #time_decay_factor = 1.0 #消融5关闭时间退火

        # 空间掩码：利用真实图像计算纹理复杂度，平整区域权重趋近1，纹理区域趋近0
        with torch.no_grad():
            dx = torch.abs(gt_image[:, :, 1:] - gt_image[:, :, :-1]).mean(dim=0)
            dy = torch.abs(gt_image[:, 1:, :] - gt_image[:, :-1, :]).mean(dim=0)
            dx = torch.nn.functional.pad(dx, (0, 1, 0, 0))
            dy = torch.nn.functional.pad(dy, (0, 0, 0, 1))
            texture_intensity = dx + dy
            spatial_weight = torch.exp(-10.0 * texture_intensity) # 形状 [H, W]
            #spatial_weight = torch.ones_like(texture_intensity) #消融4：w/o Spatial Texture Mask

        # ===============================
        # 1. Opacity Mask（动态调度）
        # ===============================
        threshold = 0.02 + 0.18 * (time_progress ** 2) # 复用 time_progress
        opacity_mask = alpha > threshold
        opacity_mask = opacity_mask.repeat(3, 1, 1)

        finite_mask = torch.isfinite(image) & torch.isfinite(gt_image)
        valid_mask = opacity_mask & finite_mask

        min_valid_ratio = 0.01 
        if valid_mask.float().mean() < min_valid_ratio:
            valid_mask = finite_mask 

        # ===============================
        # 2. RGB Loss
        # ===============================
        Ll1 = l1_loss(image[valid_mask], gt_image[valid_mask])
        try:
            ssim_val = ssim(image, gt_image, mask=valid_mask)
            if not torch.isfinite(ssim_val):
                raise ValueError("SSIM NaN")
        except:
            ssim_val = torch.tensor(0.0, device=image.device)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_val)

        # ===============================
        # 3. Flatten Loss（叠加时间退火）
        # ===============================
        if getattr(opt, 'flatten_loss', False):
            scales = gaussians.get_scaling
            min_scale, _ = torch.min(scales, dim=1)
            min_scale = torch.clamp(min_scale, 1e-6, 30)
            flatten_loss = torch.abs(min_scale).mean()
            if torch.isfinite(flatten_loss):
                current_lambda_flatten = opt.lambda_flatten * time_decay_factor
                loss += current_lambda_flatten * flatten_loss

        # ===============================
        # 4. Sparse Loss
        # ===============================
        if getattr(opt, 'sparse_loss', False) and iteration > 2000:
            sparse_progress = min(1.0, (iteration - 2000) / 5000)
            opacity = gaussians.get_opacity.clamp(1e-6, 1 - 1e-6)
            log_opacity = opacity * torch.log(opacity)
            log_one_minus_opacity = (1 - opacity) * torch.log(1 - opacity)
            entropy = log_opacity + log_one_minus_opacity
            sparse_mask = visibility_filter & torch.isfinite(entropy)
            if sparse_mask.any():
                sparse_loss = -entropy[sparse_mask].mean()
                if torch.isfinite(sparse_loss):
                    loss += sparse_progress * opt.lambda_sparse * sparse_loss

        # ===============================
        # 5. Normal + Depth Loss（叠加时间退火与空间纹理掩码）
        # ===============================
        if getattr(opt, 'normal_loss', False) and iteration > opt.normal_iter_start:
            if hasattr(viewpoint_cam, 'normal') and viewpoint_cam.normal is not None:
                rendered_normal = torch.nan_to_num(render_pkg['render_normal'], nan=0.0)
                normal_gt = torch.nan_to_num(viewpoint_cam.normal.cuda(), nan=0.0)
                rendered_depth = torch.nan_to_num(render_pkg['render_depth'], nan=0.0) 
                depth_gt = torch.nan_to_num(viewpoint_cam.depth.cuda(), nan=0.0)

                mask = (depth_gt > 0) & torch.isfinite(rendered_depth)
                mask_val = mask.float().mean()

                if mask_val > 0.005: 
                    mask_flat = mask.reshape(-1).float()
                    spatial_weight_flat = spatial_weight.reshape(-1)

                    n = F.normalize(rendered_normal.permute(1, 2, 0).reshape(-1, 3), dim=1)
                    n_gt = F.normalize(normal_gt.permute(1, 2, 0).reshape(-1, 3), dim=1)

                    # 基础相似度损失 + 空间掩码降权（纹理区免疫 MVS 压迫）
                    loss_normal_raw = 1.0 - (n * n_gt).sum(dim=1)
                    loss_normal = (loss_normal_raw * mask_flat * spatial_weight_flat).mean()

                    # 基础深度损失
                    if mask.dim() == 3:
                        _mask = mask[0, :, :].to(torch.bool)  
                    elif mask.dim() == 4:
                        _mask = mask[0, 0, :, :].to(torch.bool)
                    else:
                        _mask = mask.to(torch.bool)

                    _rendered_depth = rendered_depth.squeeze()
                    _depth_gt = depth_gt.squeeze()

                    # 对深度损失施加空间掩码
                    loss_depth_raw = torch.abs(_rendered_depth[_mask] - _depth_gt[_mask])
                    loss_depth = (loss_depth_raw * spatial_weight[_mask]).mean()

                    # 最终权重 = 预热系数(w) * 时间退火衰减(time_decay_factor) * 基础权重
                    w = min(1.0, (iteration - opt.normal_iter_start) / 5000.0) 
                    current_lambda_normal = getattr(opt, 'lambda_l1_normal', 0.05) * time_decay_factor
                    current_lambda_depth = getattr(opt, 'lambda_depth', 0.1) * time_decay_factor

                    loss += w * current_lambda_normal * loss_normal
                    loss += w * current_lambda_depth * loss_depth

        # ===============================
        # 6. NaN 防御退出机制
        # ===============================
        if not torch.isfinite(loss):
            print(f"⚠️ Skip iteration {iteration} due to NaN loss")
            gaussians.optimizer.zero_grad(set_to_none=True)
            continue

        # 4. 反向传播
        loss.backward()
        iter_end.record()
        
        num_points = gaussians.get_xyz.shape[0]
        if not hasattr(opt, "initial_point_count"):
            opt.initial_point_count = num_points

        # 5. 后处理与 Densification
        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            prev_loss = ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.5f}", "Pts": f"{gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)

#            if iteration % 500 == 0:
#                visualize_training_step(iteration, viewpoint_cam, image, depth, gaussians, pipe, background, dataset, dataset.model_path)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if (iteration > opt.densify_from_iter and (iteration - last_densify_iter) >= getattr(opt, "densification_interval", 100)):              
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    densify_happened = True
                    last_densify_iter = iteration 
                    
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    densify_happened = True
                    
            # 【Mip-Splatting 同步核心】
            if densify_happened or iteration % getattr(opt, "filter_update_interval", 500) == 0:
                gaussians.compute_3D_filter(cameras=trainCameras)
                #消融1gaussians.filter_3D.zero_()
                
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

                training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))
            if iteration in saving_iterations:
                scene.save(iteration)
            if iteration in checkpoint_iterations:
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def validate_geometry(ref_cam, prop_depth, all_cams):
    ref_K = ref_cam.K
    ref_pose = ref_cam.world_view_transform.transpose(0, 1).inverse()
    counts = None
    
    for s_idx in ref_cam.source_indices[:3]: 
        src_cam = all_cams[s_idx]
        src_depth = src_cam.depth if hasattr(src_cam, 'depth') else prop_depth
        mask, _, _, _, _ = check_geometric_consistency(
            prop_depth.unsqueeze(0), ref_K.unsqueeze(0), ref_pose.unsqueeze(0),
            src_depth.unsqueeze(0), src_cam.K.unsqueeze(0), src_cam.world_view_transform.transpose(0, 1).inverse().unsqueeze(0)
        )
        counts = mask.to(torch.uint8) if counts is None else counts + mask.to(torch.uint8)
    
    return (prop_depth < 100) & (counts.squeeze() >= 2) if counts is not None else (prop_depth < 100)


def visualize_training_step(iteration, cam, image, depth, gaussians, pipe, background, dataset, model_path):
    vis_dir = os.path.join(model_path, "training_vis")
    os.makedirs(vis_dir, exist_ok=True)
    
    rgb = torch.clamp(image, 0.0, 1.0).cpu()
    depth_vis = visualize_depth(depth).cpu()
    
    # 意见1：修改参数名
    norm_pkg = render(
        cam, gaussians, pipe, background, kernel_size=dataset.kernel_size, 
        return_normal=True, return_opacity=True, return_depth=True
    )
    normal_vis = visualize_normal(norm_pkg["render_normal"]).cpu()
    
    stack = torch.cat([rgb, depth_vis, normal_vis], dim=2)
    save_tensor_image(stack, os.path.join(vis_dir, f"iter_{iteration:05d}.png"))


def save_normal_comparison(iteration, render_normal, prop_normal, mask, save_path):
    with torch.no_grad():
        if prop_normal.dim() == 3 and prop_normal.shape[-1] == 3:
            prop_normal = prop_normal.permute(2, 0, 1)
        
        vis_render = F.normalize(render_normal, dim=0) * 0.5 + 0.5
        vis_prop = F.normalize(prop_normal, dim=0) * 0.5 + 0.5
        
        dot = (F.normalize(render_normal, dim=0) * F.normalize(prop_normal, dim=0)).sum(dim=0).clamp(-1.0, 1.0)
        angle_err = torch.acos(dot) * (180.0 / 3.1415926) 
        vis_err = torch.stack([angle_err / 90.0, 1.0 - (angle_err / 90.0), torch.zeros_like(angle_err)], dim=0).clamp(0, 1)
        
        if mask.dim() == 2:
            vis_mask = mask.float().unsqueeze(0).repeat(3, 1, 1)
        else:
            vis_mask = mask.float().view(1, mask.shape[-2], mask.shape[-1]).repeat(3, 1, 1)
        
        grid = torchvision.utils.make_grid([vis_render, vis_prop, vis_err, vis_mask], nrow=2)
        torchvision.utils.save_image(grid, os.path.join(save_path, f"comp_{iteration:05d}.png"))

    
def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    pkg = renderFunc(
                        viewpoint, scene.gaussians, *renderArgs, 
                        return_normal=False, return_opacity=True, return_depth=True
                    )
                    image = torch.clamp(pkg["render"], 0.0, 1.0)
                    depth = pkg["render_depth"]
                    alpha = pkg["render_opacity"]
                    depth_vis = depth / (depth.max() + 1e-5)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

#                    if tb_writer and (idx < 5):
#                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
#                        tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), depth_vis[None], global_step=iteration)
#                        tb_writer.add_images(config['name'] + "_view_{}/alpha".format(viewpoint.image_name), alpha[None], global_step=iteration)
#
#                        if iteration == testing_iterations[0]:
#                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Hybrid Mip-GaussianPro Training script")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default=None)
    parser.add_argument('--max_density_ratio', type=float, default=5.0)

    def add_if_not_exists(parser, *args, **kwargs):
        dest = kwargs.get('dest') or args[0].lstrip('-').replace('-', '_')
        if not any(arg.dest == dest for arg in parser._actions):
            parser.add_argument(*args, **kwargs)

    add_if_not_exists(parser, '--normal_iter_start', type=int, default=3000)
    add_if_not_exists(parser, '--propagation_interval', type=int, default=100)
    add_if_not_exists(parser, '--patch_size', type=int, default=40)
    add_if_not_exists(parser, '--patch_size_decay_steps', type=int, default=10000)
    add_if_not_exists(parser, '--lambda_flatten', type=float, default=1.0)
    add_if_not_exists(parser, '--lambda_l1_normal', type=float, default=0.05)
    add_if_not_exists(parser, '--lambda_depth', type=float, default=0.1)
    add_if_not_exists(parser, '--depth_error_threshold', type=float, default=0.15)

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    training(lp.extract(args), op.extract(args), pp.extract(args), 
             args.test_iterations, args.save_iterations, 
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    print("\nTraining complete.")