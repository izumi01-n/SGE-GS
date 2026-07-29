import torch
import torch.nn.functional as F


def get_depth_and_normal(render_pkg, viewpoint_cam):
    """从渲染结果中提取深度并计算法线图"""
    depth = render_pkg["depth"]  # [1, H, W]

    # 1. 计算深度图梯度 (简单的中心差分)
    dz_dx = (depth[:, :, 2:] - depth[:, :, :-2]) / 2.0
    dz_dy = (depth[:, 2:, :] - depth[:, :-2, :]) / 2.0

    # 填充回原来尺寸
    dz_dx = F.pad(dz_dx, (1, 1, 0, 0), mode='replicate')
    dz_dy = F.pad(dz_dy, (0, 0, 1, 1), mode='replicate')

    # 2. 构造相机空间法线 (-dz/dx, -dz/dy, 1)
    # 这里的 1/focal 是为了把像素梯度转为物理空间的斜率
    normal_cam = torch.cat([
        -dz_dx * viewpoint_cam.focal_x / depth.clamp(min=1e-3),
        -dz_dy * viewpoint_cam.focal_y / depth.clamp(min=1e-3),
        torch.ones_like(depth)
    ], dim=0)

    normal_cam = F.normalize(normal_cam, dim=0)  # [3, H, W]

    # 3. 转到世界空间
    R_w2c = viewpoint_cam.world_view_transform[:3, :3]
    normal_world = R_w2c.transpose(0, 1) @ normal_cam.view(3, -1)

    return depth, normal_world.view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)


def find_geometry_holes(render_pkg, depth_threshold=0.1):
    """寻找透明度低但有深度信息的区域（即需要生长的空洞）"""
    alpha = render_pkg["alpha"]
    depth = render_pkg["depth"]
    # 透明度低于0.1 且 深度不为0的区域
    mask = (alpha < 0.1) & (depth > 0.01)
    return mask