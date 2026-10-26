import torch
import torch.nn.functional as F


def get_depth_and_normal(render_pkg, viewpoint_cam):
    """Extract the rendered depth map and estimate surface normals."""
    depth = render_pkg["depth"]  # [1, H, W]

    dz_dx = (depth[:, :, 2:] - depth[:, :, :-2]) / 2.0
    dz_dy = (depth[:, 2:, :] - depth[:, :-2, :]) / 2.0

    dz_dx = F.pad(dz_dx, (1, 1, 0, 0), mode='replicate')
    dz_dy = F.pad(dz_dy, (0, 0, 1, 1), mode='replicate')

    normal_cam = torch.cat([
        -dz_dx * viewpoint_cam.focal_x / depth.clamp(min=1e-3),
        -dz_dy * viewpoint_cam.focal_y / depth.clamp(min=1e-3),
        torch.ones_like(depth)
    ], dim=0)

    normal_cam = F.normalize(normal_cam, dim=0)  # [3, H, W]

    # Transform normals from camera space to world space.
    R_w2c = viewpoint_cam.world_view_transform[:3, :3]
    normal_world = R_w2c.transpose(0, 1) @ normal_cam.view(3, -1)

    return depth, normal_world.view(3, viewpoint_cam.image_height, viewpoint_cam.image_width)


