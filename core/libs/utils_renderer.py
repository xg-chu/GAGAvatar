#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import math
import torch
from diff_gaussian_rasterization_32d import GaussianRasterizationSettings, GaussianRasterizer

NUM_CHANNELS = 32

def render_gaussian(gs_params, cam_matrix, cam_params=None, sh_degree=0, bg_color=None):
    # Build params
    batch_size = cam_matrix.shape[0]
    focal_x, focal_y, cam_size = cam_params['focal_x'], cam_params['focal_y'], cam_params['size']
    points, colors, opacities, scales, rotations = \
        gs_params['xyz'], gs_params['colors'], gs_params['opacities'], gs_params['scales'], gs_params['rotations']
    view_mat, proj_mat, cam_pos = build_camera_matrices(cam_matrix, focal_x, focal_y)
    bg_color = cam_matrix.new_zeros(batch_size, NUM_CHANNELS, dtype=torch.float32) if bg_color is None else bg_color
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    means2D = torch.zeros_like(points, dtype=points.dtype, requires_grad=True, device="cuda") + 0
    try:
        means2D.retain_grad()
    except:
        pass
    # Run rendering
    all_rendered, all_radii = [], []
    for bid in range(batch_size):
        raster_settings = GaussianRasterizationSettings(
            sh_degree=sh_degree, bg=bg_color, 
            image_height=cam_size[0], image_width=cam_size[1],
            tanfovx=1.0 / focal_x, tanfovy=1.0 / focal_y,
            viewmatrix=view_mat[bid], projmatrix=proj_mat[bid], campos=cam_pos[bid],
            scale_modifier=1.0, prefiltered=False, debug=False
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rendered, radii = rasterizer(
            means3D=points[bid], means2D=means2D[bid], 
            shs=None, colors_precomp=colors[bid], 
            opacities=opacities[bid], scales=scales[bid], 
            rotations=rotations[bid], cov3D_precomp=None
        )
        all_rendered.append(rendered)
        all_radii.append(radii)
    all_rendered = torch.stack(all_rendered, dim=0)
    all_radii = torch.stack(all_radii, dim=0)
    return {
        "images": all_rendered, "radii": all_radii, "viewspace_points": means2D,
    }


def build_camera_matrices(cam_matrix, focal_x, focal_y):
    def get_projection_matrix(fov_x, fov_y, z_near=0.01, z_far=100, device='cpu'):
        K = torch.zeros(4, 4, device=device)
        z_sign = 1.0
        K[0, 0] = 1.0 / math.tan((fov_x / 2))
        K[1, 1] = 1.0 / math.tan((fov_y / 2))
        K[3, 2] = z_sign
        K[2, 2] = z_sign * z_far / (z_far - z_near)
        K[2, 3] = -(z_far * z_near) / (z_far - z_near)
        return K

    def get_world_to_view_matrix(transforms):
        assert transforms.shape[-2:] == (3, 4)
        viewmatrix = transforms.new_zeros(transforms.shape[0], 4, 4)
        for i in range(4):
            viewmatrix[:, i, i] = 1.0
        viewmatrix[:, :3, :3] = transforms[:, :3, :3]
        viewmatrix[:, 3, :3] = transforms[:, :3, 3]
        viewmatrix[:, :, :2] *= -1.0
        return viewmatrix

    def get_full_projection_matrix(viewmatrix, fov_x, fov_y):
        proj_matrix = get_projection_matrix(fov_x, fov_y, device=viewmatrix.device)
        full_proj_matrix = viewmatrix @ proj_matrix.transpose(0, 1)
        return full_proj_matrix

    fov_x = 2 * math.atan(1.0 / focal_x)
    fov_y = 2 * math.atan(1.0 / focal_y)
    view_matrix = get_world_to_view_matrix(cam_matrix)
    full_proj_matrix = get_full_projection_matrix(view_matrix, fov_x, fov_y)
    cam_pos = cam_matrix[:, :3, 3]
    return view_matrix, full_proj_matrix, cam_pos
