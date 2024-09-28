#!/usr/bin/env python
# Copyright (c) Xuangeng Chu (xg.chu@outlook.com)

import os
import torch
import numpy as np
import torch.nn as nn
from pytorch3d.io import load_obj
from pytorch3d.structures import (
    Meshes, Pointclouds
)
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform, RasterizationSettings, 
    FoVPerspectiveCameras, PointsRasterizationSettings,
    PointsRenderer, PointsRasterizer, AlphaCompositor,
    PointLights, AmbientLights, TexturesVertex, TexturesUV, 
    SoftPhongShader, MeshRasterizer, MeshRenderer, SoftSilhouetteShader
)

class RenderMesh(nn.Module):
    def __init__(self, image_size, obj_filename=None, faces=None, device='cpu'):
        super(RenderMesh, self).__init__()
        self.device = device
        self.image_size = image_size
        if obj_filename is not None:
            verts, faces, aux = load_obj(obj_filename, load_textures=False)
            self.faces = faces.verts_idx
        elif faces is not None:
            import numpy as np
            self.faces = torch.tensor(faces.astype(np.int32))
        else:
            raise NotImplementedError('Must have faces.')
        self.raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
        self.lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]])

    def _build_cameras(self, transform_matrix, focal_length):
        batch_size = transform_matrix.shape[0]
        screen_size = torch.tensor(
            [self.image_size, self.image_size], device=self.device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': torch.zeros(batch_size, 2, device=self.device).float(), 'focal_length': focal_length, 
            'image_size': screen_size, 'device': self.device,
        }
        cameras = PerspectiveCameras(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])
        return cameras

    def forward(
            self, vertices, cameras=None, transform_matrix=None, focal_length=None
        ):
        if cameras is None:
            cameras = self._build_cameras(transform_matrix, focal_length)
        faces = self.faces[None].repeat(vertices.shape[0], 1, 1)
        # Initialize each vertex to be white in color.
        verts_rgb = torch.ones_like(vertices)  # (1, V, 3)
        textures = TexturesVertex(verts_features=verts_rgb.to(self.device))
        mesh = Meshes(
            verts=vertices.to(self.device),
            faces=faces.to(self.device),
            textures=textures
        )
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=self.raster_settings),
            shader=SoftPhongShader(cameras=cameras, lights=self.lights, device=self.device)
        )
        render_results = renderer(mesh).permute(0, 3, 1, 2)
        images = render_results[:, :3]
        alpha_images = render_results[:, 3:]
        images[alpha_images.expand(-1, 3, -1, -1)<0.5] = 0.0
        return images*255, alpha_images


class PointRenderer(nn.Module):
    def __init__(self, image_size=256, device='cpu'):
        super(PointRenderer, self).__init__()
        self.device = device
        R, T = look_at_view_transform(4, 30, 30) # d, e, a
        self.cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=0.01, zfar=1.0)
        raster_settings = PointsRasterizationSettings(
            image_size=image_size, radius=0.005, points_per_pixel=10
        )
        rasterizer = PointsRasterizer(cameras=self.cameras, raster_settings=raster_settings)
        self.renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
        
    def forward(self, points, D=3, E=15, A=30, coords=True, ex_points=None):
        if D !=8 or E != 30 or A != 30:
            R, T = look_at_view_transform(D, E, A) # d, e, a
            self.cameras = FoVPerspectiveCameras(device=self.device, R=R, T=T, znear=0.01, zfar=1.0)
        verts = torch.Tensor(points).to(self.device)
        verts = verts[:, torch.randperm(verts.shape[1])[:10000]]
        if ex_points is not None:
            verts = torch.cat([verts, ex_points.expand(verts.shape[0], -1, -1)], dim=1)
        if coords:
            coords_size = verts.shape[1]//10
            cod = verts.new_zeros(coords_size*3, 3)
            li = torch.linspace(0, 1.0, steps=coords_size, device=cod.device)
            cod[:coords_size, 0], cod[coords_size:coords_size*2, 1], cod[coords_size*2:, 2] = li, li, li
            verts = torch.cat(
                [verts, cod.unsqueeze(0).expand(verts.shape[0], -1, -1)], dim=1
            )
        rgb = torch.Tensor(torch.rand_like(verts)).to(self.device)
        point_cloud = Pointclouds(points=verts, features=rgb)
        images = self.renderer(point_cloud, cameras=self.cameras,).permute(0, 3, 1, 2)
        return images*255


class TextureRenderer(nn.Module):
    def __init__(self, obj_filename=None, tuv=None, flame_mask=None, device='cpu'):
        super(TextureRenderer, self).__init__()
        self.device = device
        # objects
        if obj_filename is not None:
            _, faces, aux = load_obj(obj_filename, load_textures=False)
            self.uvverts = aux.verts_uvs[None, ...].to(self.device)  # (N, V, 2)
            self.uvfaces = faces.textures_idx[None, ...].to(self.device)  # (N, F, 3)
            self.faces = faces.verts_idx[None, ...].to(self.device) # (N, F, 3)
        elif tuv is not None:
            import numpy as np
            self.uvverts = tuv['verts_uvs'][None, ...].to(self.device)  # (N, V, 2)
            self.uvfaces = tuv['textures_idx'][None, ...].to(self.device) # (N, F, 3)
            self.faces = tuv['verts_idx'][None, ...].to(self.device) # (N, F, 3)
        else:
            raise NotImplementedError('Must have faces and uvs.')
        # setting
        self.lights = AmbientLights(device=self.device)
        # flame mask
        if flame_mask is not None:
            reduced_faces = []
            for f in self.faces[0]:
                valid = 0
                for v in f:
                    if v.item() in flame_mask:
                        valid += 1
                reduced_faces.append(True if valid == 3 else False)
            reduced_faces = torch.tensor(reduced_faces).to(self.device)
            self.flame_mask = reduced_faces
        ## lighting
        pi = np.pi
        sh_const = torch.tensor(
            [
                1 / np.sqrt(4 * pi),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                ((2 * pi) / 3) * (np.sqrt(3 / (4 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (3 / 2) * (np.sqrt(5 / (12 * pi))),
                (pi / 4) * (1 / 2) * (np.sqrt(5 / (4 * pi))),
            ],
            dtype=torch.float32,
        )
        self.constant_factor = sh_const.to(self.device)

    def add_SHlight(self, normal_images, sh_coeff):
        # sh_coeff: [bz, 9, 3]
        N = normal_images
        sh = torch.stack([
            N[:, 0] * 0. + 1., N[:, 0], N[:, 1],
            N[:, 2], N[:, 0] * N[:, 1], N[:, 0] * N[:, 2],
            N[:, 1] * N[:, 2], N[:, 0] ** 2 - N[:, 1] ** 2, 3 * (N[:, 2] ** 2) - 1
        ], 1)  # [bz, 9, h, w]
        sh = sh * self.constant_factor[None, :, None, None]
        shading = torch.sum(sh_coeff[:, :, :, None, None] * sh[:, :, None, :, :], 1)  # [bz, 9, 3, h, w]
        return shading

    def _build_cameras(self, transform_matrix, focal_length, principal_point, image_size):
        batch_size = transform_matrix.shape[0]
        screen_size = torch.tensor(
            [image_size, image_size], device=self.device
        ).float()[None].repeat(batch_size, 1)
        cameras_kwargs = {
            'principal_point': principal_point.repeat(batch_size, 1), 'focal_length': focal_length, 
            'image_size': screen_size, 'device': self.device,
        }
        cameras = PerspectiveCameras(**cameras_kwargs, R=transform_matrix[:, :3, :3], T=transform_matrix[:, :3, 3])
        return cameras

    def forward(
            self, vertices_world, texture_images, lights=None, image_size=512, 
            cameras=None, transform_matrix=None, focal_length=None, principal_point=None
        ):
        if cameras is None:
            cameras = self._build_cameras(transform_matrix, focal_length, principal_point, image_size)
        batch_size = vertices_world.shape[0]
        faces = self.faces.expand(batch_size, -1, -1)
        textures_uv = TexturesUV(
            maps=texture_images.expand(batch_size, -1, -1, -1).permute(0, 2, 3, 1), 
            faces_uvs=self.uvfaces.expand(batch_size, -1, -1), 
            verts_uvs=self.uvverts.expand(batch_size, -1, -1)
        )
        meshes_world = Meshes(verts=vertices_world, faces=faces, textures=textures_uv)
        # phong renderer
        raster_settings = RasterizationSettings(
            image_size=image_size, blur_radius=0.0, faces_per_pixel=1, 
            perspective_correct=True, cull_backfaces=True
        )
        phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=self.device, cameras=cameras, lights=self.lights)
        )
        image_ref = phong_renderer(meshes_world=meshes_world)
        images = image_ref[..., :3].permute(0, 3, 1, 2)
        masks_all = image_ref[..., 3:].permute(0, 3, 1, 2) > 0.0
        if lights is not None:
            images = self.add_SHlight(images, lights)
            images[~masks_all.expand(-1, 3, -1, -1)] = 0.0
        # silhouette renderer
        with torch.no_grad():
            if hasattr(self, 'flame_mask'):
                textures_verts = TexturesVertex(verts_features=vertices_world.new_ones(vertices_world.shape))
                meshes_masked = Meshes(
                    verts=vertices_world, faces=faces[:, self.flame_mask], textures=textures_verts
                )
                silhouette_renderer = MeshRenderer(
                    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                    shader=SoftSilhouetteShader()
                )
                masks_face = silhouette_renderer(meshes_world=meshes_masked)
                masks_face = masks_face[..., 3:].permute(0, 3, 1, 2) > 0.0
            else:
                masks_face = None
        return images, masks_all, masks_face
