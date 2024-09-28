"""
Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.
Modified from smplx code for FLAME by Xuangeng Chu (xg.chu@outlook.com)
"""
import os

import torch
import pickle
import numpy as np
import torch.nn as nn

from .lbs import lbs, batch_rodrigues, vertices2landmarks

class FLAMEModel(nn.Module):
    """
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(self, n_shape, n_exp, scale=1.0, no_lmks=False):
        super().__init__()
        self.scale = scale
        self.no_lmks = no_lmks
        # print("creating the FLAME Model")
        _abs_path = os.path.dirname(os.path.abspath(__file__))
        self.flame_path = os.path.join(_abs_path, '../../../assets')
        self.flame_ckpt = torch.load(
            os.path.join(self.flame_path, 'FLAME_with_eye.pt'), map_location='cpu', weights_only=True
        )
        flame_model = self.flame_ckpt['flame_model']
        flame_lmk = self.flame_ckpt['lmk_embeddings']
        
        self.dtype = torch.float32
        self.register_buffer('faces_tensor', flame_model['f'])
        self.register_buffer('v_template', flame_model['v_template'])
        shapedirs = flame_model['shapedirs']
        self.register_buffer('shapedirs', torch.cat([shapedirs[:, :, :n_shape], shapedirs[:, :, 300:300 + n_exp]], 2))
        num_pose_basis = flame_model['posedirs'].shape[-1]
        self.register_buffer('posedirs', flame_model['posedirs'].reshape(-1, num_pose_basis).T)
        self.register_buffer('J_regressor', flame_model['J_regressor'])
        parents = flame_model['kintree_table'][0]
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', flame_model['weights'])
        # Fixing Eyeball and neck rotation
        self.register_buffer('eye_pose', torch.zeros([1, 6], dtype=torch.float32))
        self.register_buffer('neck_pose', torch.zeros([1, 3], dtype=torch.float32))

        # Static and Dynamic Landmark embeddings for FLAME
        self.register_buffer('lmk_faces_idx', flame_lmk['static_lmk_faces_idx'])
        self.register_buffer('lmk_bary_coords', flame_lmk['static_lmk_bary_coords'].to(dtype=self.dtype))
        self.register_buffer('dynamic_lmk_faces_idx', flame_lmk['dynamic_lmk_faces_idx'].to(dtype=torch.long))
        self.register_buffer('dynamic_lmk_bary_coords', flame_lmk['dynamic_lmk_bary_coords'].to(dtype=self.dtype))
        self.register_buffer('full_lmk_faces_idx', flame_lmk['full_lmk_faces_idx_with_eye'].to(dtype=torch.long))
        self.register_buffer('full_lmk_bary_coords', flame_lmk['full_lmk_bary_coords_with_eye'].to(dtype=self.dtype))

        neck_kin_chain = [];
        NECK_IDX = 1
        curr_idx = torch.tensor(NECK_IDX, dtype=torch.long)
        while curr_idx != -1:
            neck_kin_chain.append(curr_idx)
            curr_idx = self.parents[curr_idx]
        self.register_buffer('neck_kin_chain', torch.stack(neck_kin_chain))
        # print("FLAME Model Done.")

    def get_faces(self, ):
        return self.faces_tensor.long()

    def _find_dynamic_lmk_idx_and_bcoords(
            self, pose, dynamic_lmk_faces_idx, dynamic_lmk_b_coords,
            neck_kin_chain, dtype=torch.float32
        ):
        """
            Selects the face contour depending on the reletive position of the head
            Input:
                vertices: N X num_of_vertices X 3
                pose: N X full pose
                dynamic_lmk_faces_idx: The list of contour face indexes
                dynamic_lmk_b_coords: The list of contour barycentric weights
                neck_kin_chain: The tree to consider for the relative rotation
                dtype: Data type
            return:
                The contour face indexes and the corresponding barycentric weights
        """

        batch_size = pose.shape[0]

        aa_pose = torch.index_select(pose.view(batch_size, -1, 3), 1,
                                     neck_kin_chain)
        rot_mats = batch_rodrigues(
            aa_pose.view(-1, 3), dtype=dtype).view(batch_size, -1, 3, 3)

        rel_rot_mat = torch.eye(3, device=pose.device,
                                dtype=dtype).unsqueeze_(dim=0).expand(batch_size, -1, -1)
        for idx in range(len(neck_kin_chain)):
            rel_rot_mat = torch.bmm(rot_mats[:, idx], rel_rot_mat)

        y_rot_angle = torch.round(
            torch.clamp(rot_mat_to_euler(rel_rot_mat) * 180.0 / np.pi, max=39)
        ).to(dtype=torch.long)

        neg_mask = y_rot_angle.lt(0).to(dtype=torch.long)
        mask = y_rot_angle.lt(-39).to(dtype=torch.long)
        neg_vals = mask * 78 + (1 - mask) * (39 - y_rot_angle)
        y_rot_angle = (neg_mask * neg_vals +
                       (1 - neg_mask) * y_rot_angle)

        dyn_lmk_faces_idx = torch.index_select(dynamic_lmk_faces_idx,
                                               0, y_rot_angle)
        dyn_lmk_b_coords = torch.index_select(dynamic_lmk_b_coords,
                                              0, y_rot_angle)
        return dyn_lmk_faces_idx, dyn_lmk_b_coords

    # @torch.no_grad()
    def forward(self, shape_params=None, expression_params=None, pose_params=None, eye_pose_params=None, verts_sclae=None):
        """
            Input:
                shape_params: N X number of shape parameters
                expression_params: N X number of expression parameters
                pose_params: N X number of pose parameters (6)
            return:d
                vertices: N X V X 3
                landmarks: N X number of landmarks X 3
        """
        batch_size = shape_params.shape[0]
        if pose_params is None:
            pose_params = self.eye_pose.expand(batch_size, -1)
        if eye_pose_params is None:
            eye_pose_params = self.eye_pose.expand(batch_size, -1)
        if expression_params is None:
            expression_params = torch.zeros(batch_size, self.cfg.n_exp).to(shape_params.device)

        betas = torch.cat([shape_params, expression_params], dim=1)
        full_pose = torch.cat([
                pose_params[:, :3], self.neck_pose.expand(batch_size, -1), 
                pose_params[:, 3:], eye_pose_params
            ], dim=1
        )
        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, _ = lbs(
            betas, full_pose, template_vertices,
            self.shapedirs, self.posedirs, self.J_regressor, self.parents,
            self.lbs_weights, dtype=self.dtype, detach_pose_correctives=False
        )
        if self.no_lmks:
            return vertices * self.scale
        landmarks3d = vertices2landmarks(
            vertices, self.faces_tensor, 
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1)
        )
        landmark_3d = reselect_eyes(vertices, landmarks3d)
        if verts_sclae is not None:
            return vertices * verts_sclae, landmark_3d * verts_sclae
        return vertices * self.scale, landmarks3d * self.scale

    def _vertices2landmarks(self, vertices):
        landmarks3d = vertices2landmarks(
            vertices, self.faces_tensor, 
            self.full_lmk_faces_idx.repeat(vertices.shape[0], 1),
            self.full_lmk_bary_coords.repeat(vertices.shape[0], 1, 1)
        )
        landmark_3d = reselect_eyes(vertices, landmarks3d)
        return landmark_3d


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


def rot_mat_to_euler(rot_mats):
    # Calculates rotation matrix to euler angles
    # Careful for extreme cases of eular angles like [0.0, pi, 0.0]

    sy = torch.sqrt(rot_mats[:, 0, 0] * rot_mats[:, 0, 0] +
                    rot_mats[:, 1, 0] * rot_mats[:, 1, 0])
    return torch.atan2(-rot_mats[:, 2, 0], sy)


def reselect_eyes(vertices, lmks70):
    lmks70 = lmks70.clone()
    eye_in_shape = [2422,2422, 2452, 2454, 2471, 3638, 2276, 2360, 3835, 1292, 1217, 1146, 1146, 999, 827, ]
    eye_in_shape_reduce = [0,2,4,5,6,7,8,9,10,11,13,14]
    cur_eye = vertices[:, eye_in_shape]
    cur_eye[:, 0] = (cur_eye[:, 0] + cur_eye[:, 1]) * 0.5
    cur_eye[:, 2] = (cur_eye[:, 2] + cur_eye[:, 3]) * 0.5
    cur_eye[:, 11] = (cur_eye[:, 11] + cur_eye[:, 12]) * 0.5
    cur_eye = cur_eye[:, eye_in_shape_reduce]
    lmks70[:, [37,38,40,41,43,44,46,47]] = cur_eye[:, [1,2,4,5,7,8,10,11]]
    return lmks70
