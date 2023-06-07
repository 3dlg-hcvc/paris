import torch
import torch.nn as nn
from math import sin, cos
import models
from models.base import BaseModel
from models.utils import chunk_batch
from systems.utils import update_module_step
from nerfacc import ContractionType, OccupancyGrid, ray_marching
from nerfacc.vol_rendering import render_transmittance_from_alpha, rendering
from torch_efficient_distloss import flatten_eff_distloss
from models.utils import scale_anything
from utils.rotation import quaternion_to_axis_angle

@models.register('d2nerf_se3')
class D2NeRFQuatModel(BaseModel):
    def setup(self):
        self.static_geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.static_texture = models.make(self.config.texture.name, self.config.texture)
        self.dynamic_geometry = models.make(self.config.geometry.name, self.config.geometry)
        self.dynamic_texture = models.make(self.config.texture.name, self.config.texture)

        # if self.config.get('motion_gt', False):
        #     self.load_gt_motion(self.config['motion_gt_path'])
        #     # self.load_gt_motion_real(self.config['motion_gt_path'])

        # elif self.config.get('motion_init_gt', False):
        #     self.init_gt_motion(self.config['motion_gt_path'])
        # elif self.config.get('init_guaranteed', False):
        #     self.init_acute_diff_w_GT(self.config['motion_gt_path'])
        # else:
        #     init_angle = self.config.get('init_angle', 0.1)
        #     init_dir = self.config.get('init_dir', [1., 1., 1.])
        #     # self.axis_o = nn.Parameter(torch.tensor([0., 0., 0.], dtype=torch.float32), requires_grad=True)
        #     self.quaternions = nn.Parameter(self.init_quaternions(half_angle=init_angle, init_dir=init_dir), requires_grad=True) # real part first
            # self.translation = nn.Parameter(torch.tensor([0.001, 0.001, 0.001], dtype=torch.float32), requires_grad=True) # edit: temp ignore
        init_angle = self.config.get('init_angle', 0.1)
        init_dir = self.config.get('init_dir', [1., 1., 1.])
        self.quaternions = nn.Parameter(self.init_quaternions(half_angle=init_angle, init_dir=init_dir), requires_grad=True) # real part first
        self.translation = nn.Parameter(torch.tensor([0.001, 0.001, 0.001], dtype=torch.float32), requires_grad=True) # edit: temp ignore
        self.canonical = 0.5 # the canonical state ranging in [0, 1]

        self.register_buffer('scene_aabb', torch.as_tensor([-self.config.radius, -self.config.radius, -self.config.radius, self.config.radius, self.config.radius, self.config.radius], dtype=torch.float32))
        if self.config.grid_prune:
            self.grid_warmup = self.config['grid_warmup']
            self.occupancy_grid = OccupancyGrid(
                roi_aabb=self.scene_aabb,
                resolution=128, # the resolution is open to discuss
                contraction_type=ContractionType.AABB
            )

        self.randomized = self.config.randomized
        if self.config.white_bkgd:
            self.register_buffer('background_color', torch.as_tensor([1.0, 1.0, 1.0], dtype=torch.float32), persistent=False)
            self.background_color.to(self.rank)

        self.render_step_size = 1.732 * 2 * self.config.radius / self.config.num_samples_per_ray

        self.occ_alpha_thre = 1e-4

        self.use_maximum = self.config.get('use_maximum', False)
        if not self.use_maximum:
            self.occ_alpha_thre = 2. * self.occ_alpha_thre
        self.use_distortion = self.config.get('use_distortion', False)
        self.use_swi = self.config.get('use_swi', False)
        self.use_intersect = self.config.get('use_intersect', False)
        self.use_part_mask = self.config.get('use_part_mask', False)
        self.use_align_noi = self.config.get('use_align_noi', False)
        self.use_deg_reg = self.config.get('use_deg_reg', False)

        
        # alpha filter and early stop
        self.rm_alpha_thre = self.config.ray_marching.alpha_thre
        self.rm_early_stop_eps = self.config.ray_marching.early_stop_eps
        self.cr_alpha_thre = self.config.comp_rendering.alpha_thre
        self.cr_early_stop_eps = self.config.comp_rendering.early_stop_eps
    
    def reset_layer_weights(self, layer):
        if hasattr(layer, 'native_tcnn_module'):
            init = layer.native_tcnn_module.initial_params(layer.seed)
            layer.params.data = init
        else:
            if hasattr(layer, 'children'):
                for child in layer.children():
                    self.reset_layer_weights(child)
    
    def reset_model_weights(self, model):
        for layer in model.children():
            self.reset_layer_weights(layer)
    
    def reset_dynamic(self):
        '''
        This function is to reset the dynamic field and motion params for the second stage of the training
        '''
        # self.dynamic_geometry = models.make(self.config.geometry.name, self.config.geometry)
        # self.dynamic_texture = models.make(self.config.texture.name, self.config.texture)

        # for layer in self.dynamic_geometry.children():
        #     self.reset_model_weights(layer)
        
        # for layer in self.dynamic_texture.children():
        #     self.reset_model_weights(layer)
        
        self.reset_model_weights(self.dynamic_geometry)
        self.reset_model_weights(self.dynamic_texture)

        self.axis_o.data = torch.tensor([0., 0., 0.]).to(self.rank)
        self.quaternions.data = self.init_quaternions(half_angle=0.1).to(self.rank) # real part first
        
        # self.axis_o = nn.Parameter(torch.tensor([0., 0., 0.], dtype=torch.float32).to(self.rank), requires_grad=True)
        # self.quaternions = nn.Parameter(self.init_quaternions(half_angle=0.1).to(self.rank), requires_grad=True)
        print('Reset dynamics...start the second stage')
        

    def load_gt_motion(self, path):
        import json
        import numpy as np
        with open(path, mode='r') as f:
            meta = json.load(f)
            trans_info = meta['trans_info']
            f.close()
        axis_o = np.array(trans_info['axis']['o']).astype(np.float32)
        axis_d = np.array(trans_info['axis']['d']).astype(np.float32)
        frame_R_sapien = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        half_angle = (trans_info['rotate']['r'] - trans_info['rotate']['l']) * 0.5
        # half_translate = (trans_info['translate']['l'] + trans_info['translate']['r']) * 0.5
        
        axis_d = torch.matmul(frame_R_sapien, torch.tensor(axis_d))
        axis_o = torch.matmul(frame_R_sapien, torch.tensor(axis_o))
        gt_angle = torch.deg2rad(torch.tensor([half_angle]))
        q = self.axis_angle_to_quaternions(axis_d, gt_angle)
        self.register_buffer('quaternions', q, persistent=False)
        self.register_buffer('axis_o', axis_o, persistent=False)
    
    def load_gt_motion_real(self, path):
        import json
        import numpy as np
        with open(path, mode='r') as f:
            meta = json.load(f)
            trans_info = meta['trans_info']
            f.close()
        axis_o = np.array(trans_info['axis']['o']).astype(np.float32)
        axis_d = np.array(trans_info['axis']['d']).astype(np.float32)
        frame_R_sapien = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        half_angle = (trans_info['rotate']['r'] - trans_info['rotate']['l']) * 0.5
        # half_translate = (trans_info['translate']['l'] + trans_info['translate']['r']) * 0.5
        
        axis_d = torch.tensor(axis_d)
        axis_o = torch.tensor(axis_o)
        gt_angle = torch.deg2rad(torch.tensor([half_angle]))
        q = self.axis_angle_to_quaternions(axis_d, gt_angle)
        self.register_buffer('quaternions', q, persistent=False)
        self.register_buffer('axis_o', axis_o, persistent=False)
    
    def init_gt_motion(self, path):
        import json
        import numpy as np
        with open(path, mode='r') as f:
            meta = json.load(f)
            trans_info = meta['trans_info']
            f.close()
        axis_o = np.array(trans_info['axis']['o']).astype(np.float32)
        axis_d = np.array(trans_info['axis']['d']).astype(np.float32)
        frame_R_sapien = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        half_angle = (trans_info['rotate']['r'] - trans_info['rotate']['l']) * 0.5
        # half_translate = (trans_info['translate']['l'] + trans_info['translate']['r']) * 0.5
        
        axis_d = torch.matmul(frame_R_sapien, torch.tensor(axis_d)).float()
        axis_o = torch.matmul(frame_R_sapien, torch.tensor(axis_o)).float()
        gt_angle = torch.deg2rad(torch.tensor([half_angle]))
        q = self.axis_angle_to_quaternions(axis_d, gt_angle).float()

        self.axis_o = nn.Parameter(axis_o, requires_grad=True)
        self.quaternions = nn.Parameter(q, requires_grad=True) 

    def init_acute_diff_w_GT(self, path):
        import json
        import numpy as np
        with open(path, mode='r') as f:
            meta = json.load(f)
            trans_info = meta['trans_info']
            f.close()
        axis_o = np.array(trans_info['axis']['o']).astype(np.float32)
        axis_d = np.array(trans_info['axis']['d']).astype(np.float32)
        frame_R_sapien = torch.tensor([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
        half_angle = (trans_info['rotate']['r'] - trans_info['rotate']['l']) * 0.5
        # half_translate = (trans_info['translate']['l'] + trans_info['translate']['r']) * 0.5
        
        axis_d = torch.matmul(frame_R_sapien, torch.tensor(axis_d)).float()
        axis_o = torch.matmul(frame_R_sapien, torch.tensor(axis_o)).float()
        gt_angle = torch.deg2rad(torch.tensor([half_angle]))
        q_gt = self.axis_angle_to_quaternions(axis_d, gt_angle).float()

        def random_unit_vector():
            while True:    
                random_vector = torch.rand(3) * 2. - 1.
                norm = torch.linalg.norm(random_vector)
                if norm > 0.001:
                    return random_vector / norm

        def random_orthogonal_unit_vector(axis):
            while True:
                random_vector = random_unit_vector()
                orthogonal_vector = random_vector - torch.dot(random_vector, axis) * axis
                norm = torch.linalg.norm(orthogonal_vector)
                if norm > 0.001:
                    return orthogonal_vector / norm
                
        def quaternion_multiply(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
            x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
            y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
            z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
            return torch.tensor([w, x, y, z]).float()

        orthogonal_axis = random_orthogonal_unit_vector(axis_d)
        q_orthogonal = self.axis_angle_to_quaternions(orthogonal_axis, 0.1)
        q_composed = quaternion_multiply(q_gt, q_orthogonal)

        self.axis_o = nn.Parameter(axis_o, requires_grad=True)
        self.quaternions = nn.Parameter(q_composed, requires_grad=True) 
        

    def update_step(self, epoch, global_step):
        # progressive viewdir PE frequencies
        update_module_step(self.static_texture, epoch, global_step)
        update_module_step(self.dynamic_texture, epoch, global_step)
        
        def occ_eval_fn(x):
            density_s, _ = self.static_geometry(x)
            x_d = self.rigid_transform(x)
            density_d, _ = self.dynamic_geometry(x_d)
            if self.use_maximum:
                density = torch.max(density_s, density_d)
            else:
                density = density_s + density_d
            # approximate for 1 - torch.exp(-density[...,None] * self.render_step_size) based on taylor series
            return density[...,None] * self.render_step_size
        
        if self.training and self.config.grid_prune:
            # occ_thre is related to render_step_size and mc_thre
            # increase warmup_steps to get more chance to optimize the entire space
            # can apply adaptive alpha_thre to increase with the iteration to cut more noises
            self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn, occ_thre=self.occ_alpha_thre, warmup_steps=self.grid_warmup)

    def quaternion_to_axis_angle(self, quaternions):
        norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
        half_angles = torch.atan2(norms, quaternions[..., :1])
        angles = 2 * half_angles
        eps = 1e-6
        small_angles = angles.abs() < eps
        sin_half_angles_over_angles = torch.empty_like(angles)
        sin_half_angles_over_angles[~small_angles] = (
            torch.sin(half_angles[~small_angles]) / angles[~small_angles]
        )
        # for x small, sin(x/2) is about x/2 - (x/2)^3/6
        # so sin(x/2)/x is about 1/2 - (x*x)/48
        sin_half_angles_over_angles[small_angles] = (
            0.5 - (angles[small_angles] * angles[small_angles]) / 48
        )
        
        axis_angle = quaternions[..., 1:] / sin_half_angles_over_angles # the magnitude is the angle rad
        axis = torch.nn.functional.normalize(axis_angle, p=2, dim=0)
        return axis, angles
    
    def axis_angle_to_quaternions(self, axis, angle):
        half_angle = 0.5 * angle
        sin_ = sin(half_angle)
        cos_ = cos(half_angle)
        r = cos_
        i = axis[0] * sin_
        j = axis[1] * sin_
        k = axis[2] * sin_
        return torch.tensor([r, i, j, k], dtype=torch.float32)


    def get_init_axis(self):
        '''for the system to call for visualization'''
        axis_d, angle = self.quaternion_to_axis_angle(self.quaternions)
        axis_o = self.axis_o.data.detach()
        axis_d = axis_d.data.detach()
        print('angle', angle)
        print('axis_d', axis_d)
        return torch.cat([axis_o.unsqueeze(0), (axis_o+axis_d).unsqueeze(0)], dim=0)
   
    def isosurface(self):
        mesh_s = self.static_geometry.isosurface()
        mesh_d = self.dynamic_geometry.isosurface()
        return {'static': mesh_s, 'dynamic': mesh_d}
    
    def extract_volume(self, res):
        volume_s = self.static_geometry.extract_volume(res)
        volume_d = self.dynamic_geometry.extract_volume(res)
        return {'static': volume_s.cpu(), 'dynamic': volume_d.cpu()}
    
    def rot_axis_angle(self, k, theta):
        k = torch.nn.functional.normalize(k, p=2., dim=0)
        kx, ky, kz = k[0], k[1], k[2]
        cos, sin = torch.cos(theta), torch.sin(theta)
        R = torch.zeros((3, 3))
        R[0, 0] = cos + (kx**2) * (1 - cos)
        R[0, 1] = kx * ky * (1 - cos) - kz * sin
        R[0, 2] = kx * kz * (1 - cos) + ky * sin
        R[1, 0] = kx * ky * (1 - cos) + kz * sin
        R[1, 1] = cos + (ky**2) * (1 - cos)
        R[1, 2] = ky * kz * (1 - cos) - kx * sin
        R[2, 0] = kx * kz * (1 - cos) - ky * sin
        R[2, 1] = ky * kz * (1 - cos) + kx * sin
        R[2, 2] = cos + (kz**2) * (1 - cos)
        return R.to(k)
    
    def init_quaternions(self, half_angle, init_dir):
        a = torch.tensor([init_dir[0], init_dir[1], init_dir[2]], dtype=torch.float32)
        # a = torch.tensor([1., 1., 1.], dtype=torch.float32)
        a = torch.nn.functional.normalize(a, p=2., dim=0)
        sin_ = sin(half_angle)
        cos_ = cos(half_angle)
        r = cos_
        i = a[0] * sin_
        j = a[1] * sin_
        k = a[2] * sin_
        q = torch.tensor([r, i, j, k], dtype=torch.float32)
        return q
    
    def quaternion_to_matrix(self, q):
        """
        Convert rotations given as quaternions to rotation matrices.
        Args:
            quaternions: quaternions with real part first,
                as tensor of shape (..., 4).
        Returns:
            Rotation matrices as tensor of shape (..., 3, 3).
        """

        q = torch.nn.functional.normalize(q, p=2., dim=0)
        r, i, j, k = torch.unbind(q, -1)
        two_s = 2.0 / (q * q).sum(-1)
        R = torch.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            -1,
        )
        R = R.reshape((3, 3))
        return R.to(q)

    def rigid_transform(self, positions, state=0.):
        '''
        Perform the rigid transformation: R_axis_d,rot_angle(center=axis_o) @ x + t

        Transform the positions from state=0 or state=1 to canonical state=0.5

        The rot_angle is defined as the half angle from start to end state.

        The canonical state is t=0.5, so scaling of rot_angle and translation 
        from t=0 to t=0.5 is positive, from t=1 to t=0.5 iss negative
        '''
        # # to mimic STaR
        # if state == 0.:
        #     return positions
        # else:
        #     R = self.quaternion_to_matrix(self.quaternions)
        #     positions = torch.matmul(R, positions.T).T
        #     positions = positions + self.translation
        #     return positions
        
        scaling = (self.canonical - state) / self.canonical
        # positions = positions - self.axis_o
        if scaling == 1.:
            R = self.quaternion_to_matrix(self.quaternions)
            positions = torch.matmul(R, positions.T).T
            positions = positions + self.translation
        elif scaling == -1.:
            # linalg.solve(A, B) == linalg.inv(A) @ B
            # It is always prefered to use solve() when possible, 
            # as it is faster and more numerically stable than computing the inverse explicitly.
            # positions = torch.linalg.solve(R, positions.T).T
            positions = positions - self.translation
            inv_sc = torch.tensor([1., -1., -1., -1]).to(self.quaternions)
            inv_q = inv_sc * self.quaternions
            R = self.quaternion_to_matrix(inv_q)
            positions = torch.matmul(R, positions.T).T
        else:
            axis, angle = self.quaternion_to_axis_angle(self.quaternions) # the angle means from t=0 to t=0.5
            tgt_angle = scaling * angle
            R = self.rot_axis_angle(axis, tgt_angle)
            positions = torch.matmul(R, positions.T).T
        # axis, angle = self.quaternion_to_axis_angle(self.quaternions) # the angle means from t=0 to t=0.5
        # tgt_angle = scaling * angle
        # R = self.rot_axis_angle(axis, tgt_angle)
        # positions = torch.matmul(R, positions.T).T

        # positions = positions + self.axis_o
        # translation

        
        return positions
    
    def swipe_transform(self, p, tgt_state=0.):
        '''
        transform from t=0.5 to t=tgt_state
        '''
        axis, angle = self.quaternion_to_axis_angle(self.quaternions) # the angle means from t=0 to t=0.5
        tgt_angle = (self.canonical - tgt_state) * 2. * angle
        R = self.rot_axis_angle(axis, tgt_angle)
        p = p - self.axis_o
        p = torch.matmul(R, p.T).T
        p = p + self.axis_o
        return p

    @torch.no_grad()
    def extract_aabb(self, res=50, thre=0.3):
        points, volume_s = self.static_geometry.extract_volume(res)
        points = scale_anything(points, (0, 1), (-self.config.radius, self.config.radius))
        _, volume_d = self.dynamic_geometry.extract_volume(res)
        point_s = points[volume_s > thre]
        if point_s.shape[0] == 0:
            bb_max_s = torch.tensor([self.config.radius, self.config.radius, self.config.radius])
            bb_min_s = torch.tensor([-self.config.radius, -self.config.radius, -self.config.radius])
        else:
            bb_max_s, bb_min_s = torch.max(point_s, dim=0)[0], torch.min(point_s, dim=0)[0]

        point_d = points[volume_d > thre]
        if point_d.shape[0] == 0:
            bb_max_d = torch.tensor([self.config.radius, self.config.radius, self.config.radius])
            bb_min_d = torch.tensor([-self.config.radius, -self.config.radius, -self.config.radius])
        else:
            bb_max_d, bb_min_d = torch.max(point_d, dim=0)[0], torch.min(point_d, dim=0)[0]
        return {
            'static': {
                'max': bb_max_s.to(self.rank),
                'min': bb_min_s.to(self.rank)
            },
            'dynamic':{
                'max': bb_max_d.to(self.rank),
                'min': bb_min_d.to(self.rank)
            }
        }
    
    @torch.no_grad()
    def extract_state_aabb(self, res=50, thre=5.0):
        def aabb_max_min(points):
            if point_s.shape[0] == 0:
                bb_max_s = torch.tensor([1e-5, 1e-5, 1e-5])
                bb_min_s = torch.tensor([-1e-5, -1e-5, -1e-5])
            else:
                bb_max_s, bb_min_s = torch.max(points, dim=0)[0], torch.min(points, dim=0)[0]
            return bb_max_s, bb_min_s

        points, volume_s = self.static_geometry.extract_volume(res)
        points = scale_anything(points, (0.02, 0.98), (-self.config.radius, self.config.radius))
        _, volume_d = self.dynamic_geometry.extract_volume(res)

        point_s = points[volume_s > thre]
        bb_max_s, bb_min_s = aabb_max_min(point_s)

        point_d = points[volume_d > thre]
        point_d_0 = self.rigid_transform(point_d, state=0)
        point_d_1 = self.rigid_transform(point_d, state=1)
        bb_max_d_0, bb_min_d_0 = aabb_max_min(point_d_0)
        bb_max_d_1, bb_min_d_1 = aabb_max_min(point_d_1)

        
        return {
            'static': {
                'max': bb_max_s.to(self.rank),
                'min': bb_min_s.to(self.rank)
            },
            'state_0':{
                'max': bb_max_d_0.to(self.rank),
                'min': bb_min_d_0.to(self.rank)
            },
            'state_1':{
                'max': bb_max_d_1.to(self.rank),
                'min': bb_min_d_1.to(self.rank)
            }
        }

    def acc_along_rays(self, src, ray_indices, n_rays):
        '''src: weighted sum of some value'''
        assert ray_indices.dim() == 1 

        if ray_indices.numel() == 0:
            assert n_rays is not None
            return torch.zeros((n_rays, src.shape[-1]), device=src.device)
            
        if n_rays is None:
            n_rays = int(ray_indices.max()) + 1
        else:
            assert n_rays > ray_indices.max()

        ray_indices = ray_indices.int()
        index = ray_indices[:, None].long().expand(-1, src.shape[-1])
        outputs = torch.zeros((n_rays, src.shape[-1]), device=src.device)
        outputs.scatter_add_(0, index, src)
        return outputs

    
    def forward_(self, rays, scene_state):
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)

        def get_distortion_loss(t_starts, t_ends, ray_indices, weights):
            ray_indices = ray_indices.long()
            midpoints = 0.5 * (t_starts + t_ends)
            interval = t_ends - t_starts
            if ray_indices.shape[0] == 0:
                return torch.zeros([], requires_grad=True, device=ray_indices.device)
            loss = flatten_eff_distloss(weights, midpoints, interval, ray_indices)
            return loss

        def sigma_fn_composite(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            sigma_s, _ = self.static_geometry(positions)
            positions = self.rigid_transform(positions, scene_state)
            sigma_d, _ = self.dynamic_geometry(positions)
            # sigma = torch.maximum(sigma_s, sigma_d) # temp max
            if self.use_maximum:
                sigma = torch.maximum(sigma_s, sigma_d)
            else:
                sigma = sigma_s + sigma_d
            return sigma[...,None]
    
        def rgb_sigma_fn_static(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            density, feature = self.static_geometry(positions) 
            rgb = self.static_texture(feature, t_dirs)
            return rgb, density[...,None]
        
        def rgb_sigma_fn_dynamic(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            positions = self.rigid_transform(positions, scene_state)
            density, feature = self.dynamic_geometry(positions) 
            dirs_d = self.rigid_transform(t_dirs, scene_state)
            rgb = self.dynamic_texture(feature, dirs_d)
            return rgb, density[...,None]
        
        def sigma_ratio(t_starts, t_ends, ray_indices):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions1 = t_origins + t_dirs * (t_starts + t_ends) / 2.
            sigma_s1, _ = self.static_geometry(positions1)
            sigma_d1, _ = self.dynamic_geometry(positions1)
            positions2 = self.rigid_transform(positions1, scene_state)
            sigma_s2, _ = self.static_geometry(positions2)
            sigma_d2, _ = self.dynamic_geometry(positions2)
            return sigma_s1[..., None], sigma_d1[..., None], sigma_s2[..., None], sigma_d2[..., None]
        
        def sigma_fn_swipe_surface(t_surface, n_interval=3):
            # swipe to a 0.25 angle from canonical to the scene state
            p_can = rays_o + t_surface * rays_d
            sigma_d, _ = self.dynamic_geometry(p_can)
            interval = 1./n_interval
            swi_points = []
            for i in range(1, n_interval + 1):
                p_swi = self.swipe_transform(p_can, i * interval * (scene_state + 0.5))
                swi_points.append(p_swi)
            ps_swi = torch.cat(swi_points, dim=0)
            sigma_s, _ = self.static_geometry(ps_swi)
            sigma_d = sigma_d.repeat(n_interval)
            return sigma_s, sigma_d
        
        def signma_fn_swipe(t_starts, t_ends, ray_indices, sigma_s, sigma_d, n_interval=10):
            sigma_s, sigma_d = sigma_s.squeeze(-1), sigma_d.squeeze(-1)
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            base = 0 if scene_state==0 else 0.5
            swi_points = []
            interval = 1./n_interval
            for i in range(0, n_interval):
                p_swi = self.swipe_transform(positions, 0.5 * i * interval + base)
                swi_points.append(p_swi)
            ps_swi = torch.cat(swi_points, dim=0)
            # chunk = int(ps_swi.shape[0]/2)
            # sigma_s_swi_1, _ = self.static_geometry(ps_swi[:chunk])
            # sigma_s_swi_2, _ = self.static_geometry(ps_swi[chunk:])
            # sigma_s_swi = torch.cat([sigma_s_swi_1, sigma_s_swi_2], dim=0)
            sigma_s_swi, _ = self.static_geometry(ps_swi)
            sigma_s = torch.cat([sigma_s, sigma_s_swi], dim=0)
            sigma_d = sigma_d.repeat(n_interval + 1)
            return sigma_s, sigma_d
        
        def sigma_fn_intersect(res=10):
            grid_sc = res * self.render_step_size
            x = torch.linspace(0., 1., steps=res)
            y = torch.linspace(0., 1., steps=res)
            z = torch.linspace(0., 1., steps=res)
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
            points = torch.cat((grid_x[..., None], grid_y[..., None], grid_z[..., None]), dim=3).to(self.rank) # (res, res, res, 3)
            ps_cub = torch.reshape(points * grid_sc + self.axis_o - 0.5 * self.render_step_size * res, (-1, 3))
            sigma_s_cub, _ = self.static_geometry(ps_cub)
            sigma_d_cub, _ = self.dynamic_geometry(ps_cub)
            return sigma_s_cub, sigma_d_cub
        
        def signma_fn_neighbor(t_starts, t_ends, ray_indices, pt_idx, res=5, ada=True, fix_len=0.03):
            @torch.no_grad()
            def ada_cube_size(pt, min_len=0.005, max_len=0.02, ada=True, fix_len=0.03):
                dist2axis = distance_point_to_line(pt, self.axis_o, self.quaternions)
                ##############################################################################################
                # should not use the diagonal to normalize the dist2axis because d2a_norm is hard to reach 0.5
                # this will make the weights for minimizing the static part > the ones for minimizing dynamic part
                # diagonal = 2.8845 * self.config.radius # the length of the diagonal as 2 * sqrt(3)* radius
                # d2a_norm = dist2axis / diagonal
                ##############################################################################################
                range_near_axis = 0.1 # threshold 5cm as the near axis region
                d2a_norm = dist2axis / range_near_axis
                d2a_norm = torch.clamp(d2a_norm, min=0., max=8.) # make the weights for outside_axis_region as 1.

                if ada:
                    grid_size = d2a_norm * (max_len - min_len) + min_len
                else: 
                    grid_size = torch.ones_like(d2a_norm) * fix_len
                return d2a_norm, grid_size

            def prepare_cube(pt, res, grid_size):
                # construct a grid with randomness
                RAND = ((grid_size / res)).repeat(1, res**3)[..., None] # 50% randomness around the corners of the cells
                x = torch.linspace(0., 1., steps=res)
                # x += torch.randn_like(x) * RAND
                y = torch.linspace(0., 1., steps=res)
                # y += torch.randn_like(y) * RAND
                z = torch.linspace(0., 1., steps=res)
                # z += torch.randn_like(z) * RAND
                grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
                cub = torch.cat((grid_x[..., None], grid_y[..., None], grid_z[..., None]), dim=3).to(self.rank).view(-1, 3).repeat(pt.shape[0], 1, 1) # (n_ray, res**3, 3)
                grid_sc = grid_size.repeat(1, res ** 3)[..., None] # (n_ray, res**3, 1)
                rand = (torch.randn((pt.shape[0], res ** 3, 3))).to(self.rank) * RAND # (n_ray, res**3, 3)
                rand_cub = (cub * grid_sc + rand).reshape(-1, 3)  # (n_ray * res**3, 3)
                offset = pt.repeat(res ** 3, 1)
                pts_cub = rand_cub + offset - 0.5 * grid_sc.view(-1, 1)
                return pts_cub

            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            # filter points with pt_idx
            pt = positions[pt_idx]
            d2a, grid_size = ada_cube_size(pt, ada=ada, fix_len=fix_len)
            # static points
            pts_cub_s = prepare_cube(pt, res, grid_size)
            sigma_s_cub, _ = self.static_geometry(pts_cub_s)
            # dynamic points
            pt_d = self.rigid_transform(pt, scene_state)
            pts_cub_d = prepare_cube(pt_d, res, grid_size)
            sigma_d_cub, _ = self.dynamic_geometry(pts_cub_d)
            # max grid size
            max_grid_size = grid_size.max()
            return sigma_s_cub, sigma_d_cub, d2a, max_grid_size
        
        def signma_fn_neighbor_old(t_starts, t_ends, ray_indices, pt_idx, res=10, cube_len=0.001):
            ray_indices = ray_indices.long()
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.
            # filter points with pt_idx
            pt = positions[pt_idx]
            # construct a grid with randomness
            grid_sc = res * cube_len
            x = torch.linspace(0., 1., steps=res)
            x += torch.randn_like(x) * 1e-4
            y = torch.linspace(0., 1., steps=res)
            y += torch.randn_like(y) * 1e-4
            z = torch.linspace(0., 1., steps=res)
            z += torch.randn_like(z) * 1e-4
            grid_x, grid_y, grid_z = torch.meshgrid(x, y, z, indexing='ij')
            cub = torch.cat((grid_x[..., None], grid_y[..., None], grid_z[..., None]), dim=3).to(self.rank).view(-1, 3).repeat(pt.shape[0],1) # (res, res, res, 3)
            # static points
            offset = pt.repeat(res * res * res, 1)
            pts_cub_s = torch.reshape(cub * grid_sc + offset - 0.5 * grid_sc, (-1, 3))
            sigma_s_cub, _ = self.static_geometry(pts_cub_s)
            # dynamic points
            pt_d = self.rigid_transform(pt, scene_state)
            offset_d = pt_d.repeat(1, res * res * res, 1)
            pts_cub_d = torch.reshape(cub * grid_sc + offset_d - 0.5 * grid_sc, (-1, 3))
            sigma_d_cub, _ = self.dynamic_geometry(pts_cub_d)
            return sigma_s_cub, sigma_d_cub, pt_d.detach()
        
        def argmax_per_ray(src, ray_indices, n_rays):
            '''src: weighted sum of some value'''
            assert ray_indices.dim() == 1 

            if ray_indices.numel() == 0:
                assert n_rays is not None
                return torch.zeros((n_rays, src.shape[-1]), device=src.device)
                
            if n_rays is None:
                n_rays = int(ray_indices.max()) + 1
            else:
                assert n_rays > ray_indices.max()

            ray_indices = ray_indices.int()
            index = ray_indices[:, None].long().expand(-1, src.shape[-1])
            # for i in range(n_rays):
            #     ray_ind, _ = torch.where(index==i)
            #     values = src[ray_ind]
            #     max_ind = ray_ind[torch.argmax(values)]
           
            # Create a mask with shape (n_rays, N)
            mask = (index.T == torch.arange(n_rays)[:, None].to(self.rank))

            # Calculate the maximum values and indices for each ray
            _, max_indices = torch.max((mask * src.T + (~mask).float() * -1e10).T, dim=0)
            return max_indices
        
        def distance_point_to_line(P, A, q):
            '''
                args:
                    * P: the point to compute distance, shape (N, 3)
                    * A: a point on the line, here we give the axis origin, shape (3,)
                    * q: the quaternion of the rotation, we will convert it to the direction of the axis, shape (4,)
            '''
            d, _ = quaternion_to_axis_angle(q)
            # Ensure d is a unit vector
            d = d / torch.norm(d)
            # Compute the vector AP
            AP = P - A
            # Compute the projection of AP onto d
            # projection = torch.dot(AP, d) * d
            projection = torch.matmul(AP, d.unsqueeze(-1)) * d.unsqueeze(0)
            # Compute the orthogonal vector V from point P to the line
            V = P - projection
            # Compute the distance as the length of the orthogonal vector V
            # distance = torch.norm(V)
            distance = torch.norm(V, dim=1, keepdim=True)

            return distance

        def composite_rendering(ray_indices, t_starts, t_ends):
            '''
            The implementation from original nerf for reference:

            # Add noise to model's predictions for density. Can be used to 
            # regularize network during training (prevents floater artifacts).
            noise = 0.
            if raw_noise_std > 0.:
                noise = tf.random.normal(raw[..., 3].shape) * raw_noise_std

            # Predict density of each sample along each ray. Higher values imply
            # higher likelihood of being absorbed at this point.
            alpha = raw2alpha(raw[..., 3] + noise, dists)  # [N_rays, N_samples]

            About Transmittance:
            - original nerf use
                weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
            - in pytorch, the equivalent implementation is 
                weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
            '''
            n_rays = rays_o.shape[0]
            rgb_s, sigma_s = rgb_sigma_fn_static(t_starts, t_ends, ray_indices)
            rgb_d, sigma_d = rgb_sigma_fn_dynamic(t_starts, t_ends, ray_indices)

            dists = t_ends - t_starts
            alpha_s = 1. - torch.exp(-sigma_s * dists)
            alpha_d = 1. - torch.exp(-sigma_d * dists)

            if self.cr_alpha_thre > 0.: # can smooth the surface a bit
                alpha_s = alpha_s * (alpha_s > self.cr_alpha_thre)
                alpha_d = alpha_d * (alpha_d > self.cr_alpha_thre)
            
            if self.use_maximum: # terminate the ray with the larger one only
                sigma_max, idx = torch.max(torch.cat([sigma_s, sigma_d], dim=1), dim=1)
                mask_d = idx[..., None]
                mask_s = torch.ones_like(mask_d) - idx[..., None]
                alpha_max = 1. - torch.exp(-sigma_max[...,None] * dists) 
                Ts = render_transmittance_from_alpha(alpha_max, ray_indices=ray_indices)
                if self.cr_early_stop_eps > 0.:
                    Ts = Ts * (Ts > self.cr_early_stop_eps)
                weights_s = alpha_s * Ts * mask_s
                weights_d = alpha_d * Ts * mask_d
            else:
                alpha_add = 1. - (1. - alpha_s) * (1. - alpha_d)
                Ts = render_transmittance_from_alpha(alpha_add, ray_indices=ray_indices)
                if self.cr_early_stop_eps > 0.:
                    Ts = Ts * (Ts > self.cr_early_stop_eps)
                weights_s = alpha_s * Ts
                weights_d = alpha_d * Ts
            
            weights = weights_s + weights_d
            # opacity
            opacity = self.acc_along_rays(weights, ray_indices, n_rays)
            opacity = opacity.squeeze(-1)
            # acc color
            rgb = weights_s * rgb_s + weights_d * rgb_d
            rgb = self.acc_along_rays(rgb, ray_indices, n_rays)
            # Background composition.
            if self.config.white_bkgd:
                rgb = rgb + self.background_color * (1. - opacity[..., None])
            # regularization
            # ratio = None
            ratio = (sigma_s > 1.).detach() * sigma_d.detach() / torch.clamp_min(sigma_s + sigma_d.detach(), 1e-10) # only reg static
            
            # s1, d1, s2, d2 = sigma_ratio(t_starts, t_ends, ray_indices)
            # r1 = d1 / torch.clamp_min(s1 + d1, 1e-10)
            # r2 = d2 / torch.clamp_min(s2 + d2, 1e-10)
            # ratio = torch.concat([r1, r2])
            
            
            # validation and testing
            if not self.training:
                # depth
                depth = weights * ((t_starts + t_ends) * 0.5)
                depth = self.acc_along_rays(depth, ray_indices, n_rays)
                depth = depth.squeeze(-1)

                rgb_s_only, opacity_s, depth_s_only = rendering(t_starts, t_ends, ray_indices, n_rays,
                                                                rgb_sigma_fn=rgb_sigma_fn_static, 
                                                                render_bkgd=self.background_color)
                rgb_d_only, opacity_d, depth_d_only = rendering(t_starts, t_ends, ray_indices, n_rays,
                                                                rgb_sigma_fn=rgb_sigma_fn_dynamic, 
                                                                render_bkgd=self.background_color)
                return {
                    'rgb': rgb, 
                    'opacity': opacity, 
                    'depth': depth,  
                    'rgb_s': rgb_s_only, 
                    'rgb_d': rgb_d_only, 
                    'depth_s': depth_s_only, 
                    'depth_d': depth_d_only, 
                    'opacity_s': opacity_s, 
                    'opacity_d': opacity_d, 
                }
            
            # regularization on part mask
            part_mask_ratio = None
            if self.use_part_mask:
                opacity_s = self.acc_along_rays(weights_s, ray_indices, n_rays)
                opacity_d = self.acc_along_rays(weights_d, ray_indices, n_rays)
                part_mask_ratio = opacity_s / torch.clamp_min(opacity_s + opacity_d, 1e-10)
            
            # # if self.use_depth_reg:
            # if True:
            #     depth_s = weights_s * ((t_starts + t_ends) * 0.5)
            #     depth_s = self.acc_along_rays(depth_s, ray_indices, n_rays)
            #     depth_d = weights_d * ((t_starts + t_ends) * 0.5)
            #     depth_d = self.acc_along_rays(depth_d, ray_indices, n_rays)
            #     ind = torch.logical_and((depth_s > 0.1) , (depth_d > 0.1)) 
            #     if ind.sum().item() == 0:
            #         dep_reg = torch.tensor(0.).to(self.rank)
            #     else:
            #         dep_reg = torch.mean(torch.max(3. - ((depth_s[ind] - depth_d[ind]) ** 2.)**0.5, torch.tensor(0.)))
            #     if torch.isnan(dep_reg):
            #         import pdb
            #         pdb.set_trace()
            
            # not working, too much memory
            distortion_loss = None
            # if self.use_distortion:
            #     distortion_loss = get_distortion_loss(t_starts, t_ends, ray_indices, weights_s)
            #     distortion_loss += get_distortion_loss(t_starts, t_ends, ray_indices, weights_d)
            #     distortion_loss = 0.5 * distortion_loss
            
            # swipe surface
            # Ts_d = render_transmittance_from_alpha(alpha_d, ray_indices=ray_indices)
            # t_surface_d = self.acc_along_rays(Ts_d * alpha_d, ray_indices, n_rays) # depth of dynamic field
            # sigma_s_swi, sigma_d_swi = sigma_fn_swipe_surface(t_surface_d, n_interval=10)
            # mask_surface = sigma_d_swi > 1.0
            # if mask_surface.sum().data.item() == 0:
            #     ratio_swi = torch.ones(1)
            # else:
            #     print('here we have a ratio_swi', mask_surface.sum().data.item())
            #     valid_d, valid_s = sigma_d_swi[mask_surface], sigma_s_swi[mask_surface]
            #     ratio_swi = valid_d / (valid_d + valid_s)

            # old implementation
            # sigma_s_swi, sigma_d_swi = signma_fn_swipe(t_starts, t_ends, ray_indices, sigma_s, sigma_d)
            # ratio_swi = sigma_d_swi * sigma_d_swi / torch.clip(sigma_s_swi + sigma_d_swi, 1e-13)

            weight_swi, ratio_swi = None, None
            # if self.use_swi:
            #     sigma_s_swi, sigma_d_swi = signma_fn_swipe(t_starts, t_ends, ray_indices, sigma_s, sigma_d)
            #     # ratio_swi = sigma_d_swi / torch.clip(sigma_s_swi + sigma_d_swi, 1e-13)
            #     sigma_d_min, sigma_d_max = 0.5, 3.
            #     weight_swi = torch.clamp((sigma_d_swi - sigma_d_min) / (sigma_d_max - sigma_d_min), 0., 1.)
            #     # only punish sigma_s_swi
            #     ratio_swi = torch.clamp(sigma_s_swi - 0.5, min=0., max=10.)
            
            
            entropy_both, entropy_xor = None, None
            # if self.use_intersect:
            #     sig_s_cub, sig_d_cub = sigma_fn_intersect(30)
            #     mask_s, mask_d = sig_s_cub > 1.0, sig_d_cub > 1.0
            #     sum_s, sum_d = mask_s.sum().float().requires_grad_(), mask_d.sum().float().requires_grad_()
            #     ratio = torch.clip(sum_s / torch.clip(sum_s + sum_d, min=1e-13), 1e-5, 1.-1e-5)
            #     entropy_both = torch.exp(ratio * torch.log(ratio) + (1.-ratio) * torch.log(1.-ratio))
            #     xor = mask_s * mask_d * sig_d_cub / torch.clip(sig_d_cub + sig_s_cub, min=1e-13)
            #     xor = torch.clip(xor, 1e-5, 1.-1e-5)
            #     entropy_xor = (-(xor * torch.log(xor) + (1.-xor) * torch.log(1.-xor))).mean()

            align_reg_s, align_reg_d, d2a_max, max_grid_size = None, None, None, None
            if self.use_align_noi:
                # find solid dynamic point per ray
                with torch.no_grad():
                    max_w_idx_d = argmax_per_ray(weights_d, ray_indices, n_rays)
                ################################################################
                # config cube size as cub_res**3, cube_len is the fixed cube size if not using ada
                use_ada = True
                cub_res, cube_len = 5, 0.03
                # query neighborhood [sig_s_cub, sig_d_cub are require_grad, pt_d is detached]
                sig_s_cub, sig_d_cub, d2a, max_grid_size = signma_fn_neighbor(t_starts, t_ends, ray_indices, 
                                                                              max_w_idx_d, 
                                                                              res=cub_res, 
                                                                              ada=use_ada, 
                                                                              fix_len=cube_len)
                d2a_max = d2a.max()
                # filter out the point with density lower than 1
                sig_d_cub_ = sig_d_cub.detach()
                solid_d = (sig_d_cub_ * (sig_d_cub_ > 1.)).reshape(n_rays, -1).clamp_max(100.)
                # solid_d_mean = (sig_d_cub_[sig_d_cub_>1.]).mean()
                # solid_d_mean = solid_d.clamp_max(20.).mean() / 20.
                sig_s_cub_ = sig_s_cub.detach()
                solid_s = (sig_s_cub_ * (sig_s_cub_ > 1.)).reshape(n_rays, -1).clamp_max(100.)
                # solid_s_mean = (sig_s_cub_[sig_s_cub_>1.]).mean()
                # solid_s_mean = solid_s.mean()
                # solid_s_mean = solid_s.clamp_max(20.).mean() / 20.

                def avg_non_zero_per_row(src):
                    '''Given a tensor in shape (A, B), return the average of non-zero values per row as (A, 1)'''
                    non_zero_values = torch.where(src != 0, src, torch.tensor(float('nan')).to(self.rank))
                    non_zero_counts = torch.count_nonzero(src, dim=1).unsqueeze(1)
                    # Calculate the sum of non-zero values for each row
                    sum_per_row = torch.nansum(non_zero_values, dim=1, keepdim=True)
                    # Calculate the average of non-zero values for each row
                    average_per_row = sum_per_row / (non_zero_counts + 1e-10)
                    return average_per_row

                
                solid_s_avg_per_ray = avg_non_zero_per_row(solid_s)
                solid_d_avg_per_ray = avg_non_zero_per_row(solid_d)


                # strategy: never overlapping in 3D within a small neighborhood
                # if near the axis, then favors the dynamic field when overlapping; otherwise, favors the static field (by minimizing the density)
                # the loss is linearly proportional to the distance to the axis
                sig_s_cub = sig_s_cub.reshape(n_rays, -1).clamp_max(100.)
                sig_d_cub = sig_d_cub.reshape(n_rays, -1).clamp_max(100.)

                align_reg_s = torch.mean((1. - d2a).clamp_min_(0.) * solid_d_avg_per_ray * sig_s_cub, dim=1).mean()
                align_reg_d = torch.mean((d2a * (d2a > 1.) / 8.) * solid_s_avg_per_ray * sig_d_cub, dim=1).mean()  
                       
                ################################################################

                # ################################################################
                #  # config cube size as cub_res * cub_res * cub_res
                # cub_res = 5
                # # query neighborhood [sig_s_cub, sig_d_cub are require_grad, pt_d is detached]
                # sig_s_cub, sig_d_cub, pt_d = signma_fn_neighbor_old(t_starts, t_ends, ray_indices, max_w_idx_d, res=cub_res, cube_len=0.004)
                
                # # loss
                # # diagonal = 2.8845 * self.config.radius # the length of the diagonal as 2 * sqrt(3)* radius
                # diagonal = self.config.radius # the length of the diagonal as radius

                # with torch.no_grad():
                #     dist2axis = distance_point_to_line(pt_d, self.axis_o, self.quaternions)
                #     solid_s = sig_s_cub > 1
                #     solid_d = sig_d_cub > 1
                #     dist2axis_norm = dist2axis / diagonal
                
                # # strategy: never overlapping in 3D within a small neighborhood
                # # if near the axis, then favors the dynamic field when overlapping; otherwise, favors the static field (by minimizing the density)
                # # the loss is linearly pproportional to the distance to the axis
                # align_reg_s = ((1.-dist2axis_norm) * solid_d * sig_d_cub.detach() * sig_s_cub).squeeze(1).mean()
                # align_reg_d = ( dist2axis_norm * solid_s * sig_s_cub.detach() * sig_d_cub).squeeze(1).mean()
                # ################################################################

                ################################################################
                # hard mask version of the non-overlapping constraint depending on if it is near_axis
                # near_axis = dist2axis < 0.5 * cub_res * 0.001
                # align_reg_s = (near_axis * dist2axis * solid_d * sig_d_cub.detach() * sig_s_cub).squeeze(1).mean()
                # align_reg_d = (~near_axis * solid_s * sig_s_cub.detach() * sig_d_cub).squeeze(1).mean()
                ################################################################

                ################################################################
                # larger patch to consider the neighborhood
                # cub_res_nei = 8
                # cub_size_nei = cub_res_nei * cub_res_nei * cub_res_nei
                # with torch.no_grad(): 
                #     sig_s_cub_rec, sig_d_cub_rec, _ = signma_fn_neighbor(t_starts, t_ends, ray_indices, max_w_idx_d, res=cub_res_nei)
                #     near_axis_rec = dist2axis < cub_res_nei * 0.001
                #     neighbor_s = (sig_s_cub_rec.view(-1, cub_size_nei).mean(dim=1) / sig_d_cub_rec.view(-1, cub_size_nei).mean(dim=1) + 1e-10) > 0.5
                #     neighbor_s.unsqueeze(1).repeat(1, 125).flatten()
                # align_reg = (neighbor_s * (~near_axis * solid_s * sig_s_cub.detach() * sig_d_cub).squeeze(1).mean()).mean()
                # align_reg_d = (~near_axis * solid_s * sig_s_cub.detach() * sig_d_cub).squeeze(1).mean()
                # align_reg_s = ~neighbor_s * ((near_axis * solid_d * sig_d_cub.detach() * sig_s_cub).squeeze(1).mean()).mean()

                ################################################################

                



                ####################################!

            return {
                'rgb': rgb, 
                'ratio': ratio.squeeze(-1), 
                'rays_valid': opacity > 0, 
                'opacity': opacity,
                'distortion_loss': distortion_loss,
                'ratio_swi': ratio_swi,
                'weight_swi': weight_swi,
                'entropy_both': entropy_both,
                'entropy_xor': entropy_xor,
                'part_mask_ratio': part_mask_ratio,
                'align_reg_d': align_reg_d,
                'align_reg_s': align_reg_s,
                'd2a': d2a_max,
                'max_grid_size': max_grid_size,
                # 'dep_reg': dep_reg,
                # 'n_idx': ind.sum().item()
            }

        with torch.no_grad():
            # packed_info, t_starts, t_ends = ray_marching(
            # [edit] nerfacc 0.3.2
            ray_indices, t_starts, t_ends, _ = ray_marching(
                rays_o, rays_d,
                scene_aabb=self.scene_aabb,
                grid=self.occupancy_grid if self.config.grid_prune else None,
                sigma_fn=sigma_fn_composite,
                near_plane=None, far_plane=None,
                render_step_size=self.render_step_size,
                stratified=self.randomized,
                cone_angle=0.0,
                alpha_thre=self.rm_alpha_thre, # default 0.0
                early_stop_eps=self.rm_early_stop_eps
            )
        render_out = composite_rendering(ray_indices, t_starts, t_ends)

        # render_out = composite_rendering(packed_info, t_starts, t_ends)
        if self.training:
            return {
                'comp_rgb': render_out['rgb'],
                'ratio': render_out['ratio'],
                'opacity': render_out['opacity'],
                'rays_valid': render_out['rays_valid'],
                'num_samples': torch.as_tensor([len(t_starts)], dtype=torch.int32, device=rays.device),
                'distortion_loss': render_out['distortion_loss'],
                'ratio_swi': render_out['ratio_swi'],
                'weight_swi': render_out['weight_swi'],
                'entropy_both': render_out['entropy_both'],
                'entropy_xor': render_out['entropy_xor'],
                'part_mask_ratio': render_out['part_mask_ratio'],
                'align_reg_d': render_out['align_reg_d'],
                'align_reg_s': render_out['align_reg_s'],
                'd2a': render_out['d2a'],
                'max_grid_size': render_out['max_grid_size'],
                # 'dep_reg': render_out['dep_reg'],
                # 'n_idx': render_out['n_idx'],

            }

        return {
            'comp_rgb': render_out['rgb'],
            'opacity': render_out['opacity'],
            'depth': render_out['depth'],
            'rgb_s': render_out['rgb_s'],
            'rgb_d': render_out['rgb_d'],
            'depth_s': render_out['depth_s'],
            'depth_d': render_out['depth_d'],
            'opacity_s': render_out['opacity_s'],
            'opacity_d': render_out['opacity_d'],
        }


    def forward(self, rays_0, rays_1):
        if self.training:
            out_0 = self.forward_(rays_0, scene_state=0.)
            out_1 = self.forward_(rays_1, scene_state=1.)
        else:
            out_0 = chunk_batch(self.forward_, self.config.ray_chunk, rays_0, scene_state=0.)
            out_1 = chunk_batch(self.forward_, self.config.ray_chunk, rays_1, scene_state=1.)
            del rays_0, rays_1
        return [{**out_0}, {**out_1}]

    def train(self, mode=True):
        self.randomized = mode and self.config.randomized
        return super().train(mode=mode)
    
    def eval(self):
        self.randomized = False
        return super().eval()
    
    def regularizations(self, outs):
        losses = {}
        # losses.update({'axis_d': torch.abs(torch.norm(self.axis_d) - 1.)})
        # losses.update({'translation': torch.norm(self.translation) - self.config.radius})
        # losses.update({'axis_o': torch.sum(torch.pow(self.axis_o, exponent=2), dim=-1)})
        return losses