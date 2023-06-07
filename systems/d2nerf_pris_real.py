import torch
import torch.nn.functional as F
from os.path import join, dirname
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_debug

from models.ray_utils import get_rays
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, binary_cross_entropy,  axis_error, SSIM
from systems.utils import parse_optimizer, parse_scheduler, load_gt_axis_real, proj2img, load_gt_info_pris_real
from utils.chamfer import eval_CD, eval_CD_start
from utils.rotation import R_from_axis_angle


@systems.register('d2nerf_pris_real-system')
class D2NeRFPrisRealSystem(BaseSystem):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def prepare(self):
        self.criterions = {
            'psnr': PSNR(),
            'ssim': SSIM()
        }
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        self.train_num_rays = self.config.model.train_num_rays
        self.gt_axis = load_gt_axis_real(self.config.model.motion_gt_path)
        self.init_axis = self.model.get_init_axis()
        # metric purpose
        self.gt_info = load_gt_info_pris_real(self.config.model.motion_gt_path)

    
    def on_train_start(self) -> None:
        self.dataset = self.trainer.datamodule.train_dataloader().dataset

        return super().on_train_start()
    
    def on_validation_start(self) -> None:
        self.dataset = self.trainer.datamodule.val_dataloader().dataset
        return super().on_validation_start()
    
    def on_test_start(self) -> None:
        self.dataset = self.trainer.datamodule.test_dataloader().dataset
        return super().on_test_start()
    
    def on_predict_start(self) -> None:
        self.dataset = self.trainer.datamodule.predict_dataloader().dataset
        return super().on_predict_start()

    def forward(self, batch):
        return self.model(batch['rays_0'], batch['rays_1'])
    
    def preprocess_data(self, batch, stage):
        '''construct batch for each iteration'''
        # print('--------',self.dataset.split, '--------')

        if 'index' in batch: # validation / testing
            index = batch['index']
        else:
            if self.config.model.batch_image_sampling:
                index_0 = torch.randint(0, len(self.dataset.all_images_0), size=(self.train_num_rays,), device=self.dataset.all_images_0.device)
                index_1 = torch.randint(0, len(self.dataset.all_images_1), size=(self.train_num_rays,), device=self.dataset.all_images_1.device)
            else:
                index_0 = torch.randint(0, len(self.dataset.all_images_0), size=(1,), device=self.dataset.all_images_0.device)
                index_1 = torch.randint(0, len(self.dataset.all_images_1), size=(1,), device=self.dataset.all_images_1.device)

        if stage in ['train']:
            x = torch.randint(
                0, self.dataset.w, size=(self.train_num_rays,), device=self.dataset.all_images_0.device
            )
            y = torch.randint(
                0, self.dataset.h, size=(self.train_num_rays,), device=self.dataset.all_images_0.device
            )

            c2w_0 = self.dataset.all_c2w_0[index_0]
            c2w_1 = self.dataset.all_c2w_1[index_1]

            directions_0 = self.dataset.directions_0[y, x]
            directions_1 = self.dataset.directions_1[y, x]
            
            rays_o_0, rays_d_0 = get_rays(directions_0, c2w_0)
            rays_o_1, rays_d_1 = get_rays(directions_1, c2w_1)

            rgb_0 = self.dataset.all_images_0[index_0, y, x].view(-1, self.dataset.all_images_0.shape[-1])
            rgb_1 = self.dataset.all_images_1[index_1, y, x].view(-1, self.dataset.all_images_1.shape[-1])

            fg_mask_0 = self.dataset.all_fg_masks_0[index_0, y, x].view(-1)
            fg_mask_1 = self.dataset.all_fg_masks_1[index_1, y, x].view(-1)

            rays_0 = torch.cat([rays_o_0, rays_d_0], dim=-1)
            rays_1 = torch.cat([rays_o_1, rays_d_1], dim=-1)
    
            batch.update({
                'rays_0': rays_0,
                'rays_1': rays_1,
                'rgb_0': rgb_0,
                'rgb_1': rgb_1,
                'fg_mask_0': fg_mask_0,
                'fg_mask_1': fg_mask_1,
            })
        
        elif stage in ['predict']:
            c2w = self.dataset.all_c2w[index][0]
            rgb_0 = self.dataset.start_img[index][0]
            rgb_1 = self.dataset.end_img[index][0]

            directions = self.dataset.directions
            rays_o, rays_d = get_rays(directions, c2w)
            rays = torch.cat([rays_o, rays_d], dim=-1)
            batch.update({
                'rays': rays,
                'rgb_0': rgb_0,
                'rgb_1': rgb_1,
            })

        else: # validation / testing / predicting
            c2w_0 = self.dataset.all_c2w_0[index][0]
            c2w_1 = self.dataset.all_c2w_1[index][0]

            c2w_0_h = torch.eye(4)
            c2w_0_h[:3, :4] = c2w_0
            c2w_1_h = torch.eye(4)
            c2w_1_h[:3, :4] = c2w_1
            w2c_0 = torch.linalg.inv(c2w_0_h)
            w2c_1 = torch.linalg.inv(c2w_1_h)
            batch.update({
                'w2c_0': w2c_0,
                'w2c_1': w2c_1,
                'K_0': self.dataset.K_0,
                'K_1': self.dataset.K_1
            })

            directions_0 = self.dataset.directions_0
            directions_1 = self.dataset.directions_1

            rays_o_0, rays_d_0 = get_rays(directions_0, c2w_0)
            rays_o_1, rays_d_1 = get_rays(directions_1, c2w_1)

            rgb_0 = self.dataset.all_images_0[index].view(-1, self.dataset.all_images_0.shape[-1])
            rgb_1 = self.dataset.all_images_1[index].view(-1, self.dataset.all_images_1.shape[-1])

            fg_mask_0 = self.dataset.all_fg_masks_0[index].view(-1)
            fg_mask_1 = self.dataset.all_fg_masks_1[index].view(-1)

            rays_0 = torch.cat([rays_o_0, rays_d_0], dim=-1)
            rays_1 = torch.cat([rays_o_1, rays_d_1], dim=-1)
    
            batch.update({
                'rays_0': rays_0,
                'rays_1': rays_1,
                'rgb_0': rgb_0,
                'rgb_1': rgb_1,
                'fg_mask_0': fg_mask_0,
                'fg_mask_1': fg_mask_1,
            })     
    
    def configure_optimizers(self):
        if self.config.model.motion_gt:
            model_optim = parse_optimizer(self.config.system.model_optimizer, self.model)
            model_scheduler = parse_scheduler(self.config.system.model_scheduler, model_optim)
            return [model_optim], [model_scheduler]
        else:
            model_optim = parse_optimizer(self.config.system.model_optimizer, self.model)
            motion_optim = parse_optimizer(self.config.system.motion_optimizer, self.model)
            model_scheduler = parse_scheduler(self.config.system.model_scheduler, model_optim)
            motion_scheduler = parse_scheduler(self.config.system.motion_scheduler, motion_optim)
            return [model_optim, motion_optim], [model_scheduler, motion_scheduler]

    def weighted_ratio_entropy_loss(self, ratio, weight, clip_threshold=1e-5, skewness=1.0):
        ratio = torch.clip(ratio, clip_threshold, 1-clip_threshold)
        entropy = - (ratio * torch.log(ratio) + (1.-ratio) * torch.log(1.-ratio))
        weighted_entropy = weight * entropy
        return weighted_entropy.mean()    
    
    def weighted_ratio_f1(self, ratio, weight):
        return (weight * F.smooth_l1_loss(ratio, torch.ones_like(ratio))).mean()
    
    def weighted_s_f1(self, s, weight):
        return (weight * F.smooth_l1_loss(s, torch.zeros_like(s))).mean()
    
    def compute_blend_ratio_loss(self, ratio, clip_threshold=1e-6, skew=1.0, use_lap=False):
        """
        Compute the blendw loss based on entropy or lap
        "skew" is used to control the skew of entropy loss.
        A value larger than 1.0 means the entropy loss is more skewed to 1, which in our case
        means more skewed to the static part
        """

        ratio = torch.clip(ratio ** skew, clip_threshold, 1-clip_threshold)
        entropy = - (ratio * torch.log(ratio) + (1.-ratio) * torch.log(1.-ratio))
        # torch.nan_to_num(torch.log(1.-ratio), neginf=-1e20))
        # isnan = torch.isnan(entropy).sum(-1)
        # self.log('entropy_nan', isnan, prog_bar=True)
        if use_lap:
            lap = torch.exp(-ratio) + torch.exp(ratio-1.)
            lap = -torch.log(torch.clip(lap, clip_threshold))
            return lap.mean()
        
        return entropy.mean()

    def aabb_center_crop(self):
        aabb = self.model.extract_aabb(res=100, thre=5.0)
        # self.model.axis_o = torch.clamp(self.model.axis_o, min=aabb['dynamic']['min'], max=aabb['dynamic']['max'])
        # print(self.model.axis_o)
        axis_o = self.model.axis_o.data
        # print('min:', aabb['static']['min'])
        # print('max:', aabb['static']['max'])
        # axis_o.clamp_(min=aabb['dynamic']['min'].to(self.model.axis_o), max=aabb['dynamic']['max'].to(self.model.axis_o))
        axis_o.clamp_(min=aabb['static']['min'].to(self.model.axis_o), max=aabb['static']['max'].to(self.model.axis_o))

        # print(self.model.axis_o)
    
    def aabb_center_loss(self):
        aabb = self.model.extract_aabb(res=100, thre=10.0)
        axis_o = self.model.axis_o.data.clone()
        axis_o.clamp_(min=aabb['dynamic']['min'].to(self.model.axis_o), max=aabb['dynamic']['max'].to(self.model.axis_o))
        axis_o.clamp_(min=aabb['static']['min'].to(self.model.axis_o), max=aabb['static']['max'].to(self.model.axis_o))
        loss = F.mse_loss(self.model.axis_o, axis_o)
        return loss
    
    def aabb_overlap_loss(self):
        aabb = self.model.extract_aabb(res=100, thre=10.0)
        min_bound = torch.maximum(aabb['static']['min'], aabb['dynamic']['min'])
        max_bound = torch.minimum(aabb['static']['max'], aabb['dynamic']['max'])
        center_overlapping = 0.5 * (min_bound + max_bound)
        loss = F.mse_loss(self.model.axis_o, center_overlapping)
        return loss

    def aabb_overlap_clip(self):
        aabb = self.model.extract_aabb(res=100, thre=10.0)
        min_bound = torch.maximum(aabb['static']['min'], aabb['dynamic']['min'])
        max_bound = torch.minimum(aabb['static']['max'], aabb['dynamic']['max'])
        for i in range(3):
            if min_bound[i] > max_bound[i]:
                t = min_bound[i]
                min_bound[i] = max_bound[i]
                max_bound[i] = t
        axis_o = self.model.axis_o.data
        axis_o.clamp_(min=min_bound, max=max_bound)
        

    def training_step(self, batch, batch_idx, optimizer_idx):

        outs = self.model(batch['rays_0'], batch['rays_1'])
        loss = 0.

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / \
                            (0.5*(outs[0]['num_samples'].sum().item() + outs[1]['num_samples'].sum().item())+1e-15)))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)

        # photometric loss
        loss_rgb = (F.smooth_l1_loss(outs[0]['comp_rgb'][outs[0]['rays_valid']], batch['rgb_0'][outs[0]['rays_valid']]) + \
                    F.smooth_l1_loss(outs[1]['comp_rgb'][outs[1]['rays_valid']], batch['rgb_1'][outs[1]['rays_valid']])) * 0.5
        self.log('train/loss_rgb', loss_rgb)
        loss += loss_rgb * self.config.system.loss.lambda_rgb
        # binary entropy loss for the blend ratio
        if (optimizer_idx == 0) and ((self.global_step /2) > self.config.system.loss.reg_from_iter):
            if self.config.system.loss.lambda_blend_ratio > 0.:
                loss_ratio = (self.compute_blend_ratio_loss(outs[0]['ratio']) + self.compute_blend_ratio_loss(outs[1]['ratio'])) * 0.5
                self.log('train/loss_ratio', loss_ratio)
                # lamba_blend_ratio = 0.0001 + self.global_step * (0.01 - 0.0001) / self.config.trainer.max_steps
                loss += loss_ratio * self.config.system.loss.lambda_blend_ratio
            if self.config.system.loss.lambda_part_mask > 0.:
                part_skew = self.config.system.loss.skew_part_mask
                loss_part_ratio = (self.compute_blend_ratio_loss(outs[0]['part_mask_ratio'], skew=part_skew) + \
                              self.compute_blend_ratio_loss(outs[1]['part_mask_ratio'], skew=part_skew)) * 0.5
                self.log('train/loss_part_ratio', loss_part_ratio)
                loss += loss_part_ratio * self.config.system.loss.lambda_part_mask
        # regularize weights
        # if self.config.system.loss.lambda_weights > 0.:
        #     loss_weights = (outs[0]['weights_mean'] + outs[1]['weights_mean']) * 0.5
        #     self.log('train/loss_weights', loss_weights)
        #     loss += loss_weights * self.config.system.loss.lambda_weights
        
        # if self.config.system.loss.lambda_distortion > 0.:
        #     loss_distortion = (outs[0]['distortion_loss'] + outs[1]['distortion_loss']) * 0.5
        #     self.log('train/loss_distortion', loss_distortion)
        #     loss += loss_distortion * self.config.system.loss.lambda_distortion
        
        # lambda_min_alpha = self.config.system.loss.get('lambda_min_alpha', 0.)
        # if lambda_min_alpha > 0.:
        #     loss_min_alpha = (outs[0]['min_alpha_mean'] + outs[1]['min_alpha_mean']) * 0.5
        #     self.log('train/loss_min_alpha', loss_min_alpha)
        #     loss += loss_min_alpha * lambda_min_alpha
        
        lambda_mask = self.config.system.loss.get('lambda_mask', 0.)
        if lambda_mask > 0.:
            clip_eps = 1e-5
            opacity_0 = torch.clamp(outs[0]['opacity'], clip_eps, 1.-clip_eps)
            loss_mask_0 = binary_cross_entropy(opacity_0, batch['fg_mask_0'].float())
            opacity_1 = torch.clamp(outs[1]['opacity'], clip_eps, 1.-clip_eps)
            loss_mask_1 = binary_cross_entropy(opacity_1, batch['fg_mask_1'].float())
            loss_mask = (loss_mask_0 + loss_mask_1) * 0.5
            self.log('train/loss_mask', loss_mask)
            loss += loss_mask * self.C(self.config.system.loss.lambda_mask)

        # impose prior on the ray
        # lambda_prior = self.config.system.loss.get('lambda_prior', 0.)
        # if lambda_prior > 0.:
        #     loss_prior_s = 0.5*(outs[0]['prior_s'] + outs[1]['prior_s'])
        #     loss += lambda_prior * loss_prior_s
        #     self.log('train/loss_prior_s', loss_prior_s)

        # lambda_swipe_ratio = self.config.system.loss.get('lambda_swipe_ratio', 0.)
        # # if lambda_swipe_ratio > 0. and (self.global_step/2) > 100:
        # if lambda_swipe_ratio > 0. and ((self.global_step / 2) > self.config.system.second_stage):
        #     # loss_ratio_swi = (self.weighted_ratio_entropy_loss(outs[0]['ratio_swi'], outs[0]['weight_swi']) + \
        #     #                 self.weighted_ratio_entropy_loss(outs[1]['ratio_swi'], outs[1]['weight_swi'])) * 0.5
        #     # loss_ratio_swi = (self.weighted_ratio_f1(outs[0]['ratio_swi'], outs[0]['weight_swi']) + \
        #     #                  self.weighted_ratio_f1(outs[1]['ratio_swi'], outs[1]['weight_swi'])) * 0.5   
        #     loss_ratio_swi = (self.weighted_s_f1(outs[0]['ratio_swi'], outs[0]['weight_swi']) + \
        #                      self.weighted_s_f1(outs[1]['ratio_swi'], outs[1]['weight_swi'])) * 0.5               
        #     # print(loss_ratio_swi)
        #     self.log('train/loss_ratio_swi', loss_ratio_swi)
        #     loss += loss_ratio_swi * lambda_swipe_ratio   
  

        
        # lambda_inter_ratio = self.config.system.loss.get('lambda_inter_ratio', 0.)
        # if lambda_inter_ratio > 0. and (self.global_step/2) > 50:
        #     loss_both = (outs[0]['entropy_both'] + outs[1]['entropy_both']) * 0.5
        #     loss_xor = (outs[0]['entropy_xor'] + outs[1]['entropy_xor']) * 0.5
        #     self.log('train/loss_both', loss_both)
        #     self.log('train/loss_xor', loss_xor)
        #     loss += lambda_inter_ratio * (loss_both + loss_xor)  
        
        # if ((self.global_step / 2) > 100) and (((self.global_step / 2) // 100) == 0): # hard clip loss
        #     aabb_center_loss = self.aabb_center_loss()
        #     print(aabb_center_loss)
        #     self.log('train/aabb_center_loss', aabb_center_loss)
        #     loss += aabb_center_loss

        # if (self.global_step / 2) > 100:
        # aabb_overlap_loss = self.aabb_overlap_loss()
        # print(aabb_overlap_loss)
        # self.log('train/aabb_center_loss', aabb_overlap_loss)
        # loss += aabb_overlap_loss
        # self.aabb_overlap_clip()
        # optimize the motion params


        # if ((self.global_step / 2) > 100) and (((self.global_step / 2) // 100) == 0):
        # if ((self.global_step / 2) > 200):
        # self.aabb_center_crop()
            # print('crop')

    

        # might need edit!!!!!!!!!!!
        # losses_model_reg = self.model.regularizations(outs)
        # for name, value in losses_model_reg.items():
        #     self.log(f'train/loss_{name}', value)
        #     loss_ = value * self.C(self.config.system.loss[f"lambda_{name}"])
        #     loss += loss_

        # for name, value in self.config.system.loss.items():
        #     if name.startswith('lambda'):
        #         self.log(f'train_params/{name}', self.C(value))
        
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        # if (self.global_step / 2) == self.config.system.second_stage:
        #     # reset dynamic field and motion params
        #     self.model.reset_dynamic()
        #     # freeze static
        #     self.model.static_geometry.requires_grad_(False)
        #     self.model.static_texture.requires_grad_(False)

        return loss    

    """
    # aggregate outputs from different devices (DP)
    def training_step_end(self, out):
        pass
    """
    
    """
    # aggregate outputs from different iterations
    def training_epoch_end(self, out):
        pass
    """
    
    def validation_step(self, batch, batch_idx):
        outs = self.model(batch['rays_0'], batch['rays_1'])
        psnr_0 = self.criterions['psnr'](outs[0]['comp_rgb'], batch['rgb_0'])
        psnr_1 = self.criterions['psnr'](outs[1]['comp_rgb'], batch['rgb_1'])
        psnr = 0.5 * (psnr_0 + psnr_1)
        W, H = self.dataset.w, self.dataset.h
        img_0 = outs[0]['comp_rgb'].view(H, W, 3)
        img_1 = outs[1]['comp_rgb'].view(H, W, 3)
        depth_0 = outs[0]['depth'].view(H, W)
        depth_1 = outs[1]['depth'].view(H, W)
        depth_s_0 = outs[0]['depth_s'].view(H, W)
        depth_s_1 = outs[1]['depth_s'].view(H, W)
        depth_d_0 = outs[0]['depth_d'].view(H, W)
        depth_d_1 = outs[1]['depth_d'].view(H, W)
        opacity_0 = outs[0]['opacity'].view(H, W)
        opacity_1 = outs[1]['opacity'].view(H, W)
        rgb_s_0 = outs[0]['rgb_s'].view(H, W, 3)
        rgb_d_0 = outs[0]['rgb_d'].view(H, W, 3)
        rgb_s_1 = outs[1]['rgb_s'].view(H, W, 3)
        rgb_d_1 = outs[1]['rgb_d'].view(H, W, 3)

        # transformation output
        transform_json = {
            'axis_o': self.model.axis_o.detach(),
            'axis_d': self.model.axis_d.detach(),
            'dist': self.model.dist.detach()
        }

        # axis_info to draw on the image
        axis_o = transform_json['axis_o'].unsqueeze(0)
        axis_d = F.normalize(transform_json['axis_d'], dim=0).unsqueeze(0)
        pred_axis = torch.cat([axis_o, axis_o+axis_d], dim=0)
        gt_axis_0 = proj2img(self.gt_axis, batch['w2c_0'], batch['K_0'])
        gt_axis_1 = proj2img(self.gt_axis, batch['w2c_1'], batch['K_1'])
        pred_axis_0 = proj2img(pred_axis, batch['w2c_0'], batch['K_0'])
        pred_axis_1 = proj2img(pred_axis, batch['w2c_1'], batch['K_1'])
        init_axis_0 = proj2img(self.init_axis, batch['w2c_0'], batch['K_0'])
        init_axis_1 = proj2img(self.init_axis, batch['w2c_1'], batch['K_1'])
        axis_info_0 = {
            'GT': gt_axis_0,
            'pred': pred_axis_0,
            'init': init_axis_0
        }
        axis_info_1 = {
            'GT': gt_axis_1,
            'pred': pred_axis_1,
            'init': init_axis_1
        }

        self.save_image_grid(f"it{int(self.global_step/2)}-{batch['index'][0].item()}.png", [[
            {'type': 'rgb', 'img': batch['rgb_0'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'draw_axis': True, 'axis_info': axis_info_0}},
            {'type': 'rgb', 'img': img_0, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': rgb_s_0, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': rgb_d_0, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': depth_0, 'kwargs': {}},
            {'type': 'grayscale', 'img': depth_s_0, 'kwargs': {}},
            {'type': 'grayscale', 'img': depth_d_0, 'kwargs': {}},
            {'type': 'grayscale', 'img': opacity_0, 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
        ],
    [
            {'type': 'rgb', 'img': batch['rgb_1'].view(H, W, 3), 'kwargs': {'data_format': 'HWC', 'draw_axis': True, 'axis_info': axis_info_1}},
            {'type': 'rgb', 'img': img_1, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': rgb_s_1, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'rgb', 'img': rgb_d_1, 'kwargs': {'data_format': 'HWC'}},
            {'type': 'grayscale', 'img': depth_1, 'kwargs': {}},  # 'data_range': (0, 20)
            {'type': 'grayscale', 'img': depth_s_1, 'kwargs': {}},
            {'type': 'grayscale', 'img': depth_d_1, 'kwargs': {}},
            {'type': 'grayscale', 'img': opacity_1, 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
        ]])

       
        del outs
        return {
            'psnr': psnr,
            'index': batch['index'],
            'transform': transform_json,
            'pred_axis': pred_axis
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            out_set = {}
            for step_out in out:
                # DP
                if step_out['index'].ndim == 1:
                    out_set[step_out['index'].item()] = {'psnr': step_out['psnr']}
                # DDP
                else:
                    for oi, index in enumerate(step_out['index']):
                        out_set[index[0].item()] = {'psnr': step_out['psnr'][oi]}
            psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
            self.log('val/psnr', psnr, prog_bar=True, rank_zero_only=True)  
            # translation difference
            dist_diff_l2 = torch.sqrt((((out[0]['transform']['dist'] * F.normalize(out[0]['transform']['axis_d'].squeeze(0), p=2, dim=0)).cpu() - self.gt_info['dist'] * self.gt_info['axis_d']) ** 2).sum())
            self.log('val/dist_diff_l2', dist_diff_l2, prog_bar=True, rank_zero_only=True)
            # axis error
            angular_err, _ = axis_error(out[0]['pred_axis'].squeeze(0), self.gt_axis)
            self.log('val/angular_err', angular_err, prog_bar=True, rank_zero_only=True)

            transform_tensor = out[0]['transform']
            transform_json = {}
            for key, values in transform_tensor.items():
                transform_json.update({key: values[0].cpu().tolist()})
            self.save_json(f'it{int(self.global_step/2)}_transform.json', transform_json)   

        del out


    def test_step(self, batch, batch_idx):  
        if self.config.mesh_only:
            transform_json = {
                'axis_o': self.model.axis_o.detach(),
                'axis_d': self.model.axis_d.detach(),
                'dist': self.model.dist.detach()
            }
            return {
                'index': batch['index'],
                'transform': transform_json,
            }      
        else:
            W, H = self.dataset.w, self.dataset.h

            outs = self(batch)
            psnr_0 = self.criterions['psnr'](outs[0]['comp_rgb'], batch['rgb_0'])
            psnr_1 = self.criterions['psnr'](outs[1]['comp_rgb'], batch['rgb_1'])
            ssim_0 = self.criterions['ssim'](outs[0]['comp_rgb'].view(1, H, W, 3).cpu(), batch['rgb_0'].view(1, H, W, 3).cpu())
            ssim_1 = self.criterions['ssim'](outs[1]['comp_rgb'].view(1, H, W, 3).cpu(), batch['rgb_1'].view(1, H, W, 3).cpu())

            psnr = 0.5 * (psnr_0 + psnr_1)
            ssim = 0.5 * (ssim_0 + ssim_1)

            img_0 = outs[0]['comp_rgb'].view(H, W, 3)
            img_1 = outs[1]['comp_rgb'].view(H, W, 3)
            depth_0 = outs[0]['depth'].view(H, W)
            depth_1 = outs[1]['depth'].view(H, W)
            depth_s_0 = outs[0]['depth_s'].view(H, W)
            depth_s_1 = outs[1]['depth_s'].view(H, W)
            depth_d_0 = outs[0]['depth_d'].view(H, W)
            depth_d_1 = outs[1]['depth_d'].view(H, W)
            # opacity_0 = outs[0]['opacity'].view(H, W)
            # opacity_1 = outs[1]['opacity'].view(H, W)
            rgb_s_0 = outs[0]['rgb_s'].view(H, W, 3)
            rgb_d_0 = outs[0]['rgb_d'].view(H, W, 3)
            rgb_s_1 = outs[1]['rgb_s'].view(H, W, 3)
            rgb_d_1 = outs[1]['rgb_d'].view(H, W, 3)

            # transformation output
            transform_json = {
                'axis_o': self.model.axis_o.detach(),
                'axis_d': self.model.axis_d.detach(),
                'dist': self.model.dist.detach()
            }

            # axis_info to draw on the image
            axis_o = transform_json['axis_o'].unsqueeze(0)
            axis_d = F.normalize(transform_json['axis_d'], dim=0).unsqueeze(0)
            pred_axis = torch.cat([axis_o, axis_o+axis_d], dim=0)
            gt_axis_0 = proj2img(self.gt_axis, batch['w2c_0'], batch['K_0'])
            gt_axis_1 = proj2img(self.gt_axis, batch['w2c_1'], batch['K_1'])
            pred_axis_0 = proj2img(pred_axis, batch['w2c_0'], batch['K_0'])
            pred_axis_1 = proj2img(pred_axis, batch['w2c_1'], batch['K_1'])
            init_axis_0 = proj2img(self.init_axis, batch['w2c_0'], batch['K_0'])
            init_axis_1 = proj2img(self.init_axis, batch['w2c_1'], batch['K_1'])
            axis_info_0 = {
                'GT': gt_axis_0,
                'pred': pred_axis_0,
                # 'init': init_axis_0
            }
            axis_info_1 = {
                'GT': gt_axis_1,
                'pred': pred_axis_1,
                # 'init': init_axis_1
            }
            self.save_image_grid(f"it{int(self.global_step/2)}-test/{batch['index'][0].item()}.png", [[
                {'type': 'rgb', 'img': batch['rgb_0'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': img_0, 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': rgb_s_0, 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': rgb_d_0, 'kwargs': {'data_format': 'HWC'}},
                {'type': 'grayscale', 'img': depth_0, 'kwargs': {}},
                {'type': 'grayscale', 'img': depth_s_0, 'kwargs': {}},
                {'type': 'grayscale', 'img': depth_d_0, 'kwargs': {}},
                # {'type': 'grayscale', 'img': opacity_0, 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
            ],
            [
                {'type': 'rgb', 'img': batch['rgb_1'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': img_1, 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': rgb_s_1, 'kwargs': {'data_format': 'HWC'}},
                {'type': 'rgb', 'img': rgb_d_1, 'kwargs': {'data_format': 'HWC'}},
                {'type': 'grayscale', 'img': depth_1, 'kwargs': {}},
                {'type': 'grayscale', 'img': depth_s_1, 'kwargs': {}},
                {'type': 'grayscale', 'img': depth_d_1, 'kwargs': {}},
                # {'type': 'grayscale', 'img': opacity_1, 'kwargs': {'cmap': None, 'data_range': (0, 1)}}
            ]])

            del outs
            return {
                'psnr': psnr,
                'ssim': ssim,
                'index': batch['index'],
                'transform': transform_json,
                'pred_axis': pred_axis
            }      
    
    def test_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            if self.config.mesh_only:
                mesh_dict = self.model.isosurface()
                resolution = self.config.model.geometry.isosurface.resolution
                threshold = float(self.config.model.geometry.isosurface.threshold)
                for name, mesh in mesh_dict.items():
                    self.save_mesh(
                        f"it{int(self.global_step/2)}_{name}_{resolution}_thre{threshold}.obj",
                        mesh['v_pos'],
                        mesh['t_pos_idx'],
                    )
                transform_tensor = out[0]['transform']
                self.save_trans_part_mesh_translate(f"it{int(self.global_step/2)}_dynamic_{resolution}_thre{threshold}.obj",
                                          f"it{int(self.global_step/2)}_trans2start_{resolution}_thre{threshold}.obj",
                                          transform_tensor)
                self.save_trans_part_mesh_translate(f"it{int(self.global_step/2)}_dynamic_{resolution}_thre{threshold}.obj",
                                          f"it{int(self.global_step/2)}_trans2end_{resolution}_thre{threshold}.obj",
                                          transform_tensor, to_start=False)
                # transform_tensor = out[0]['transform']
                # transform_json = {}
                # for key, values in transform_tensor.items():
                #     transform_json.update({key: values[0].cpu().tolist()})
                # axis_json = transform_json
                # axis_json['axis_o'] = self.gt_axis[0]
                # self.save_axis(f'it{int(self.global_step/2)}_axis.ply', axis_json)
                axis_o = transform_tensor['axis_o'].squeeze(0)
                axis_d = F.normalize(transform_tensor['axis_d'].squeeze(0), dim=0)
                print(axis_d)
                pred_axis = torch.cat([axis_o.unsqueeze(0), axis_o.unsqueeze(0)+axis_d.unsqueeze(0)], dim=0)
                print(pred_axis.shape)
                # rotation difference
                dist_diff_l2 = torch.sqrt((((out[0]['transform']['dist'] * F.normalize(out[0]['transform']['axis_d'].squeeze(0), p=2, dim=0)).cpu() - self.gt_info['dist'] * self.gt_info['axis_d']) ** 2).sum())
                
                self.log('test/dist_diff_l2', dist_diff_l2, prog_bar=True, rank_zero_only=True)

                # axis error
                angular_err, _ = axis_error(pred_axis.squeeze(1), self.gt_axis)
                self.log('test/angular_err', angular_err, prog_bar=True, rank_zero_only=True)
                # Chamfer-L1 Distance at start state
                cd_part_s, cd_part_ds, cd_ws = eval_CD_start(
                    self.get_save_path(f"it{int(self.global_step/2)}_static_{resolution}_thre{threshold}.obj"),
                    self.get_save_path(f"it{int(self.global_step/2)}_trans2start_{resolution}_thre{threshold}.obj"),
                    self.get_save_path(f"it{int(self.global_step/2)}_whole_trans2start_{resolution}_thre{threshold}.ply"),
                    join(dirname(self.config.model.motion_gt_path), 'start', 'start_static_rotate.ply'),
                    join(dirname(self.config.model.motion_gt_path), 'start', 'start_dynamic_rotate.ply'),
                    join(dirname(self.config.model.motion_gt_path), 'start', 'start_rotate.ply')
                )
                with open(self.get_save_path(f'it{int(self.global_step/2)}_metrics_thre{threshold}.txt'), 'w') as f:
                    f.write(f'---------- geometry start ----------\n')
                    f.write(f'CD_static_start: {cd_part_s}\n')
                    f.write(f'CD_dynamic_start: {cd_part_ds}\n')
                    f.write(f'CD_whole_start: {cd_ws}\n')
                    f.write(f'---------- motion ----------\n')
                    f.write(f'axis angular: {angular_err}\n')
                    f.write(f'translation err: {dist_diff_l2}\n')
            else:
                out_set = {}
                for step_out in out:
                    # DP
                    if step_out['index'].ndim == 1:
                        out_set[step_out['index'].item()] = {'psnr': step_out['psnr'], 'ssim': step_out['ssim']}

                    # DDP
                    else:
                        for oi, index in enumerate(step_out['index']):
                            out_set[index[0].item()] = {'psnr': step_out['psnr'][oi], 'ssim': step_out['ssim'][oi]}

                # nvs
                psnr = torch.mean(torch.stack([o['psnr'] for o in out_set.values()]))
                self.log('test/psnr', psnr, prog_bar=True, rank_zero_only=True, sync_dist=True)
                ssim = torch.mean(torch.stack([o['ssim'] for o in out_set.values()]))
                self.log('test/ssim', ssim, prog_bar=True, rank_zero_only=True, sync_dist=True)     

                # rotation difference
                dist_diff_l2 = torch.sqrt((((out[0]['transform']['dist'] * F.normalize(out[0]['transform']['axis_d'].squeeze(0), p=2, dim=0)).cpu() - self.gt_info['dist'] * self.gt_info['axis_d']) ** 2).sum())
                
                self.log('test/dist_diff_l2', dist_diff_l2, prog_bar=True, rank_zero_only=True)

                # axis error
                angular_err, _ = axis_error(out[0]['pred_axis'].squeeze(0), self.gt_axis)
                self.log('test/angular_err', angular_err, prog_bar=True, rank_zero_only=True)

                # video for the validation images
                # val_imgs = int(self.trainer.limit_val_batches)
                # for val_i in range(val_imgs):
                #     self.save_img_sequence(
                #         f"val_{val_i}",
                #         "",
                #         f'it(\d+)\-{val_i}.png',
                #         save_format='mp4',
                #         fps=10
                #     )

                # mesh for the axis
                transform_tensor = out[0]['transform']
                transform_json = {}
                for key, values in transform_tensor.items():
                    transform_json.update({key: values[0].cpu().tolist()})
                self.save_json(f'it{int(self.global_step/2)}-test/transform.json', transform_json)
                axis_json = transform_json
                axis_json['axis_o'] = self.gt_axis[0]
                self.save_axis(f'it{int(self.global_step/2)}_axis.ply', axis_json)
                

                # video for the testing images
                self.save_img_sequence(
                    f"it{int(self.global_step/2)}-test",
                    f"it{int(self.global_step/2)}-test",
                    '(\d+)\.png',
                    save_format='mp4',
                    fps=10
                )
                
                # extract mesh
                mesh_dict = self.model.isosurface()
                resolution = self.config.model.geometry.isosurface.resolution
                threshold = float(self.config.model.geometry.isosurface.threshold)
                for name, mesh in mesh_dict.items():
                    self.save_mesh(
                        f"it{int(self.global_step/2)}_{name}_{resolution}_thre{threshold}.obj",
                        mesh['v_pos'],
                        mesh['t_pos_idx'],
                    )
                self.save_trans_part_mesh_translate(f"it{int(self.global_step/2)}_dynamic_{resolution}_thre{threshold}.obj",
                                          f"it{int(self.global_step/2)}_trans2start_{resolution}_thre{threshold}.obj",
                                          transform_tensor)
                self.save_trans_part_mesh_translate(f"it{int(self.global_step/2)}_dynamic_{resolution}_thre{threshold}.obj",
                                          f"it{int(self.global_step/2)}_trans2end_{resolution}_thre{threshold}.obj",
                                          transform_tensor, to_start=False)

                # Chamfer-L1 Distance
                # cd_part_s, cd_part_d, cd_w = eval_CD(
                #     self.get_save_path(f"it{int(self.global_step/2)}_static_{resolution}_thre{threshold}.obj"),
                #     self.get_save_path(f"it{int(self.global_step/2)}_dynamic_{resolution}_thre{threshold}.obj"),
                #     self.get_save_path(f"it{int(self.global_step/2)}_whole_{resolution}_thre{threshold}.ply"),
                #     join(dirname(self.config.model.motion_gt_path), 'canonical', 'canonical_static_rotate.ply'),
                #     join(dirname(self.config.model.motion_gt_path), 'canonical', 'canonical_dynamic_rotate.ply'),
                #     join(dirname(self.config.model.motion_gt_path), 'canonical', 'canonical_rotate.ply')
                # )

                cd_part_s, cd_part_ds, cd_ws = eval_CD_start(
                    self.get_save_path(f"it{int(self.global_step/2)}_static_{resolution}_thre{threshold}.obj"),
                    self.get_save_path(f"it{int(self.global_step/2)}_trans2start_{resolution}_thre{threshold}.obj"),
                    self.get_save_path(f"it{int(self.global_step/2)}_whole_trans2start_{resolution}_thre{threshold}.ply"),
                    join(dirname(self.config.model.motion_gt_path), 'start', 'start_static_rotate.ply'),
                    join(dirname(self.config.model.motion_gt_path), 'start', 'start_dynamic_rotate.ply'),
                    join(dirname(self.config.model.motion_gt_path), 'start', 'start_rotate.ply')
                )
                
                # save the metrics
                with open(self.get_save_path(f'it{int(self.global_step/2)}_metrics.txt'), 'w') as f:
                    f.write(f'---------- geometry start----------\n')
                    f.write(f'CD_static: {cd_part_s}\n')
                    f.write(f'CD_dynamic: {cd_part_ds}\n')
                    f.write(f'CD_whole: {cd_ws}\n')
                    f.write(f'---------- texture ----------\n')
                    f.write(f'PSNR: {psnr}\n')
                    f.write(f'SSIM: {ssim}\n')
                    f.write(f'---------- motion ----------\n')
                    f.write(f'axis angular: {angular_err}\n')
                    f.write(f'euclidean distance: {dist_diff_l2}\n')
                
                # self.render_geometry(
                #     self.config.model.motion_gt_path,
                #     int(self.global_step/2), resolution, threshold,
                #     transform_tensor['axis_o'],
                #     R_from_axis_angle(transform_tensor['axis_d'].squeeze(0).cpu(), -transform_tensor['rot_angle']),
                #     self.config.dataset
                # )

        del out   

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        from models.utils import chunk_batch
        n_interpolate = 4 # render n_interpolate+1 imgs in total
        max_state= 1
        interval = max_state / n_interpolate
        self.model.eval()
        imgs, depths = [], []
        i = 0
        W, H = self.dataset.w, self.dataset.h
        while i < max_state or i == max_state:
            out = chunk_batch(self.model.forward_, self.config.model.ray_chunk, batch['rays'], scene_state=i)
            img = out['comp_rgb'].view(H, W, 3)
            # dep = out['depth'].view(H, W)
            imgs.append(img)
            # depths.append(dep)
            print('finish state ', i)
            i += interval
            del out
        img_grid = [{'type': 'rgb', 'img': batch['rgb_0'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}]
        img_grid += [{'type': 'rgb', 'img': img, 'kwargs': {'data_format': 'HWC'}} for img in imgs]
        img_grid.append({'type': 'rgb', 'img': batch['rgb_1'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}})
        source = self.config.source
        tokens = source.split('/')
        self.save_image_grid(f"it{self.global_step}-pred/{tokens[1]}_{self.config.dataset.view_idx}_RGB.png", img_grid)
        # self.save_image_grid(f"it{self.global_step}-pred/{self.config.dataset.view_idx}_depth.png", \
        #     [{'type': 'grayscale', 'img': dep, 'kwargs': {}} for dep in depths])