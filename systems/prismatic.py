import torch
import torch.nn.functional as F
from os.path import join, dirname
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, SSIM, binary_cross_entropy, entropy_loss 
from systems.utils import parse_optimizer, parse_scheduler, load_gt_axis, load_gt_info_pris

from visual_utils.plot_camera import plot_camera

@systems.register('prismatic-system')
class PrismaticSystem(BaseSystem):
    def prepare(self):
        self.criterions = {
            'psnr': PSNR(),
            'ssim': SSIM()
        }
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        self.train_num_rays = self.config.model.train_num_rays

        self.gt_info = load_gt_info_pris(self.config.model.motion_gt_path)
        self.gt_axis = load_gt_axis(self.config.model.motion_gt_path)

    
    def on_train_start(self) -> None:
        self.dataset = self.trainer.datamodule.train_dataloader().dataset
        plot_camera(self.dataset.vis_cam_0, self.get_save_path('camera/cam_start.ply'), color=[1, 0, 0])
        plot_camera(self.dataset.vis_cam_1, self.get_save_path('camera/cam_end.ply'), color=[1, 0, 0])
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
    
    def configure_optimizers(self):
        model_optim = parse_optimizer(self.config.system.model_optimizer, self.model)
        motion_optim = parse_optimizer(self.config.system.motion_optimizer, self.model)
        model_scheduler = parse_scheduler(self.config.system.model_scheduler, model_optim)
        motion_scheduler = parse_scheduler(self.config.system.motion_scheduler, motion_optim)
        return [model_optim, motion_optim], [model_scheduler, motion_scheduler]

    def training_step(self, batch, batch_idx, optimizer_idx):
        outs = self.model(batch['rays_0'], batch['rays_1'])
        loss = 0.

        # photometric loss
        loss_rgb = (F.smooth_l1_loss(outs[0]['comp_rgb'][outs[0]['rays_valid']], batch['rgb_0'][outs[0]['rays_valid']]) + \
                    F.smooth_l1_loss(outs[1]['comp_rgb'][outs[1]['rays_valid']], batch['rgb_1'][outs[1]['rays_valid']])) * 0.5
        self.log('train/loss_rgb', loss_rgb, prog_bar=True)
        loss += loss_rgb * self.config.system.loss.lambda_rgb

        # object mask loss
        loss_mask = (binary_cross_entropy(outs[0]['opacity'], batch['fg_mask_0'].float()) + \
                     binary_cross_entropy(outs[1]['opacity'], batch['fg_mask_1'].float())) * 0.5
        self.log('train/loss_mask', loss_mask, prog_bar=True)
        loss += loss_mask * self.config.system.loss.lambda_mask

        # binary entropy loss for the blend ratio
        if (optimizer_idx == 0) and ((self.global_step /2) > self.config.system.loss.reg_from_iter):
            if self.config.system.loss.lambda_blend_ratio > 0.:
                skew = self.config.system.loss.skew_blend_ratio
                loss_ratio = (entropy_loss(outs[0]['ratio']) + entropy_loss(outs[1]['ratio'])) * 0.5
                self.log('train/loss_ratio', loss_ratio, prog_bar=True)
                loss += loss_ratio * self.config.system.loss.lambda_blend_ratio
            if self.config.system.loss.lambda_part_mask > 0.:
                loss_part = (entropy_loss(outs[0]['part_mask_ratio']) + entropy_loss(outs[1]['part_mask_ratio'])) * 0.5
                self.log('train/loss_part_ratio', loss_part, prog_bar=True) 
                loss += loss_part * self.config.system.loss.lambda_part_mask
        


        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / \
                            (0.5*(outs[0]['num_samples'].sum().item() + outs[1]['num_samples'].sum().item())+1e-15)))        
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
        
        self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

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
        # image metrics
        psnr, ssim = self.evaluate_nvs(outs[0]['comp_rgb'], batch['rgb_0'], outs[1]['comp_rgb'], batch['rgb_1'])

        # convert format of motion params
        motion = self.convert_motion_format()
        
        # save the images
        self.save_visuals(outs, batch, mode='val', draw_axis=True, motion=motion, elems=['gt', 'rgb', 'dep'])

        del outs
        return {
            'psnr': psnr,
            'ssim': ssim,
            'index': batch['index'],
            'motion': motion,
        }
          
    
    """
    # aggregate outputs from different devices when using DP
    def validation_step_end(self, out):
        pass
    """
    
    def validation_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            self.save_metrics(out, mode='val')
        del out

    def test_step(self, batch, batch_idx):  
        outs = self(batch)
        # image metrics
        psnr, ssim = self.evaluate_nvs(outs[0]['comp_rgb'], batch['rgb_0'], outs[1]['comp_rgb'], batch['rgb_1'])
        # convert format of motion params
        motion = self.convert_motion_format()
        # save the images
        self.save_visuals(outs, batch, mode='test', draw_axis=False, motion=motion, elems=['gt', 'rgb', 'dep'])

        del outs
        return {
            'psnr': psnr,
            'ssim': ssim,
            'index': batch['index'],
            'motion': motion,
        }       
    
    def test_epoch_end(self, out):
        out = self.all_gather(out)
        if self.trainer.is_global_zero:
            self.save_metrics(out, mode='test')       
        del out   


    # def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
    #     from models.utils import chunk_batch
    #     n_interpolate = 4 # render n_interpolate+1 imgs in total
    #     max_state= 1
    #     interval = max_state / n_interpolate
    #     self.model.eval()
    #     imgs, depths = [], []
    #     i = 0
    #     W, H = self.dataset.w, self.dataset.h
    #     while i < max_state or i == max_state:
    #         out = chunk_batch(self.model.forward_, self.config.model.ray_chunk, batch['rays'], scene_state=i)
    #         img = out['comp_rgb'].view(H, W, 3)
    #         # dep = out['depth'].view(H, W)
    #         imgs.append(img)
    #         # depths.append(dep)
    #         print('finish state ', i)
    #         i += interval
    #         del out
    #     img_grid = [{'type': 'rgb', 'img': batch['rgb_0'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}]
    #     img_grid += [{'type': 'rgb', 'img': img, 'kwargs': {'data_format': 'HWC'}} for img in imgs]
    #     img_grid.append({'type': 'rgb', 'img': batch['rgb_1'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}})
    #     source = self.config.source
    #     tokens = source.split('/')
    #     self.save_image_grid(f"it{self.global_step}-pred/{tokens[1]}_{self.config.dataset.view_idx}_RGB.png", img_grid)
    #     # self.save_image_grid(f"it{self.global_step}-pred/{self.config.dataset.view_idx}_depth.png", \
    #     #     [{'type': 'grayscale', 'img': dep, 'kwargs': {}} for dep in depths])

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        # animation
        import cv2
        import numpy as np
        from models.utils import chunk_batch
        n_interpolate = 20 # render n_interpolate+1 imgs in total
        max_state= 1
        interval = max_state / n_interpolate
        self.model.eval()
        imgs, img_paths = [], []
        i = 0
        W, H = self.dataset.w, self.dataset.h
        while i < (max_state+interval):
            out = chunk_batch(self.model.forward_, self.config.model.ray_chunk, batch['rays'], scene_state=i)
            img = (out['comp_rgb'].view(H, W, 3).clip(0, 1) * 255).cpu().numpy().astype(np.uint8)
            img_path = self.get_save_path(f"it{self.global_step}-anim/{self.config.dataset.view_idx}_state{i}.png")
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(img_path, img)
            imgs.append(img)
            img_paths.append(img_path)
            # depths.append(dep)
            print('finish state ', i)
            i += interval
        # video
        name = self.config.source.split('/')
        imgss = [cv2.imread(f) for f in img_paths]
        exp_path = self.get_save_path(f"it{self.global_step}-anim/{name}.mp4")
        H, W, _ = imgss[0].shape
        writer = cv2.VideoWriter(exp_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (W, H), True)
        for img in imgss:
            writer.write(img)
        
        n = len(imgss)
        i = n - 1
        while i > 0 or i == 0:
            img = imgss[i]
            writer.write(img)
            i -= 1
        
        writer.release()
    
    def convert_motion_format(self):
        # output motion params
        axis_o = self.model.axis_o.detach()
        axis_d = self.model.axis_d.detach()
        dist = self.model.dist.detach()
        axis_d = F.normalize(axis_d, p=2., dim=0)

        motion = {
            'type': 'translate',
            'axis_o': axis_o,
            'axis_d': axis_d,
            'dist': dist
        }
        return motion