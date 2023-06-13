import torch
import torch.nn.functional as F
import systems
from systems.base import BaseSystem
from systems.criterions import binary_cross_entropy, entropy_loss
from utils.rotation import quaternion_to_axis_angle, R_from_axis_angle


@systems.register('revolute-system')
class RevoluteSystem(BaseSystem):
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

        # regularization only for nerfs (not for motion parameters)
        if (optimizer_idx == 0) and ((self.global_step/2) > self.config.system.loss.reg_from_iter):
            if self.config.system.loss.lambda_blend_ratio > 0.:
                loss_ratio = (entropy_loss(outs[0]['ratio']) + entropy_loss(outs[1]['ratio'])) * 0.5
                self.log('train/loss_ratio', loss_ratio)
                loss += loss_ratio * self.config.system.loss.lambda_blend_ratio
            if self.config.system.loss.lambda_part_mask > 0.:
                loss_part = (entropy_loss(outs[0]['part_mask_ratio']) + entropy_loss(outs[1]['part_mask_ratio'])) * 0.5
                self.log('train/loss_part_ratio', loss_part, prog_bar=True)
                loss += loss_part * self.config.system.loss.lambda_part_mask

        # update train_num_rays
        if self.config.model.dynamic_ray_sampling:
            num_samples = 0.5*(outs[0]['num_samples'].sum().item() + outs[1]['num_samples'].sum().item())+1e-15
            train_num_rays = int(self.train_num_rays * (self.train_num_samples / num_samples))                   
            self.train_num_rays = min(int(self.train_num_rays * 0.9 + train_num_rays * 0.1), self.config.model.max_train_num_rays)
            self.log('train/num_rays', float(self.train_num_rays), prog_bar=True)

        return loss    

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

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        self.model.eval()
        from models.utils import chunk_batch
        pred_mode = self.config.dataset.get('pred_mode', 'grid')
        n_interp = self.config.dataset.get('n_interp', 3)
        max_state= self.config.dataset.get('max_state',1)
        interval = max_state / (n_interp + 1)
        W, H = self.dataset.w, self.dataset.h
        i = 0
        if pred_mode == 'grid': # concat the image grid from state 0 to state max_state
            imgs = []
            while i < (max_state+interval):
                out = chunk_batch(self.model.forward_, self.config.model.ray_chunk, batch['rays'], scene_state=i)
                img = out['comp_rgb'].view(H, W, 3)
                imgs.append(img)
                print('finish state ', i)
                i += interval
            # image grid
            img_grid = [{'type': 'rgb', 'img': batch['rgb_0'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}] # start with GT
            img_grid += [{'type': 'rgb', 'img': img, 'kwargs': {'data_format': 'HWC'}} for img in imgs]
            img_grid.append({'type': 'rgb', 'img': batch['rgb_1'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}) # end with GT
            self.save_image_grid(f"it{self.global_step}-pred/{self.config.dataset.view_idx}_RGB.png", img_grid)
        elif pred_mode == 'anim': # generate states and export the animation video
            img_paths = []
            while i < (max_state+interval):
                out = chunk_batch(self.model.forward_, self.config.model.ray_chunk, batch['rays'], scene_state=i)
                fname = f"it{self.global_step}-anim/{self.config.dataset.view_idx}_state{round(i, 3)}.png"
                self.save_rgb_image(fname, out['comp_rgb'].view(H, W, 3), data_format='HWC', data_range=(0, 1))
                img_path = self.get_save_path(fname)
                img_paths.append(img_path)
                print('finish state ', i)
                i += interval
            # video
            self.save_anim_video(f"it{self.global_step}-anim", img_paths, save_format='mp4', fps=10)
        else:
            raise NotImplementedError

    
    def convert_motion_format(self):
        # output motion params
        axis_o = self.model.axis_o.detach()
        quaternions = self.model.quaternions.detach()

        # the output from the network is the rotation angle from t=0 to t=0.5   
        axis_d, half_angle = quaternion_to_axis_angle(quaternions) 
        angle = 2. * half_angle 
        # convert to rotation matrix (from t=0 to t=1)
        if torch.isnan(axis_d).sum().item() > 0: # the rotation is an Identity Matrix
            axis_d = torch.ones(3, device=self.local_rank) # random direction
            angle = torch.zeros(1, device=self.local_rank)
        else:
            R = R_from_axis_angle(axis_d, angle)
        # convert to degree
        rot_angle = torch.rad2deg(angle)
        motion = {
            'type': 'rotate',
            'axis_o': axis_o,
            'axis_d': axis_d,
            'rot_angle': rot_angle,
            'R': R,
        }
        return motion