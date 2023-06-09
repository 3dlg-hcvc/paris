import torch
import torch.nn.functional as F
import systems
from systems.base import BaseSystem
from systems.criterions import binary_cross_entropy

from systems.utils import load_gt_info
from utils.rotation import R_from_quaternions, R_from_axis_angle
from utils.plot_camera import plot_camera


@systems.register('se3-system')
class SE3System(BaseSystem):
    '''This framework is used to identify the motion type only'''

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
        self.save_visuals(outs, batch, mode='val', draw_axis=False, elems=['gt', 'rgb', 'dep'])

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
        self.save_visuals(outs, batch, mode='test', draw_axis=False, elems=['gt', 'rgb', 'dep'])

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

    def decompose_se3(self, q, t):
        '''
        decomposition of SE(3) group
            * support rotation or translation only
            * if rotation, decompose into a axis origin, direction, and a angle (might end up with a invalid axis)
            * if translation, decompose into a tranlational direction and distance
        The inputs are the params from t=0 to t=0.5
        '''
        R = R_from_quaternions(q)
        R = R.cpu()
        t = t.cpu()
        # decompose R to axis-angle    
        theta = torch.arccos(0.5 * (R[0][0]+R[1][1]+R[2][2] - 1))
        coeff = 1. / (2 * torch.sin(theta))
        k = coeff * torch.tensor([R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]]).float()
        k = F.normalize(k, p=2, dim=0)
        # classify a motion type based on the degree of angle
        angle_theta = torch.rad2deg(theta)
        if angle_theta < 5.:
            motion_type = 'translate'
        else:
            motion_type = 'rotate'
        
        ## if translation
        t_unit = t / torch.linalg.norm(t)
        t_dist = torch.linalg.norm(t)

        ## if rotation
        # Compute the determinant of the matrix to see if R is invertible
        det = torch.det(torch.eye(3, 3) - R)
        if torch.abs(det) < 1e-8: # Check if the matrix is singular
            print("The (I - R) matrix is not invertible. Invalid axis.")
            center = torch.zeros(3)
        else:
            center = torch.matmul(torch.linalg.inv(torch.eye(3, 3) - R), t) 
        # scale the rotational center as close to the origin as possible
        center = center + torch.dot(k, -center) * k 

        # return the transformation from t=0 to t=1
        R_ang = angle_theta * 2.
        t_dist = t_dist * 2
        R_mat = R_from_axis_angle(k, R_ang)
        return {
            'type': motion_type,
            't_dist': t_dist,
            't_axis_d': t_unit,
            'R_axis_o': center,
            'R_axis_d': k,
            'R_ang': R_ang,
            'R_mat': R_mat
        }
    
    def convert_motion_format(self):
        # output motion params
        translation = self.model.translation.detach()
        quaternions = self.model.quaternions.detach()
        # decompose SE(3) into rotation and translation respectively
        motion = self.decompose_se3(quaternions, translation)
        return motion
    
    def save_metrics(self, out, mode='val'):
        it = int(self.global_step/2)
        motion = out[0]['motion']  
        motion_type = self.gt_info['type'] 
        motion.update({'gt_type': motion_type})
        
        ## image metrics 
        psnr, ssim = self.metrics_nvs(out, mode)

        # export motion
        self.export_motion(motion, mode)

        # metrics
        metrics = {
            "nvs": {
                "psnr": psnr,
                "ssim": ssim,
            }
        }

        # save the metrics
        if motion_type == 'rotate':
            motion_ref = {
                'axis_d': motion['R_axis_d'],
                'axis_o': motion['R_axis_o'],
                'rot_angle': motion['R_ang'],
                'R': motion['R_mat']
            }
            errs = self.metrics_motion(motion_ref, mode)
            metrics.update({
                "motion": {
                    "ang_err": errs['ang_err'],
                    "pos_err": errs['pos_err'],
                    "geo_dist": errs['geo_dist']
                }
            }) 
        elif motion_type == 'translate':
            motion_ref = {
                'axis_d': motion['t_axis_d'],
                'dist': motion['t_dist'],
            }
            errs = self.metrics_motion(motion_ref, mode)
            metrics.update({
                "motion": {
                    "ang_err": errs['ang_err'],
                    "trans_err": errs['trans_err'],
                }
            }) 

        if mode == 'test':
            # export meshes
            self.export_meshes(motion_ref)
            # save meshes for motion axis
            self.save_axis(f'it{it}_axis.ply', motion)

            # metrics for surface quality
            cd_s, cd_d_start, cd_w_start = self.metrics_surface()

            
            # video for the testing images
            it = int(self.global_step/2)
            self.save_img_sequence(
                f"it{it}-test",
                f"it{it}-test",
                '(\d+)\.png',
                save_format='mp4',
                fps=10
            )

            metrics.update({
                "surface": {
                    "CD-w": cd_w_start,
                    "CD-s": cd_s,
                    "CD-d": cd_d_start,
                }
            })
        
        self.save_json(f'it{it}_{mode}_metrics.json', metrics)
    
