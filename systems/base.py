import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import models
from tqdm import tqdm
from models.ray_utils import get_rays
from systems.utils import parse_optimizer, parse_scheduler, update_module_step, proj2img, load_gt_info
from utils.mixins import SaverMixin
from utils.misc import config_to_primitive
from systems.criterions import binary_cross_entropy, entropy_loss
from systems.criterions import PSNR, SSIM, geodesic_distance, axis_metrics, translational_error
from os.path import join, dirname
from utils.chamfer import eval_CD

class BaseSystem(pl.LightningModule, SaverMixin):
    """
    Two ways to print to console:
    1. self.print: correctly handle progress bar
    2. rank_zero_info: use the logging module
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = models.make(self.config.model.name, self.config.model)
        self.train_num_samples = self.config.model.train_num_rays * self.config.model.num_samples_per_ray
        self.train_num_rays = self.config.model.train_num_rays
        self.gt_info = load_gt_info(self.config.model.motion_gt_path)
    
    def configure_optimizers(self):
        model_optim = parse_optimizer(self.config.system.model_optimizer, self.model)
        motion_optim = parse_optimizer(self.config.system.motion_optimizer, self.model)
        model_scheduler = parse_scheduler(self.config.system.model_scheduler, model_optim)
        motion_scheduler = parse_scheduler(self.config.system.motion_scheduler, motion_optim)
        return [model_optim, motion_optim], [model_scheduler, motion_scheduler]

    """
    Implementing on_after_batch_transfer of DataModule does the same.
    But on_after_batch_transfer does not support DP.
    """
    def on_train_batch_start(self, batch, batch_idx, unused=0):
        self.dataset = self.trainer.datamodule.train_dataloader().dataset
        self.preprocess_data(batch, 'train')
        update_module_step(self.model, self.current_epoch, self.global_step)
    
    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx):
        self.preprocess_data(batch, 'validation')
        update_module_step(self.model, self.current_epoch, self.global_step)
    
    def on_test_batch_start(self, batch, batch_idx, dataloader_idx):
        self.preprocess_data(batch, 'test')
        update_module_step(self.model, self.current_epoch, self.global_step)

    def on_predict_batch_start(self, batch, batch_idx, dataloader_idx):
        self.preprocess_data(batch, 'predict')
        update_module_step(self.model, self.current_epoch, self.global_step)
    
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
        # convert format of motion params
        motion = self.convert_motion_format()
        if not self.config.get('mesh_only', False):
            # image metrics
            psnr, ssim = self.evaluate_nvs(outs[0]['comp_rgb'], batch['rgb_0'], outs[1]['comp_rgb'], batch['rgb_1'])
            # save the images
            self.save_visuals(outs, batch, mode='val', draw_axis=True, motion=motion, elems=['gt', 'rgb', 'dep'])
            return {
                'psnr': psnr,
                'ssim': ssim,
                'index': batch['index'],
                'motion': motion,
            }
        return {
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

    def test_step(self, batch, batch_idx):  
        # convert format of motion params
        motion = self.convert_motion_format()
        # render images
        if not self.config.get('mesh_only', False):
            outs = self(batch)
            # image metrics
            psnr, ssim = self.evaluate_nvs(outs[0]['comp_rgb'], batch['rgb_0'], outs[1]['comp_rgb'], batch['rgb_1'])
            # save the images
            self.save_visuals(outs, batch, mode='test', draw_axis=False, motion=motion, elems=['gt', 'rgb', 'dep'])
            return {
                'psnr': psnr,
                'ssim': ssim,
                'index': batch['index'],
                'motion': motion,
            }
        return {
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
        assert n_interp > 0 or n_interp == 0
        max_state= self.config.dataset.get('max_state',1)
        assert max_state > 0 
        interval = max_state / (n_interp + 1)
        W, H = self.dataset.w, self.dataset.h
        state_i = 0
        if pred_mode == 'grid': # concat the image grid from state 0 to state max_state
            imgs = []
            for k in tqdm(range(n_interp+2)):
                out = chunk_batch(self.model.forward_, self.config.model.ray_chunk, batch['rays'], scene_state=state_i)
                img = out['comp_rgb'].view(H, W, 3)
                imgs.append(img)
                # print('finish state ', state_i)
                state_i += interval
            # image grid
            img_grid = [{'type': 'rgb', 'img': batch['rgb_0'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}] # start with GT
            img_grid += [{'type': 'rgb', 'img': img, 'kwargs': {'data_format': 'HWC'}} for img in imgs]
            img_grid.append({'type': 'rgb', 'img': batch['rgb_1'].view(H, W, 3), 'kwargs': {'data_format': 'HWC'}}) # end with GT
            self.save_image_grid(f"it{self.global_step}-pred/{self.config.dataset.view_idx}_RGB.png", img_grid)
        elif pred_mode == 'anim': # generate states and export the animation video
            img_paths = []
            for k in tqdm(range(n_interp+2)):
                out = chunk_batch(self.model.forward_, self.config.model.ray_chunk, batch['rays'], scene_state=state_i)
                fname = f"it{self.global_step}-anim/{self.config.dataset.view_idx}_state{round(state_i, 3)}.png"
                self.save_rgb_image(fname, out['comp_rgb'].view(H, W, 3), data_format='HWC', data_range=(0, 1))
                img_path = self.get_save_path(fname)
                img_paths.append(img_path)
                # print('finish state ', i)
                state_i += interval
            # video
            self.save_anim_video(f"it{self.global_step}-anim", img_paths, save_format='mp4', fps=10)
            # gif
            self.save_anim_video(f"it{self.global_step}-anim", img_paths, save_format='gif', fps=10)

        else:
            raise NotImplementedError

    def convert_motion_format(self):
        raise NotImplementedError
    
    def C(self, value):
        if isinstance(value, int) or isinstance(value, float):
            pass
        else:
            value = config_to_primitive(value)
            if not isinstance(value, list):
                raise TypeError('Scalar specification only supports list, got', type(value))
            if len(value) == 3:
                value = [0] + value
            assert len(value) == 4
            start_step, start_value, end_value, end_step = value
            if isinstance(end_step, int):
                current_step = self.global_step
                value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
            elif isinstance(end_step, float):
                current_step = self.current_epoch
                value = start_value + (end_value - start_value) * max(min(1.0, (current_step - start_step) / (end_step - start_step)), 0.0)
        return value
    
    def preprocess_data(self, batch, stage):
        if 'index' in batch:
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
        else: # validation / testing 
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
    
    def vis_motion_axis(self, motion, gt):
        axis_o = motion['axis_o'].unsqueeze(0)
        axis_d = motion['axis_d'].unsqueeze(0)
        pred_axis = torch.cat([axis_o, axis_o+axis_d], dim=0)

        gt_axis_0 = proj2img(self.gt_info['vis_axis'], gt['w2c_0'], gt['K_0'])
        gt_axis_1 = proj2img(self.gt_info['vis_axis'], gt['w2c_1'], gt['K_1'])
        pred_axis_0 = proj2img(pred_axis, gt['w2c_0'], gt['K_0'])
        pred_axis_1 = proj2img(pred_axis, gt['w2c_1'], gt['K_1'])

        axis_info_0 = {
            'GT': gt_axis_0,
            'pred': pred_axis_0,
        }
        axis_info_1 = {
            'GT': gt_axis_1,
            'pred': pred_axis_1,
        }
        return [axis_info_0, axis_info_1]

    def evaluate_nvs(self, src_0, tgt_0, src_1, tgt_1):
        criterions = {
            'psnr': PSNR(),
            'ssim': SSIM()
        }
        W, H = self.dataset.w, self.dataset.h
        psnr_0 = criterions['psnr'](src_0, tgt_0)
        psnr_1 = criterions['psnr'](src_1, tgt_1)
        ssim_0 = criterions['ssim'](src_0.view(1, H, W, 3).cpu(), tgt_0.view(1, H, W, 3).cpu())
        ssim_1 = criterions['ssim'](src_1.view(1, H, W, 3).cpu(), tgt_1.view(1, H, W, 3).cpu())

        psnr = 0.5 * (psnr_0 + psnr_1)
        ssim = 0.5 * (ssim_0 + ssim_1)
        return psnr, ssim

    def export_meshes(self, motion):
        it = int(self.global_step/2)
        # configurations
        res = self.config.model.geometry.isosurface.resolution
        thre = float(self.config.model.geometry.isosurface.threshold)
        dyn_can_filename = f"it{it}_dynamic_{res}_thre{thre}.ply"
        dyn_start_filename = f"it{it}_dyn_t0_{res}_thre{thre}.ply"
        dyn_end_filename = f"it{it}_dyn_t1_{res}_thre{thre}.ply"

        # extract geometry from the fields
        mesh_dict = self.model.isosurface()
        # save static and dynamic parts in the canonical state
        for name, mesh in mesh_dict.items():
            self.save_mesh_ply(f"it{it}_{name}_{res}_thre{thre}.ply", **mesh)
        # save dynamic part at given states
        self.save_trans_part_mesh(dyn_can_filename, [dyn_start_filename, dyn_end_filename], motion)
    
    def export_motion(self, motion, mode='val'):
        it = int(self.global_step/2)
        # save json file for motion information
        motion_json = {}
        for key, values in motion.items():
            if type(values) is str:
                motion_json.update({key: values})
            else:
                motion_json.update({key: values[0].cpu().tolist()})
        self.save_json(f'it{it}_{mode}_motion.json', motion_json)

    def metrics_motion(self, motion, mode='val'):
        motion_type = self.gt_info['type']
        if motion_type == 'rotate':
            # rotation difference
            geo_dist = geodesic_distance(motion['R'].squeeze_(0), self.gt_info['R'])
            self.log(f'{mode}/geo_dist', geo_dist, prog_bar=True, rank_zero_only=True)
            # angular and positional errors
            ang_err, pos_err = axis_metrics(motion, self.gt_info)
            self.log(f'{mode}/ang_err', ang_err, prog_bar=True, rank_zero_only=True)
            self.log(f'{mode}/pos_err', pos_err, prog_bar=True, rank_zero_only=True)
            return {
                "geo_dist": geo_dist.item(), 
                "ang_err": ang_err.item(), 
                "pos_err": pos_err.item()
            }
        elif motion_type == 'translate':
            # translational error
            trans_err = translational_error(motion, self.gt_info)
            self.log(f'{mode}/trans_err', trans_err, prog_bar=True, rank_zero_only=True)
            # angular errors
            ang_err, _ = axis_metrics(motion, self.gt_info)
            self.log(f'{mode}/ang_err', ang_err, prog_bar=True, rank_zero_only=True)
            return {
                "trans_err": trans_err.item(), 
                "ang_err": ang_err.item(), 
            }
        else:
            raise ValueError("the motion type is not supported")
    
    def metrics_nvs(self, out, mode='val'):
        ## image metrics
        psnr = out[0]['psnr'].item()
        self.log(f'{mode}/psnr', psnr, prog_bar=True, rank_zero_only=True)  
        ssim = out[0]['ssim'].item()
        self.log(f'{mode}/ssim', ssim, prog_bar=True, rank_zero_only=True)
        return psnr, ssim
    
    def metrics_surface(self):
        it = int(self.global_step/2)
        # configurations
        res = self.config.model.geometry.isosurface.resolution
        thre = float(self.config.model.geometry.isosurface.threshold)

        # Chamfer-L1 Distance at start state
        cd_s, cd_d_start, cd_w_start = eval_CD(
            self.get_save_path(f"it{it}_static_{res}_thre{thre}.ply"),
            self.get_save_path(f"it{it}_dyn_t0_{res}_thre{thre}.ply"),
            self.get_save_path(f"it{it}_start_{res}_thre{thre}.ply"),
            join(dirname(self.config.model.motion_gt_path), 'start', 'start_static_rotate.ply'),
            join(dirname(self.config.model.motion_gt_path), 'start', 'start_dynamic_rotate.ply'),
            join(dirname(self.config.model.motion_gt_path), 'start', 'start_rotate.ply')
        )
        return cd_s, cd_d_start, cd_w_start
    
    def save_visuals(self, outs, gt, mode='val', draw_axis=True, motion=None, elems=['gt', 'rgb', 'dep', 'opa']):
        # visual elements
        visuals = self.visual_elem(outs)
        
        # project predicted motion axis
        axis_info = [{}, {}] 
        if draw_axis:
            assert motion is not None
            axis_info = self.vis_motion_axis(motion, gt)

        # save image grid
        W, H = self.dataset.w, self.dataset.h
        if mode == 'test':
            filename = f"it{int(self.global_step/2)}-test/{gt['index'][0].item()}.png"
        else:
            filename = f"it{int(self.global_step/2)}_{gt['index'][0].item()}.png"
        
        grid_info = [[], []]
        for i in range(2):
            if 'gt' in elems:
                grid_info[i].append({
                    'type': 'rgb', 
                    'img': gt[f'rgb_{str(i)}'].view(H, W, 3), 
                    'kwargs': {
                        'data_format': 'HWC', 
                        'draw_axis': draw_axis, 
                        'axis_info': axis_info[i]}
                })
            if 'rgb' in elems:
               grid_info[i].append({'type': 'rgb', 'img': visuals['rgb'][i], 'kwargs': {'data_format': 'HWC'}}) 
               grid_info[i].append({'type': 'rgb', 'img': visuals['rgb_s'][i], 'kwargs': {'data_format': 'HWC'}})
               grid_info[i].append({'type': 'rgb', 'img': visuals['rgb_d'][i], 'kwargs': {'data_format': 'HWC'}})
            if 'dep' in elems:
                grid_info[i].append({'type': 'grayscale', 'img': visuals['dep'][i], 'kwargs': {}})
                grid_info[i].append({'type': 'grayscale', 'img': visuals['dep_s'][i], 'kwargs': {}})
                grid_info[i].append({'type': 'grayscale', 'img': visuals['dep_d'][i], 'kwargs': {}})
            if 'opa' in elems:
                grid_info[i].append({'type': 'grayscale', 'img': visuals['opa'][i], 'kwargs': {'cmap': None, 'data_range': (0, 1)}})

        self.save_image_grid(filename, grid_info)


    def visual_elem(self, outs):
        W, H = self.dataset.w, self.dataset.h
        rgb_0 = outs[0]['comp_rgb'].view(H, W, 3)
        rgb_1 = outs[1]['comp_rgb'].view(H, W, 3)
        rgb_s_0 = outs[0]['rgb_s'].view(H, W, 3)
        rgb_d_0 = outs[0]['rgb_d'].view(H, W, 3)
        rgb_s_1 = outs[1]['rgb_s'].view(H, W, 3)
        rgb_d_1 = outs[1]['rgb_d'].view(H, W, 3)
        depth_0 = outs[0]['depth'].view(H, W)
        depth_1 = outs[1]['depth'].view(H, W)
        depth_s_0 = outs[0]['depth_s'].view(H, W)
        depth_s_1 = outs[1]['depth_s'].view(H, W)
        depth_d_0 = outs[0]['depth_d'].view(H, W)
        depth_d_1 = outs[1]['depth_d'].view(H, W)
        opacity_0 = outs[0]['opacity'].view(H, W)
        opacity_1 = outs[1]['opacity'].view(H, W)
        
        return {
            'rgb': [rgb_0, rgb_1],
            'rgb_s': [rgb_s_0, rgb_s_1],
            'rgb_d': [rgb_d_0, rgb_d_1],
            'dep': [depth_0, depth_1],
            'dep_s': [depth_s_0, depth_s_1],
            'dep_d': [depth_d_0, depth_d_1],
            'opa': [opacity_0, opacity_1]
        }


    def save_metrics(self, out, mode='val'):
        it = int(self.global_step/2)
        motion = out[0]['motion']   
        motion_type = motion['type']
        metrics = {}

        # export motion
        self.export_motion(motion, mode)

        if not self.config.get('mesh_only', False):
            # image metrics 
            psnr, ssim = self.metrics_nvs(out, mode)
            metrics.update({
                "nvs": {
                    "psnr": psnr,
                    "ssim": ssim,
                }
            })
            if mode == 'test':
                # video for the testing images
                it = int(self.global_step/2)
                self.save_img_sequence(
                    f"it{it}-test",
                    f"it{it}-test",
                    '(\d+)\.png',
                    save_format='mp4',
                    fps=10
                )

        # metrics for motion estimation
        errs = self.metrics_motion(motion, mode)
        
        # save the metrics
        if motion_type == 'rotate':
            metrics.update({
                "motion": {
                    "ang_err": errs['ang_err'],
                    "pos_err": errs['pos_err'],
                    "geo_dist": errs['geo_dist']
                }
            }) 
        elif motion_type == 'translate':
            metrics.update({
                "motion": {
                    "ang_err": errs['ang_err'],
                    "trans_err": errs['trans_err'],
                }
            }) 

        # export meshes
        self.export_meshes(motion)
        # save meshes for motion axis
        self.save_axis(f'it{it}_axis.ply', motion)

        # metrics for surface quality
        cd_s, cd_d_start, cd_w_start = self.metrics_surface()


        metrics.update({
            "surface": {
                "CD-w": cd_w_start,
                "CD-s": cd_s,
                "CD-d": cd_d_start,
            }
        })
        
        self.save_json(f'it{it}_{mode}_metrics.json', metrics)