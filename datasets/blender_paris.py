import os, sys
# ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(ROOT_DIR)
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import torchvision.transforms.functional as TF

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import _get_rank

import datasets
from models.ray_utils import get_ray_directions
from utils.axis import get_rotation_axis_angle
from pytorch3d.ops import sample_farthest_points

import open3d as o3d


class BlenderDatasetBase():
    def setup(self, config, split):
        self.config = config

        self.view_downsample = self.config.get('view_downsample', False)
        self.n_downsample = self.config.get('n_downsample', 0)


        self.split = split
        if split == 'train':
            img_scale = self.config.train_scale
        elif split == 'val': 
            val_scale = self.config.get('val_scale', self.config.test_scale)
            img_scale = val_scale
        elif split == 'test':
            img_scale = self.config.test_scale
        elif split == 'pred':
            img_scale = self.config.pred_scale
        else:
            raise NotImplementedError
        
        self.w, self.h = int(self.config.img_wh[0] * img_scale), int(self.config.img_wh[1] * img_scale)
        self.near, self.far = self.config.near_plane, self.config.far_plane
        self.rank = _get_rank()

        if split == 'pred':
            self.setup_pred(img_scale, view_idx=str(config.get('view_idx')).rjust(4, "0"))
            return
        

        # load data for two states
        self.K_0, self.all_c2w_0, self.all_images_0, self.all_fg_masks_0, self.directions_0, self.vis_cam_0 = self.load_data('start', img_scale)
        self.K_1, self.all_c2w_1, self.all_images_1, self.all_fg_masks_1, self.directions_1, self.vis_cam_1 = self.load_data('end', img_scale)



    def load_data(self, state, img_scale):
        with open(os.path.join(self.config.root_dir, state, f"camera_{self.split}.json"), 'r') as f:
            cam_dict = json.load(f)
            f.close()
        # intrinsics
        K_ = np.array(cam_dict['K']).astype(np.float32)
        focal = K_[0][0] * img_scale
        K = torch.tensor(K_)
        K[0][0], K[1][1] = focal, focal
        K[0][2], K[1][2] = self.w/2, self.h/2
 
        # ray directions for all pixels, same for all images here
        directions = get_ray_directions(self.h, self.w, focal, self.config.use_pixel_centers).to(self.rank)     

        all_c2w, all_images, all_fg_masks = [], [], []

        cam_dict.pop('K')
        vis_cam = []
        for k, v in cam_dict.items():
            pose = np.array(v).astype(np.float32)
            if self.split == 'train':
                vis_cam.append(torch.from_numpy(pose))
            c2w = torch.from_numpy(pose[:3, :4])
            all_c2w.append(c2w)

            img_path = os.path.join(self.config.root_dir, state, self.split,  f"{k}.png")
            img = Image.open(img_path)
            img = img.resize((self.w, self.h), Image.BICUBIC)
            img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)

            all_fg_masks.append(img[..., -1]>0) # (h, w)
            if self.config.white_bkgd:
                img = img[...,:3] * img[...,-1:] + (1 - img[...,-1:]) # blend A to RGB
            else:
                img = img[...,:3] * img[...,-1:]
            all_images.append(img)

        if self.split == 'train' and self.view_downsample:
            all_c2w, all_images, all_fg_masks = \
            torch.stack(all_c2w, dim=0).float(), \
            torch.stack(all_images, dim=0).float(), \
            torch.stack(all_fg_masks, dim=0).float()

            if self.config.random_downsample:
                ds_idx = torch.randint(low=0, high=100, size=(self.n_downsample,))
            else:
                cam_pos = all_c2w[:, :3, 3]
                _, ds_idx = sample_farthest_points(cam_pos[None, :], K=self.n_downsample, random_start_point=True)
                ds_idx = ds_idx[0]

            vis_cam = torch.stack(vis_cam, dim=0).float()
            vis_cam = vis_cam[ds_idx].tolist()
            all_c2w = all_c2w[ds_idx].to(self.rank)
            all_images = all_images[ds_idx].to(self.rank)
            all_fg_masks = all_fg_masks[ds_idx].to(self.rank)
        else:
            all_c2w, all_images, all_fg_masks = \
            torch.stack(all_c2w, dim=0).float().to(self.rank), \
            torch.stack(all_images, dim=0).float().to(self.rank), \
            torch.stack(all_fg_masks, dim=0).float().to(self.rank)
        
        return K, all_c2w, all_images, all_fg_masks, directions, vis_cam


    def setup_pred(self, img_scale, view_idx='0000'):
        print('view_idx', view_idx)
        with open(os.path.join(self.config.root_dir, 'start', f"camera_test.json"), 'r') as f:
            cam_dict = json.load(f)
            f.close()

        K = np.array(cam_dict['K']).astype(np.float32)
        focal = K[0][0] * img_scale
        self.K = torch.tensor(K)
        self.K[0][0], self.K[1][1] = focal, focal
        self.K[0][2], self.K[1][2] = int(self.w/2), int(self.h/2)

        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = \
            get_ray_directions(self.h, self.w, focal, self.config.use_pixel_centers).to(self.rank) # (h, w, 3)  

        start_img_path = os.path.join(self.config.root_dir, 'start', 'test',  f"{view_idx}.png")
        end_img_path = os.path.join(self.config.root_dir, 'end', 'test',  f"{view_idx}.png")
        img = Image.open(start_img_path)
        img = img.resize((self.w, self.h), Image.BICUBIC)
        img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
        start_img = img[...,:3] * img[...,-1:] + (1 - img[...,-1:]) # blend A to RGB

        img = Image.open(end_img_path)
        img = img.resize((self.w, self.h), Image.BICUBIC)
        img = TF.to_tensor(img).permute(1, 2, 0) # (4, h, w) => (h, w, 4)
        end_img = img[...,:3] * img[...,-1:] + (1 - img[...,-1:]) # blend A to RGB

        self.start_img = start_img.float().unsqueeze(0)
        self.end_img = end_img.float().unsqueeze(0)

        cam_dict.pop('K')
        v = cam_dict[view_idx]
        pose = np.array(v).astype(np.float32)
        self.all_c2w = torch.from_numpy(pose[:3, :4]).float().unsqueeze(0).to(self.rank)


class BlenderDataset(Dataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __len__(self):
        if self.split != 'pred':
            return len(self.all_images_0)
        else:
            return len(self.all_c2w)
    
    def __getitem__(self, index):
        return {
            'index': index
        }


class BlenderIterableDataset(IterableDataset, BlenderDatasetBase):
    def __init__(self, config, split):
        self.setup(config, split)

    def __iter__(self):
        while True:
            yield {}
            
    # def __len__(self):
    #     return len(self.all_images_0)
    
        

@datasets.register('blender_paris')
class BlenderDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.misalign = self.config.get('misalign', False)
        if self.misalign:         
            R_align, t_align, axis_R_align = self.simulate_misalign(std_angle=self.config.std_angle, std_dist=self.config.std_dist)
            config.update({"align_params": {
                "R": R_align.tolist(),
                "t": t_align.tolist(),
                "axis_R": axis_R_align.tolist()
            }})
    
    def simulate_misalign(self, std_angle=15, std_dist=0.01):
        # rotation error
        theta = np.radians(std_angle)
        random_axis = np.random.normal(0., 1., 3)
        R = get_rotation_axis_angle(random_axis, theta)
        print('misalignment R: ', R)           
        # translational error
        t = np.random.normal(0., std_dist, 3)
        print('misalignment t: ', t)
        return R, t, random_axis
    
    def setup(self, stage=None):
        if stage in [None, 'fit']:
            self.train_dataset = BlenderIterableDataset(self.config, self.config.train_split)
        if stage in [None, 'fit', 'validate']:
            self.val_dataset = BlenderDataset(self.config, self.config.val_split)
        if stage in [None, 'test']:
            self.test_dataset = BlenderDataset(self.config, self.config.test_split)
        if stage in [None, 'predict']:
            self.predict_dataset = BlenderDataset(self.config, self.config.pred_split)

    def prepare_data(self):
        pass
    
    def general_loader(self, dataset, batch_size):
        return DataLoader(
            dataset, 
            # num_workers=os.cpu_count(), 
            num_workers=0, 
            batch_size=batch_size,
            pin_memory=True,
        )
    
    def train_loader(self, dataset, batch_size) :
        return DataLoader(
            dataset, 
            # num_workers=os.cpu_count(), 
            # num_workers=len(os.sched_getaffinity(0)),
            num_workers=0,

            batch_size=batch_size,
            pin_memory=True,
        )
    
    def train_dataloader(self):
        return self.train_loader(self.train_dataset, batch_size=1)

    def val_dataloader(self):
        return self.general_loader(self.val_dataset, batch_size=1)

    def test_dataloader(self):
        return self.general_loader(self.test_dataset, batch_size=1) 

    def predict_dataloader(self):
        return self.general_loader(self.predict_dataset, batch_size=1)    