import torch
import torch.nn.functional as F
from os.path import join, dirname
import systems
from systems.base import BaseSystem
from systems.criterions import PSNR, SSIM, binary_cross_entropy, entropy_loss 
from systems.utils import parse_optimizer, parse_scheduler, load_gt_info
from utils.plot_camera import plot_camera

@systems.register('prismatic-system')
class PrismaticSystem(BaseSystem):
    def convert_motion_format(self):
        # output motion params
        axis_o = self.model.axis_o.detach()
        axis_d = self.model.axis_d.detach()
        dist_half = self.model.dist.detach()
        dist = 2. * dist_half
        axis_d = F.normalize(axis_d, p=2., dim=0)

        motion = {
            'type': 'translate',
            'axis_o': axis_o,
            'axis_d': axis_d,
            'dist': dist
        }
        return motion