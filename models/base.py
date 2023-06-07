import torch
import torch.nn as nn

from pytorch_lightning.utilities.rank_zero import _get_rank

class BaseModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rank = _get_rank()
        self.setup()
        if self.config.get('weights', None):
            self.load_state_dict(torch.load(self.config.weights))
            print('load model')
    
    def setup(self):
        raise NotImplementedError
    
    def update_step(self, epoch, global_step):
        pass
    
    def train(self, mode=True):
        return super().train(mode=mode)
    
    def eval(self):
        return super().eval()
    
    def regularizations(self, out):
        return {}
    
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
