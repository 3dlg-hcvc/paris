import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.rotation import R_from_quaternions

class WeightedLoss(nn.Module):
    @property
    def func(self):
        raise NotImplementedError

    def forward(self, inputs, targets, weight=None, reduction='mean'):
        assert reduction in ['none', 'sum', 'mean', 'valid_mean']
        loss = self.func(inputs, targets, reduction='none')
        if weight is not None:
            while weight.ndim < inputs.ndim:
                weight = weight[..., None]
            loss *= weight.float()
        if reduction == 'none':
            return loss
        elif reduction == 'sum':
            return loss.sum()
        elif reduction == 'mean':
            return loss.mean()
        elif reduction == 'valid_mean':
            return loss.sum() / weight.float().sum()


class MSELoss(WeightedLoss):
    @property
    def func(self):
        return F.mse_loss


class L1Loss(WeightedLoss):
    @property
    def func(self):
        return F.l1_loss


class PSNR(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, inputs, targets, valid_mask=None, reduction='mean'):
        assert reduction in ['mean', 'none']
        value = (inputs - targets)**2
        if valid_mask is not None:
            value = value[valid_mask]
        if reduction == 'mean':
            return -10 * torch.log10(torch.mean(value))
        elif reduction == 'none':
            return -10 * torch.log10(torch.mean(value, dim=tuple(range(value.ndim)[1:])))


class SSIM():
    def __init__(self, data_range=(0, 1), kernel_size=(5, 5), sigma=(1.5, 1.5), k1=0.01, k2=0.03, gaussian=True):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.gaussian = gaussian
        
        if any(x % 2 == 0 or x <= 0 for x in self.kernel_size):
            raise ValueError(f"Expected kernel_size to have odd positive number. Got {kernel_size}.")
        if any(y <= 0 for y in self.sigma):
            raise ValueError(f"Expected sigma to have positive number. Got {sigma}.")
        
        data_scale = data_range[1] - data_range[0]
        self.c1 = (k1 * data_scale)**2
        self.c2 = (k2 * data_scale)**2
        self.pad_h = (self.kernel_size[0] - 1) // 2
        self.pad_w = (self.kernel_size[1] - 1) // 2
        self._kernel = self._gaussian_or_uniform_kernel(kernel_size=self.kernel_size, sigma=self.sigma)
    
    def _uniform(self, kernel_size):
        max, min = 2.5, -2.5
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        for i, j in enumerate(kernel):
            if min <= j <= max:
                kernel[i] = 1 / (max - min)
            else:
                kernel[i] = 0

        return kernel.unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian(self, kernel_size, sigma):
        ksize_half = (kernel_size - 1) * 0.5
        kernel = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
        gauss = torch.exp(-0.5 * (kernel / sigma).pow(2))
        return (gauss / gauss.sum()).unsqueeze(dim=0)  # (1, kernel_size)

    def _gaussian_or_uniform_kernel(self, kernel_size, sigma):
        if self.gaussian:
            kernel_x = self._gaussian(kernel_size[0], sigma[0])
            kernel_y = self._gaussian(kernel_size[1], sigma[1])
        else:
            kernel_x = self._uniform(kernel_size[0])
            kernel_y = self._uniform(kernel_size[1])

        return torch.matmul(kernel_x.t(), kernel_y)  # (kernel_size, 1) * (1, kernel_size)

    def __call__(self, output, target, reduction='mean'):
        if output.dtype != target.dtype:
            raise TypeError(
                f"Expected output and target to have the same data type. Got output: {output.dtype} and y: {target.dtype}."
            )

        if output.shape != target.shape:
            raise ValueError(
                f"Expected output and target to have the same shape. Got output: {output.shape} and y: {target.shape}."
            )

        if len(output.shape) != 4 or len(target.shape) != 4:
            raise ValueError(
                f"Expected output and target to have BxCxHxW shape. Got output: {output.shape} and y: {target.shape}."
            )

        assert reduction in ['mean', 'sum', 'none']

        channel = output.size(1)
        if len(self._kernel.shape) < 4:
            self._kernel = self._kernel.expand(channel, 1, -1, -1)

        output = F.pad(output, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")
        target = F.pad(target, [self.pad_w, self.pad_w, self.pad_h, self.pad_h], mode="reflect")

        input_list = torch.cat([output, target, output * output, target * target, output * target])
        outputs = F.conv2d(input_list, self._kernel, groups=channel)

        output_list = [outputs[x * output.size(0) : (x + 1) * output.size(0)] for x in range(len(outputs))]

        mu_pred_sq = output_list[0].pow(2)
        mu_target_sq = output_list[1].pow(2)
        mu_pred_target = output_list[0] * output_list[1]

        sigma_pred_sq = output_list[2] - mu_pred_sq
        sigma_target_sq = output_list[3] - mu_target_sq
        sigma_pred_target = output_list[4] - mu_pred_target

        a1 = 2 * mu_pred_target + self.c1
        a2 = 2 * sigma_pred_target + self.c2
        b1 = mu_pred_sq + mu_target_sq + self.c1
        b2 = sigma_pred_sq + sigma_target_sq + self.c2

        ssim_idx = (a1 * a2) / (b1 * b2)
        _ssim = torch.mean(ssim_idx, (1, 2, 3))

        if reduction == 'none':
            return _ssim
        elif reduction == 'sum':
            return _ssim.sum()
        elif reduction == 'mean':
            return _ssim.mean()


def binary_cross_entropy(src, target, clip_eps=1e-6):
    src = torch.clamp(src, clip_eps, 1.-clip_eps)
    return -(target * torch.log(src) + (1 - target) * torch.log(1 - src)).mean()

def entropy_loss(src, clip_eps=1e-6, skew=1.0):
    """
    "skew" is used to control the skew of entropy loss.
    """
    src = torch.clip(src ** skew, clip_eps, 1-clip_eps)
    entropy = - (src * torch.log(src) + (1.-src) * torch.log(1.-src))
    return entropy.mean()

# def rotation_diff_quat(p, q):
#     p, q = p.cpu(), q.cpu()
#     conj_q = torch.tensor([q[0], -q[1], -q[2], -q[3]]).float()
#     cos_half_angle = torch.clip(torch.dot(p, conj_q), min=-1., max=1.)
#     angle = 2. * torch.arccos(cos_half_angle)
#     deg = torch.rad2deg(angle)
#     return deg

def geodesic_distance(q, gt_R):
    pred_R = R_from_quaternions(q.squeeze(0)).cpu()
    R = torch.matmul(pred_R, gt_R.T)
    cos_angle = torch.clip((torch.trace(R) - 1.0) * 0.5, min=-1., max=1.)
    angle = torch.rad2deg(torch.arccos(cos_angle)) 
    return angle

# def geodesic_distance_R(pred_R, gt_R):
#     pred_R, gt_R = pred_R.cpu(), gt_R.cpu()
#     R = torch.matmul(pred_R, gt_R.T)
#     cos_angle = torch.clip((torch.trace(R) - 1.0) * 0.5, min=-1., max=1.)
#     angle = torch.rad2deg(torch.arccos(cos_angle)) 
#     return angle

def axis_metrics(motion_info, gt_axis):
    # pred axis
    pred_axis_d = motion_info['axis_d'].cpu().squeeze(0)
    pred_axis_o = motion_info['axis_o'].cpu().squeeze(0)
    # gt axis
    gt_axis_d = gt_axis[1] - gt_axis[0]
    gt_axis_o = gt_axis[0]
    # angular difference between two vectors
    cos_theta = torch.dot(pred_axis_d, gt_axis_d) / (torch.norm(pred_axis_d) * torch.norm(gt_axis_d))
    ang_err = torch.rad2deg(torch.acos(torch.abs(cos_theta)))
    # positonal difference between two axis lines
    w = gt_axis_o - pred_axis_o
    cross = torch.cross(pred_axis_d, gt_axis_d)
    if (cross == torch.zeros(3)).sum().item() == 3:
        pos_err = torch.tensor(0)
    else:
        pos_err = torch.abs(torch.sum(w * cross)) / torch.norm(cross)
    return ang_err, pos_err

def translational_error(motion, gt):
    dist = motion['dist'].cpu()
    axis_d = F.normalize(motion['axis_d'].cpu().squeeze(0), p=2, dim=0)
    gt_dist = gt['dist']
    gt_axis_d = F.normalize(gt['axis_d'].cpu(), p=2, dim=0)

    err = torch.sqrt(((dist * axis_d - gt_dist * gt_axis_d) ** 2).sum())
    return err