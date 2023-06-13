import torch
import systems
from systems.base import BaseSystem
from utils.rotation import quaternion_to_axis_angle, R_from_axis_angle


@systems.register('revolute-system')
class RevoluteSystem(BaseSystem):
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