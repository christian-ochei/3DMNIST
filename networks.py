import torch
import torch.nn as nn
import numpy as np
from tools import (_indices_from_shape,_ray_march)

class LowGradeDiscriminator(nn.Module):
    def __init__(self,shape):
        super(LowGradeDiscriminator, self).__init__()
        self.shape = shape
        self.size = np.prod(shape)
        self.layers = nn.Sequential(*[
            nn.Sequential(nn.Linear(self.size,self.size),nn.Sigmoid()) for _ in range(5)],
            nn.Linear(self.size,1),nn.Sigmoid()
        )

    def forward(self,inputs):
        """
        :param inputs:  Batch of torch Tensor [N,...] where prod of ... == self.size
        :return:
        """
        assert np.prod(inputs.shape[1:]) == self.size
        return self.layers(inputs.flatten(1))


class VoxelGrid(nn.Module):
    def __init__(self,shape):
        super(VoxelGrid, self).__init__()
        H, W, D = shape
        self.shape = torch.tensor(shape)
        self.scaled_indices = _indices_from_shape(shape)-self.shape/2
        self.absorbance  = nn.Parameter(torch.ones(H,W,D,1))
        self.attenuation = nn.Parameter(torch.logit(torch.empty(H,W,D,1).fill_(0.02)))


class DifferentiableRenderer(nn.Module):
    def __init__(self):
        super(DifferentiableRenderer, self).__init__()

    def forward(self,camera_R,volume:VoxelGrid):
        indices = ((volume.scaled_indices @ camera_R) + volume.shape/2 + (40-volume.shape)/2)

        absorbance  = torch.logit(torch.zeros(40,40,40,1))
        attenuation = torch.logit(torch.zeros(40,40,40,1))

        indices = torch.clip(indices,0,39).long()

        absorbance [indices[...,0],indices[...,1],indices[...,2]] = volume.absorbance
        attenuation[indices[...,0],indices[...,1],indices[...,2]] = volume.attenuation

        render = _ray_march(absorbance,attenuation)
        return render[None]
