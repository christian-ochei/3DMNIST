import torch
import torch.nn as nn
import numpy as np

def _indices_from_shape(shape):
    return torch.cat([x[...,None] for x in torch.meshgrid([torch.arange(length) for length in shape])],dim=-1)

def _ray_march(absorbance,attenuation):
    
    """
    This function converts volume to an image by compositing the volumes depth
    
    """
    
    attenuation = torch.sigmoid(attenuation)
    absorbance = torch.sigmoid(absorbance)
    contribution = torch.cat([
        torch.ones(attenuation.shape[0], attenuation.shape[1], 1, attenuation.shape[3]),
        torch.cumprod(1 - attenuation, dim=-2)[:, :, :-1, :]], dim=-2
    ) * attenuation

    values = contribution * (1-absorbance)
    render = values.sum(-2)
    return render
