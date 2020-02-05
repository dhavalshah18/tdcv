import torch
from torch.nn import functional as F
import numpy as np


def total_loss(output):
    batch_size = output.size()[0]
    return (triplet_loss(output) + pair_loss(output))/(batch_size//3)


def triplet_loss(output):
    batch_size = output.size()[0]
    diff_pos = output[0::3,:] - output[1::3,:]
    diff_neg = output[0::3,:] - output[2::3,:]
    
    norm_pos = torch.norm(diff_pos.view(batch_size//3, -1), p=2, dim=1)**2
    norm_neg = torch.norm(diff_neg.view(batch_size//3, -1), p=2, dim=1)**2
    loss = F.relu(1 - (norm_neg / (norm_pos + 0.01)))
    
    return loss.sum()


def pair_loss(output):
    batch_size = output.size()[0]
    diff_pos = output[0::3,:] - output[1::3,:]
    loss = torch.norm(diff_pos.view(batch_size//3, -1), p=2, dim=1)**2
    
    return loss.sum()
