"""Contains losses defined as per DeepVesselNet paper by Giles Tetteh"""

import torch
from torch.nn import functional as F
import numpy as np


def total_loss(anchor, pusher, puller, m=0.01):
    return triplet_loss(anchor, pusher, puller, m) + pair_loss(anchor, puller)


def triplet_loss(anchor, pusher, puller, m=0.01):
    loss = 0
    for i in range(anchor.shape[0]):
        numerator = torch.norm(anchor[i] - pusher[i])**2
        denominator = torch.norm(anchor[i] - puller[i])**2 + m
        max_check = [0, 1-(numerator/denominator)]
        max_return = max(max_check)
        loss += max_return

    return loss


def pair_loss(anchor, puller):
    loss = 0
    for i in range(anchor.shape[0]):
        loss += torch.norm(anchor[i] - puller[i])**2
    return loss
