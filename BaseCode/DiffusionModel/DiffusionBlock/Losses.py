import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedLoss(nn.Module):
    def __init__(self):
        super(WeightedLoss, self).__init__()

    def forward(self, pred, targ, weighted=1.0):
        """
        :param pred:
        :param targ:
        """
        loss = self._loss(pred, targ)
        weightedLoss = (loss * weighted).mean()
        return weightedLoss


class WeightedL1(WeightedLoss):
    def _loss(self, pred, targ):
        return torch.abs(pred - targ)

class WeightedL2(WeightedLoss):
    def _loss(self, pred, targ):
        return F.mse_loss(pred, targ, reduction='none')