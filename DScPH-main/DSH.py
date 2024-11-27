
import os
import torch, torch.nn as nn, torch.nn.functional as F
import torch.optim as optim
import time
import numpy as np

class DSHLoss(torch.nn.Module):
    def __init__(self, bit, device):
        super(DSHLoss, self).__init__()
        self.device = device
        self.m = 2 * bit
        # self.U = torch.zeros(config["num_train"], bit).float().to(config["device"])
        # self.Y = torch.zeros(config["num_train"], config["n_class"]).float().to(config["device"])

    def forward(self, u, y,feat2=None ):
        # self.U[ind, :] = u.data
        # self.Y[ind, :] = y.float()
        # .to("cuda:1")
        y = y.float()
        if feat2 is not None:
            dist = ((u.unsqueeze(1) - feat2.unsqueeze(0)).pow(2).sum(dim=2))
        else:
            dist = ((u.unsqueeze(1) - u.unsqueeze(0)).pow(2).sum(dim=2))
        y = (y @ y.t() == 0).float()
        y = y.to(0)
        dist = dist.to(0)
        loss = (1 - y) / 2 * dist + y / 2 * (self.m - dist).clamp(min=0)
        loss1 = loss.mean()

        return loss1

