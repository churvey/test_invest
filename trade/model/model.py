import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import cloudpickle

from torch.utils.tensorboard import SummaryWriter
import datetime

from trade.model.zoo import Net

def mse(pred, target, weight = 1.0):
    # if not weight:
    sqr_loss = torch.mul(pred - target, pred - target)
    loss = torch.mul(sqr_loss, weight).mean()
    return loss


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()

    def train_step(self, data, step):
        pass

    def valid_step(self, data, step):
        pass

    def predict_step(self, data, step):
        pass

    def forward(*args, **kvargs):
        pass


