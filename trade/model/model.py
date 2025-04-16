import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import cloudpickle

from torch.utils.tensorboard import SummaryWriter
import datetime

from catboost import CatBoostRegressor, Pool

from trade.data.feature.feature import Feature

from trade.model.zoo import Net

def mse(pred, target, weight = 1.0):
    # if not weight:
    sqr_loss = torch.mul(pred - target, pred - target)
    loss = torch.mul(sqr_loss, weight).mean()
    return loss


class Model:

    def __init__(self, features):
        self.features = features

    def train_step(self, data, step):
        pass

    def valid_step(self, data, step):
        pass

    def predict_step(self, data, step):
        pass

    def forward(self, *args, **kwargs):
        pass
    
    def to_device(self, data):
        return data
    
    def predict(self, data):
        data = Feature(data)()
        data = np.concatenate(
            [data[p].reshape([-1, 1])  for p in self.features], axis = -1
        ).astype("float32")
        # print(data.shape)
        data = self.to_device(data)
        rs = self.forward(data).cpu().detach().numpy()
        # print(rs)
        return rs