import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import cloudpickle

from torch.utils.tensorboard import SummaryWriter
import datetime

from catboost import CatBoostRegressor, Pool

from trade.model.zoo import Net

def mse(pred, target, weight = 1.0):
    # if not weight:
    sqr_loss = torch.mul(pred - target, pred - target)
    loss = torch.mul(sqr_loss, weight).mean()
    return loss


class Model:

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


class CatModel(Model):
    def __init__(self):
        self.model = CatBoostRegressor(
                        iterations=2,
                          depth=2,
                          learning_rate=1,
                          eval_metric='Logloss',
                          loss_function='RMSE')

    def train_step(self, data, step):
        x = data["x"]
        y = data["y_pred"]
        train_pool = Pool(x, y)
        self.model.fit(train_pool)
        return {}

    def valid_step(self, data, step):
        pass

    def predict_step(self, data, step):
        x = data["x"]
        test_pool = Pool(x)
        preds = self.model.predict(test_pool)
        return preds