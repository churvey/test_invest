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
                        # iterations=200,
                        #   depth=20,
                        #   learning_rate=1,
                            loss_function='MAE',
                          eval_metric='MAE',
                          )

    def train_step(self, data, step):
        x = data["x"].numpy()
        y = data["y_pred"].numpy()
        print("x shape", x.shape)
        train_pool = Pool(x, y)
        self.model.fit(train_pool, plot=True)
       
        return {}

    def valid_step(self, data, step):
        x = data["x"].numpy()
        y = data["y_pred"].numpy()
        eval_pool = Pool(x, y)
        m = self.model.eval_metrics(
            eval_pool,
            metrics = ["R2",'RMSE', 'MAE']
        )
        print("eval", m)

    def predict_step(self, data, step):
        x = data["x"]
        y = data["y_pred"]
        # print("x shape", x.shape)
        test_pool = Pool(x)
        preds = self.model.predict(test_pool)
        # from sklearn.metrics import mean_absolute_error, mean_squared_error
        # print(f"MAE: {mean_absolute_error(y, preds)}")
        return preds