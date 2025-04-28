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
    # print("shapes", pred.shape, target.shape)
   
    sqr_loss = torch.mul(pred - target, pred - target)
    
    loss = torch.mul(sqr_loss, weight).mean()
    return loss

def quantile_loss(y_pred, y_true, quantile=0.9):
    error = y_true - y_pred
    loss = torch.mean(torch.max(quantile * error, (quantile - 1) * error))
    return loss

class HuberLoss(nn.Module):
    def __init__(self, delta=0.01):
       
        super(HuberLoss, self).__init__()
        self.delta = delta

    def forward(self, y_pred, y_true):
      
        error = y_true - y_pred
        abs_error = torch.abs(error)
        
        # 计算二次损失和线性损失的权重
        quadratic = torch.clamp(abs_error, max=self.delta)
        linear = (abs_error - quadratic)
        
        # 组合损失
        loss = 0.5 * quadratic ** 2 + self.delta * linear
        return torch.mean(loss)  # 返回批次平均损失


class TweedieLoss(nn.Module):
    def __init__(self, p=1.5, eps=1e-8):
        super().__init__()
        self.p = p  # 幂参数 (1 < p < 2)
        self.eps = eps  # 数值稳定性

    def forward(self, y_pred, y_true):
        # 确保预测值为正（Tweedie要求）
        y_pred = torch.exp(y_pred)
        y_pred = torch.clamp(y_pred, min=self.eps)
        
        # 计算损失
        term1 = -torch.exp(y_true) * torch.pow(y_pred, 1 - self.p) / (1 - self.p)
        term2 = torch.pow(y_pred, 2 - self.p) / (2 - self.p)
        loss = term1 + term2
        return torch.mean(loss)
    
# def tweedie_loss(logit, label, rho=1.5):
#     # logit = torch.exp(logit)
#     from pytorch_forecasting.metrics.point import TweedieLoss
#     return TweedieLoss(p=rho).loss(logit, label)

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