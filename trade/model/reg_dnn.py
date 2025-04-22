import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import cloudpickle

from torch.utils.tensorboard import SummaryWriter
import datetime

from trade.model.zoo import Net

from trade.model.model import Model, mse, TweedieLoss



from torchmetrics import (
    AUROC,
    Accuracy,
    PrecisionRecallCurve,
    F1Score,
    Precision,
    Recall,
)

from torchmetrics.regression import (
    PearsonCorrCoef,
    SpearmanCorrCoef,
    R2Score,
    MeanAbsoluteError,
    MeanAbsolutePercentageError,
    MeanSquaredError,
)


class RegDNN(Model, nn.Module):

    def __init__(self, features, output_dim=1, device="cuda", scheduler_step=20):
        nn.Module.__init__(self)
        Model.__init__(self, features)
        self.model = Net(len(self.features), output_dim, layers=(512, 256))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.002, weight_decay=0.0002
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=10,
            verbose=True,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0.00001,
            eps=1e-08,
        )
        self.loss_fn = mse
        # self.loss_fn = nn.L1Loss()
        # self.loss_fn = TweedieLoss()
        # self.loss_fn = TweedieLoss()
        self.device = device
        self.metrics = {
            "pcorr": PearsonCorrCoef().to(self.device),
            "mae": MeanAbsoluteError().to(self.device),
            "mse": MeanSquaredError().to(self.device),
            "r2": R2Score().to(self.device),
        }
        self.scheduler_step = scheduler_step

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def step(self, data, step_idx, is_train=False):
        x = data["x"]
        y = data["y_pred"]
        self.optimizer.zero_grad()
        y_p = self.forward(x)
        loss = self.loss_fn(y_p, y)
        loss_w = None
        if is_train:
            if "w" in data:
                loss_w = self.loss_fn(y_p, y, data["w"])
            else:
                loss_w = loss
            loss_w.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0, norm_type=2
            )
            self.optimizer.step()
            if step_idx > self.scheduler_step and step_idx % self.scheduler_step == 0:
                self.scheduler.step(metrics=loss_w)
            rs = y_p.detach(), y, loss.detach()
        else:
            rs = y_p, y, loss

        m = {
            "loss": rs[-1],
        }
        if is_train and "w" in data:
            m["loss_w"] = loss_w.detach()
            m["w"] = data["w"].mean().detach()
            m["w_max"] = data["w"].max().detach()
            m["w_min"] = data["w"].min().detach()
        for k, v in self.metrics.items():
            m[k] = v(rs[0], rs[1])
            m["y_abs"] = y.abs().mean().detach()
            m["yp_abs"] = y_p.abs().mean().detach()
            m["y"] = y.mean().detach()
            m["yp"] = y_p.mean().detach()
            m["y_var"] = y.std().detach()
            m["yp_var"] = y_p.std().detach()
            # m["y_q85"] = torch.quantile(y.abs(), 0.85).detach()
            # m["yp_q85"] = torch.quantile(y_p.abs(), 0.85).detach()

        return *rs, m

    def reset_metrics(self):
        for v in self.metrics.values():
            v.reset()

    def train_step(self, data, step_idx):
        return self.step(data, step_idx, True)

    def valid_step(self, data, step_idx):
        return self.step(data, step_idx)

    def predict_step(self, data, step_idx):
        return self.step(data, step_idx)

    def to_device(self, v):
        return torch.asarray(v).to(self.device)
