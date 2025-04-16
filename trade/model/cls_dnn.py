import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import cloudpickle

from torch.utils.tensorboard import SummaryWriter
import datetime

from trade.model.zoo import Net

from trade.model.model import Model

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


class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction="none")(inputs, targets)
        p = torch.sigmoid(inputs)
        focal_loss = self.alpha * (1 - p) ** self.gamma * bce_loss
        return focal_loss.mean()


class ClsDNN(Model, nn.Module):

    def __init__(
        self, features, output_dim=3, device="cuda", weight=None, scheduler_step=20
    ):
        nn.Module.__init__(self)
        Model.__init__(self, features)
        self.model = Net(len(features), output_dim)
        self.output_dim = output_dim
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
        self.device = device
        self.loss_fn = nn.BCEWithLogitsLoss(weight=weight)
        # self.loss_fn = FocalBCEWithLogitsLoss(alpha=weight.to(self.device))
        print(f"loss weight {weight}")
        self.metrics = {
            # "auc": AUROC(task="multiclass", num_classes=output_dim, average=None).to(
            #     self.device
            # ),
            "precision": Precision(
                task="multiclass", num_classes=output_dim, average=None
            ).to(self.device),
            "recall": Recall(
                task="multiclass", num_classes=output_dim, average=None
            ).to(self.device),
            "f1": F1Score(task="multiclass", num_classes=output_dim, average=None).to(
                self.device
            ),
            "acc": Accuracy(task="multiclass", num_classes=output_dim, average=None).to(
                self.device
            ),
        }
        self.scheduler_step = scheduler_step

    def step(self, data, step_idx, is_train=False):
        x = data["x"]
        y_cls = data["y_cls"].reshape([-1]).to(torch.int64)
        y = torch.nn.functional.one_hot(y_cls, self.output_dim).to(torch.float32)
        self.optimizer.zero_grad()
        y_p = self.forward(x)
        loss = self.loss_fn(y_p, y)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0, norm_type=2)
            self.optimizer.step()
            if step_idx > self.scheduler_step and step_idx % self.scheduler_step == 0:
                self.scheduler.step(metrics=loss)
            rs = y_p.detach(), y, loss.detach()
        else:
            rs = y_p, y, loss

        m = {"loss": rs[-1]}
        
        for k, v in self.metrics.items():
            value = v(y_p.detach(), y_cls.detach())
            if value.numel() > 1:
                for i in range(value.numel()):
                    m[f"{k}_{i}"] = value[i]
            else:
                m[k] = value

        # if not is_train:
        #     # y_p, y, loss, metric = self.step(data, step_idx)
        y_cls = data["y_cls"].reshape([-1]).to(torch.int64)
        yp_cls = torch.argmax(y_p, dim=1)

        for i in range(self.output_dim):
            m[f"y_{i}"] = torch.sum(y_cls == i) * 1.0 / y_cls.shape[0]
            m[f"yp_{i}"] = torch.sum(yp_cls == i) * 1.0 / yp_cls.shape[0]
        return y_p, y, loss, m

        # return *rs, m

    def reset_metrics(self):
        for v in self.metrics.values():
            v.reset()

    def train_step(self, data, step_idx):
        return self.step(data, step_idx, True)

    def valid_step(self, data, step_idx):
        return self.step(data, step_idx)

    def predict_step(self, data, step_idx):
        return self.step(data, step_idx)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
  
    def to_device(self, v):
        return torch.asarray(v, self.device)