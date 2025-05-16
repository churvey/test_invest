import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import cloudpickle
from numpy.lib.stride_tricks import sliding_window_view

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
        
        self.embedding_dim = 64
        self.embedding = nn.Embedding(num_embeddings=10000, embedding_dim=self.embedding_dim)
        
        self.model = Net(len(self.features) + self.embedding_dim, output_dim, layers=(512, 256))
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
            # "scorr":SpearmanCorrCoef().to(self.device),
            "mae": MeanAbsoluteError().to(self.device),
            "mse": MeanSquaredError().to(self.device),
            "r2": R2Score().to(self.device),
        }
        self.scheduler_step = scheduler_step

    def forward(self, *args, **kwargs):
        x = args[0]
        x_cat = kwargs["x_cat"]
        # print("forward", x.shape, x_cat.shape, x_cat)
        em = self.embedding(x_cat)
        x = x.reshape([len(x), em.shape[1], -1])
        
        v = torch.concatenate(
            [x, em],dim=-1
        ).reshape([len(x), -1])
        return self.model.forward(v)

    def step(self, data, step_idx, is_train=False):
        x = data["x"]
        y = data["y_pred"]
        # print(x.shape, y.shape)
        # 1/0
        if is_train:
            self.optimizer.zero_grad()
        y_p = self.forward(x, **data)

        # print("in step")
        # shapes = {k: v.shape for k, v in data.items()}
        # print(shapes)
        # print(y_p.shape)

        no_nan = ~torch.isnan(y).reshape([-1, 1])
        y_no_nan = y.reshape([-1, 1])[no_nan]
        y_p_no_nan = y_p.reshape([-1, 1])[no_nan]

        # print(y_no_nan.shape, "vs", y.shape)

        loss = self.loss_fn(y_no_nan, y_p_no_nan)
        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0, norm_type=2
            )
            self.optimizer.step()
            if step_idx > self.scheduler_step and step_idx % self.scheduler_step == 0:
                self.scheduler.step(metrics=loss)
            rs = y_p.detach(), y, loss.detach()
        else:
            rs = y_p, y, loss

        m = {
            "loss": rs[-1],
        }
        for k, v in self.metrics.items():
            m[k] = v(y_no_nan, y_p_no_nan)
        m["y_abs"] = y_no_nan.abs().mean().detach()
        m["yp_abs"] = y_p_no_nan.abs().mean().detach()
        m["y"] = y_no_nan.mean().detach()
        m["yp"] = y_p_no_nan.mean().detach()
        # m["y_var"] = y.std().detach()
        # m["yp_var"] = y_p.std().detach()
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


class Transformer(nn.Module):
    def __init__(
        self,
        d_feat=128,
        d_model=64,
        nhead=4,
        dim_feedforward=256,
        num_layers=2,
        dropout=0.5,
        device=None,
    ):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
        self.d_model = d_model
        self.net = Net(d_feat, d_model, layers=(512,))
        # self.pos_encoder = PositionalEncoding(d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )
        
        self.decoder_layer = nn.Linear( 2 * d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, x, mask=None):
        # print(src.shape, "shapes")
        batch_size = len(x)
        src = x.reshape(
            len(x), -1, self.d_feat
        )  # [batch_size, instruct_count, feature_dim]
        # print(src.shape, "shapes2")
        src = self.feature_layer(src)
        
        # src = src.reshape(
        #     batch_size, -1, self.d_model
        # )  # [batch_size, instruct_count, feature_dim]

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        # src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src, src_key_padding_mask=mask
        )  # [60, 512, 8]
        
        output2 = self.net(x)
        output2 = output2.reshape([batch_size, -1, self.d_model])
        
        output = torch.concatenate([output.transpose(1, 0), output2], dim=-1)

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output)  # [512, 1]

        return output.squeeze(dim=-1)
        # return output[..., 0]


class RegTransformer(RegDNN):

    def __init__(self, features, output_dim=1, device="cuda", scheduler_step=20):
        # nn.Module.__init__(self)
        super(RegTransformer, self).__init__(
            features, output_dim, device, scheduler_step
        )
        self.model = Transformer(len(self.features), device=self.device)
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

    def forward(self, *args, **kwargs):
        mask = None
        y = kwargs["y_pred"].squeeze(dim=-1)
        mask = torch.isnan(y)
        return self.model.forward(*args, mask)



class LSTMModel(nn.Module):
    def __init__(
        self,
        d_feat=6,
        hidden_size=64,
        num_layers=2,
        dropout=0.0,
        device="cuda"
    ):
        super().__init__()
        self.device = device
        self.rnn = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc_out = nn.Linear(hidden_size, 1)
        self.d_feat = d_feat

    def forward(self, x, **kwargs):
        # x: [N, T * F]
        # x = x.reshape(
        #     len(x), -1, self.d_feat
        # )  # [batch_size, datetime_count, feature_dim]
        # assert not torch.isnan(x).any(), x
        out, _ = self.rnn(x)
        # assert not torch.isnan(out).any(), out
        y =  self.fc_out(out[:, -1, ...])
        # assert not torch.isnan(y).any(), y
        
        return y

class RegLSTM(RegDNN):

    def __init__(self, features, output_dim=1, device="cuda", scheduler_step=20):
        # nn.Module.__init__(self)
        super(RegLSTM, self).__init__(features, output_dim, device, scheduler_step)
        self.model = LSTMModel(len(self.features), device=self.device)
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

    def step(self, data, step_idx, is_train=False):
        # {'datetime': (65536, 1), 'instrument': (65536, 1), 'x': torch.Size([256, 38144]), 'y_pred': torch.Size([256, 256])}
        new_data = {
            k:v.reshape([len(v), -1, 1])[:,-1,:] if k != "x" else v.reshape([len(v), -1, len(self.features)]) for k,v in data.items()
        }
        for k in new_data.keys():
            data[k] = new_data[k]
            if k == "x":
                data[k] = torch.nan_to_num(data[k])
                assert not torch.isnan(data[k]).any(), data[k]
        # new_data = {
        #     k: v[valid] if k == "x" else v[valid][:, -1, ...] for k, v in data.items()
        # }
        # for i in range(x.shape[1]):
        #     # print(i)
        #     new_data = {
        #         k: v[:, i, ...] if k == "x" else v[:, i, -1, ...] for k, v in data.items()
        #     }
        # shapes = {k: v.shape for k, v in data.items()}
        # print("step shapes", shapes)
        rs =  super().step(data, step_idx, is_train)
        return rs

    def forward(self, *args, **kwargs):
        return self.model.forward(**kwargs)
    
    
class AVG(RegDNN):

    def __init__(self, features, output_dim=1, device="cuda", scheduler_step=20):
        # nn.Module.__init__(self)
        super(AVG, self).__init__(features, output_dim, device, scheduler_step)
        self.values = None
   
    def step(self, data, step_idx, is_train=False):
        # {'datetime': (65536, 1), 'instrument': (65536, 1), 'x': torch.Size([256, 38144]), 'y_pred': torch.Size([256, 256])}
        
        # new_data = {
        #     k: v[valid] if k == "x" else v[valid][:, -1, ...] for k, v in data.items()
        # }
        # for i in range(x.shape[1]):
        #     # print(i)
        #     new_data = {
        #         k: v[:, i, ...] if k == "x" else v[:, i, -1, ...] for k, v in data.items()
        #     }
        # shapes = {k: v.shape for k, v in data.items()}
        # print("step shapes", shapes)
        rs =  super().step(data, step_idx, False)
        y = data["y_pred"]
        if self.values is None:
            self.values = y.reshape([-1])
        else:
            self.values = torch.cat([self.values, y.reshape([-1])])
        self.values = self.values[~torch.isnan(self.values)]
        self.values = self.values[:8096 * 16]
        return rs

    def forward(self, *args, **kwargs):
        # shape = kwargs["y_pred"].shape
        if self.values is not None:
            # return zero + torch.nanmean(self.values)
            return torch.normal(
                mean = self.values.mean(), std = self.values.std(),
                size = kwargs["y_pred"].shape,
                device = self.device
            )
        return torch.zeros_like(kwargs["y_pred"]) 
