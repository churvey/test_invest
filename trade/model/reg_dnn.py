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
        return self.model.forward(*args)

    def step(self, data, step_idx, is_train=False):
        x = data["x"]
        y = data["y_pred"]
        # print(x.shape, y.shape)
        # 1/0
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
        d_feat=6,
        d_model=8,
        nhead=4,
        dim_feedforward=256,
        num_layers=2,
        dropout=0.5,
        device=None,
    ):
        super(Transformer, self).__init__()
        self.feature_layer = nn.Linear(d_feat, d_model)
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
        self.decoder_layer = nn.Linear(d_model, 1)
        self.device = device
        self.d_feat = d_feat

    def forward(self, src, mask=None):
        # print(src.shape, "shapes")
        src = src.reshape(
            len(src), -1, self.d_feat
        )  # [batch_size, instruct_count, feature_dim]
        # print(src.shape, "shapes2")
        src = self.feature_layer(src)

        # src [N, T, F] --> [T, N, F], [60, 512, 8]
        src = src.transpose(1, 0)  # not batch first

        # src = self.pos_encoder(src)
        output = self.transformer_encoder(
            src, src_key_padding_mask=mask
        )  # [60, 512, 8]

        # [T, N, F] --> [N, T*F]
        output = self.decoder_layer(output.transpose(1, 0))  # [512, 1]

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
        device="cuda",
        max_seqlen=32,
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
        self.max_seqlen = max_seqlen
        self.d_feat = d_feat

    def forward(self, x, **kwargs):
        # x: [N, T * F]

        out, _ = self.rnn(x)
        return self.fc_out(out[:, -1, ...]).squeeze(dim=-1)


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

    # def step(self, data, step_idx, is_train=False):
    #     # print("x.shape", x.shape)
    #     # batch_size = x.shape[0]
    #     def unfold(x, dim):
    #         x = x.reshape(len(x), -1, dim)  # [N, T, F]
    #         # print("x.shape", x.shape)
    #         x = (
    #             torch_unfold(x, 1, self.max_seqlen, 1)
    #             if not isinstance(x, np.ndarray)
    #             else numpy_unfold(x, 1, self.max_seqlen, 1)
    #         )
    #         return x

    #     # x = unfold(x, self.len(self.features))
    #     for k in data.keys():
    #         # print(f"k:{k}")
    #         data[k] = unfold(data[k], 1 if k != "x" else len(self.features))

    #     shapes = {k: v.shape for k, v in data.items()}
    #     print("shapes", shapes)
    #     x = data["x"]
    #     valid = ~torch.any(torch.any(torch.isnan(x), dim=-1), dim=-1)
    #     # print("valid", valid.shape)
    #     print("valid count", torch.sum(valid), valid.shape)
    #     new_data = {
    #         k: v[valid] if k == "x" else v[valid][:, -1, ...] for k, v in data.items()
    #     }
    #     # for i in range(x.shape[1]):
    #     #     # print(i)
    #     #     new_data = {
    #     #         k: v[:, i, ...] if k == "x" else v[:, i, -1, ...] for k, v in data.items()
    #     #     }
    #     shapes = {k: v.shape for k, v in new_data.items()}
    #     print(shapes)
    #     rs =  super().step(new_data, step_idx, is_train)
    #     return rs

    def forward(self, *args, **kwargs):
        return self.model.forward(**kwargs)
