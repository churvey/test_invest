import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import cloudpickle

from torch.utils.tensorboard import SummaryWriter
import datetime

from trade.data.loader import QlibDataloader, FtDataloader
from trade.data.sampler import *
from trade.model.reg_dnn import RegDNN, RegTransformer ,RegLSTM
from trade.model.cls_dnn import ClsDNN
from trade.train.utils import *
import numpy as np
import torch


def get_writer(date=None):
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M")
    if not date:
        folder = f"./metric/{now}"
    else:
        folder = f"./metric/{date}"
        if os.path.exists(folder):
            folder = f"{folder}-{now}"
    folder = sub_dir(folder)
    writer = SummaryWriter(folder)
    return writer


class Trainer:
    def __init__(self, batch_size, samplers={}, models=[], metrics={}, device="cuda"):
        self.batch_size = batch_size
        self.samplers = samplers
        self.models = models
        self.device = device
        self.models = {k: m.cuda() for k, m in self.models.items()}
        self.writers = {k: get_writer(k) for k in self.models.keys()}
        self.metrics = metrics
        self.streams = [torch.cuda.Stream() for _ in range(2)]
        self.run_streams = {k: torch.cuda.Stream() for k in self.models.keys()}

    def metric_name(self, name, phase, per_step=True):
        base = f"{name}/{phase}"
        if per_step:
            return f"{base}-per_step"
        return base

    def run_phase(self, phase, epoch_idx=0, save_name = None):
        sampler = self.samplers[phase]
        if isinstance(sampler, list):
            sampler = sampler[epoch_idx]
        batch_iter = sampler.iter(self.batch_size, phase)

        save_pd = []

        desc = f"{phase}-epoch-{epoch_idx}"
        batch_iter.desc = desc
        last_metrics = {}

        is_async = phase != "predict"
        save_pred = phase == "predict"
        streams = [torch.cuda.Stream() for _ in range(2)]
        data_cache = [None for i in range(2)]

        def run(data, i):
            step = epoch_idx * batch_iter.total + i
            for name, m in self.models.items():
                with torch.cuda.stream(self.run_streams[name]):
                    method = getattr(m, f"{phase}_step")
                    y_p, y, loss, last_metric = method(data, i)
                    # last_metric["loss"] = loss
                    last_metric = {
                        k: v.detach().to("cpu", non_blocking=True)
                        for k, v in last_metric.items()
                    }
                    last_metrics[name] = last_metric
                    if save_pred:
                        import pdb
                        # pdb.set_trace()
                        save_pd.append(
                            pd.DataFrame.from_dict({
                                "instrument":data["instrument"].reshape([-1]),
                                "datetime":data["datetime"].reshape([-1]),
                                "y":y.detach().cpu().numpy().reshape([-1]),
                                "y_p":y_p.detach().cpu().numpy().reshape([-1]),
                                }
                            ).dropna()
                        )
                        

            for name in self.models:
                self.run_streams[name].synchronize()
                last_metric = last_metrics[name]
                for k, v in last_metric.items():
                    # if step >= 100:
                    self.writers[name].add_scalar(
                        self.metric_name(k, phase),
                        v,
                        step,
                    )

        i = 0
        for i, data_i in enumerate(batch_iter):
            if is_async:
                with torch.cuda.stream(self.streams[(i) % len(streams)]):
                    batch = {
                        k: v.to(self.device, non_blocking=True) if k not in ["datetime", "instrument"] else v
                        for k, v in data_i.items()
                    }
                    # if "indices" in batch:
                    #     batch["w"] = w[batch["indices"]]
                    data_cache[(i) % len(data_cache)] = batch
                if i == 0:
                    continue
                self.streams[(i - 1) % len(self.streams)].synchronize()
                data = data_cache[(i - 1) % len(data_cache)]
                run(data, i - 1)
            else:
                data = {
                    k: torch.asarray(v, dtype=torch.float32, device=self.device) if k not in ["datetime", "instrument"] else v
                    for k, v in data_i.items()
                }
                run(data, i)

        if is_async:
            self.streams[(i) % len(self.streams)].synchronize()
            data = data_cache[(i) % len(data_cache)]
            run(data, i)
            # print(f"last i:{i}")

        for name, d in last_metrics.items():
            for k, v in d.items():
                self.writers[name].add_scalar(
                    self.metric_name(k, phase, False), v, epoch_idx
                )
                print(f"{i} {phase} {k} ==> {v}")
        if save_pred:
            saved = from_cache(f"{save_name}/predict.pkl")
            if saved is not None:
                save_pd.append(saved)
            save_pd = pd.concat(save_pd)
            print("save result", len(save_pd))
            print(save_pd)
            save_cache(f"{save_name}/predict.pkl", save_pd)

    def run(self, start=-1, epoch=3, save_name=None, save_last=True):
        def save(i):
            saved = from_cache(f"{save_name}/models.pkl")
            if not saved:
                saved = []
            saved.append({"models": self.models, "epoch_idx": i})
            save_cache(f"{save_name}/models.pkl", saved)

        for i in range(start + 1, epoch):
            for m in self.models.values():
                m.train()
                m.reset_metrics()
            self.run_phase("train", i)
            if "valid" in self.samplers:
                for m in self.models.values():
                    m.eval()
                    m.reset_metrics()
                self.run_phase("valid", i)
            if not save_last or i == epoch - 1:
                save(i)

        if "predict" in self.samplers:
            for m in self.models.values():
                m.eval()
                m.reset_metrics()
            self.run_phase("predict", save_name=save_name)


def get_samplers_cpp(label_gen, date_ranges, csi=None, seq_col = "instrument"):
    loader = QlibDataloader(os.path.expanduser("~/output/qlib_bin"), [label_gen], csi)
    # loader = QlibDataloader(os.path.expanduser("~/output/qlib_bin"), [label_gen], "csi300")
    # loader = FtDataloader("tmp", [label_gen])
    return {k: SamplersCpp(loader, v, seq_col) for k, v in date_ranges.items()}


def get_label(data):
    value = data["z_pos_20"]
    v_8 = value == 8
    v_32 = (value >= 32).astype("int64")
    v_2 = (value <= 2).astype("int64")
    diff = v_32 - v_2

    for k in data.keys():
        data[k] = data[k][v_8]

    rs = np.zeros(value.shape)
    for i in range(len(rs)):
        if v_8[i]:
            for j in range(i + 1, len(rs)):
                if diff[j] != 0:
                    rs[i] = diff[j]
                    break
    rs = rs[v_8] + 1
    # print(np.unique(rs))
    return rs


if __name__ == "__main__":



    def label_gen(data):
        l = data["close"].shape[0]
        pred = np.concatenate(
                [
                    (data["open"][1:] / data["close"][:-1] - 1) * 100,
                    [float("nan")] * 1,
                ]
        )[:l]
        
        # valid = (np.abs(pred) <= 0.098) & (np.abs(data["change"]) < 0.098)
        valid = (np.abs(pred) <= 0.098 * 100)
        pred = pred[valid]
        
        for k in data.keys():
            data[k] = data[k][valid]            
        
        return {
            # "pred": np.concatenate(
            #     [np.log(data["open"][1:] / data["close"][:-1]), [float("nan")] * 1]
            # )[:l],
            "pred": pred,
            # "pred": np.concatenate(
            #     [np.log(data["open"] / data["close"]), []]
            # )[:l],
            # "pred": np.concatenate(
            #     [np.abs(np.log(data["open"][2:] / data["open"][1:-1])), [float("nan")] * 2]
            # )[:l],
            # "cls": np.concatenate([data["limit_flag"][1:], [float("nan")]])[:l],
            #  "cls": get_label(data),
        }

    stages = ["train", "valid", "predict"]

    use_roller = False
    epoch = 20
    if not use_roller:
        date_ranges = [
            ("2008-01-01", "2023-12-31"),
            ("2024-01-01", "2025-12-31"),
            # ("2008-01-01", "2023-12-31"),
            ("2024-01-01", "2025-12-31"),
        ]
        date_ranges = [date_ranges]
    else:
        date_ranges = [
            ("2012-01-01", "2023-12-31"),
            ("2024-01-01", "2024-01-31"),
            ("2024-01-01", "2024-01-31"),
        ]
        def get(date_range, i):
            
            from datetime import datetime
            from dateutil.relativedelta import relativedelta  # 需要安装

            def add_month_safe(date_str, input_format="%Y-%m-%d"):
                # 解析字符串为日期对象
                date = datetime.strptime(date_str, input_format)
                
                # 直接加一个月（自动处理月末）
                new_date = date + relativedelta(months=i)
                return new_date.strftime(input_format)
            b, e = date_range
            return add_month_safe(b), add_month_safe(e) 

        date_ranges = [[
            get(date_ranges[j], i) for j in range(len(date_ranges))
        ]  for i in range(epoch) ]
    print(date_ranges)
    for data_i in range(len(date_ranges)):
        for model_class in [ RegLSTM , RegDNN, RegTransformer]:
        # for model_class in [ RegLSTM]:
            save_name = str(model_class.__name__.split(".")[-1])
            with Context() as ctx:
                saved_models = from_cache(f"{save_name}/models.pkl")
                seq_col = "instrument"
                if "Transformer" in save_name:
                    seq_col = "instrument" 
                if "LSTM" in save_name:
                    seq_col = "datetime"
                
                samplers = get_samplers_cpp(label_gen, dict(zip(stages, date_ranges[data_i])), seq_col=seq_col)
                # for k in samplers.keys():
                #     samplers[k].use_label_weight = save_name == "cls"
                    # print(f"use_label_weight {samplers[k].use_label_weight}")
                schedule = [32]
                # schedule = [128, 256, 512]
                # schedule = [512]
                if saved_models:
                    models = saved_models[-1]["models"]
                    epoch_idx = saved_models[-1]["epoch_idx"]
                else:
                    models = {}

                    for i in schedule:
                        model_name = f"s2_{i}_{save_name}"
                        models[model_name] = model_class(
                            samplers[stages[0]].feature_columns(),
                            scheduler_step=i,
                        )

                    epoch_idx = -1

                batch_size = 64
                # if not seq_col:
                #     batch_size *= 384
                # trainer = Trainer(8092 * 4, samplers, models)
                trainer = Trainer(batch_size, samplers, models)
                # print(epoch_idx, i + 1)
                trainer.run(epoch_idx, data_i + 1 if use_roller else epoch, save_name)
