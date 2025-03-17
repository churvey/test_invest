import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import cloudpickle

from torch.utils.tensorboard import SummaryWriter
import datetime

from trade.data.loader import Dataloader
from trade.data.sampler import *
from trade.model.reg_dnn import RegDNN
from trade.model.cls_dnn import ClsDNN
from trade.train.utils import *

import torchmetrics
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
import numpy as np
import torch
import ray


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

    def run_phase(self, phase, epoch_idx=0):
        sampler = self.samplers[phase]
        batch_iter = sampler.iter(self.batch_size, phase)

        # w = torch.asarray(sampler.w).to(self.device)

        desc = f"{phase}-epoch-{epoch_idx}"
        batch_iter.desc = desc
        last_metrics = {}

        is_async = phase != "predict"
        # is_async = False
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

            for name in self.models:
                self.run_streams[name].synchronize()
                last_metric = last_metrics[name]
                for k, v in last_metric.items():
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
                        k: v.to(self.device, non_blocking=True)
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
                    k: torch.asarray(v, dtype=torch.float32, device=self.device)
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

    def run(self, start=-1, epoch=3, save_name="model", save_last=True):
        def save(i):
            saved = from_cache(f"models.pkl")
            if not saved:
                saved = {}
            if save_name in saved:
                saved[save_name].append({"models": self.models, "epoch_idx": i})
            else:
                saved.update({save_name: [{"models": self.models, "epoch_idx": i}]})
            save_cache(f"models.pkl", saved)

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
            self.run_phase("predict")


def get_samplers_cpp(label_gen, date_ranges, csi=None):
    loader = Dataloader(os.path.expanduser("~/output/qlib_bin"), [label_gen])
    return {k: SamplersCpp(loader, v, csi) for k, v in date_ranges.items()}


if __name__ == "__main__":

    with Context() as ctx:

        def label_gen(data):
            l = data["close"].shape[0]
            return {
                "pred": np.concatenate(
                    [data["close"][2:] / data["close"][1:-1] - 1, [float("nan")] * 2]
                )[:l],
                "cls": np.concatenate([data["limit_flag"][1:], [float("nan")]])[:l],
            }

        stages = ["train", "valid", "predict"]

        date_ranges = [
            ("2008-01-01", "2023-12-31"),
            ("2024-01-01", "2025-12-31"),
            ("2024-01-01", "2025-12-31"),
        ]

        # date_ranges = [
        #     ("2008-01-01", "2014-12-31"),
        #     ("2015-01-01", "2016-12-31"),
        #     ("2017-01-01", "2020-12-31"),
        # ]

        
        samplers = get_samplers_cpp(label_gen, dict(zip(stages, date_ranges)))
        saved_models = from_cache(f"models.pkl")
        # for save_name in ["cls", "reg"]:
        for save_name in ["reg", "cls"]:
            for k in samplers.keys():
                samplers[k].use_label_weight = save_name == "cls"
                # print(f"use_label_weight {samplers[k].use_label_weight}")
            # schedule = [8, 16, 64]
            schedule = [128, 256, 512]
            if saved_models:
                models = saved_models[save_name][-1]["models"]
                epoch_idx = saved_models[save_name][-1]["epoch_idx"]
            else:
                models = {}

                model_class = RegDNN if save_name == "reg" else ClsDNN

                for i in schedule:
                    model_name = f"s_{i}_{save_name}"
                    models[model_name] = model_class(
                        len(samplers[stages[0]].feature_columns()),
                        scheduler_step=i,
                    )

                epoch_idx = -1

            trainer = Trainer(8092 * 4, samplers, models)

            trainer.run(epoch_idx, 50, save_name)
