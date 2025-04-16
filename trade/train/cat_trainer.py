import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import cloudpickle

from torch.utils.tensorboard import SummaryWriter
import datetime

from trade.data.loader import QlibDataloader,FtDataloader
from catboost import CatBoostRegressor, Pool
from trade.data.sampler import *
from trade.model.reg_cat import RegCat
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


class CatTrainer:
    def __init__(self, batch_size, samplers={}, models=[], metrics={}, device="cuda"):
        self.batch_size = batch_size
        self.samplers = samplers
        self.models = models
        self.device = device
        # self.models = {k: m.cuda() for k, m in self.models.items()}
        # self.writers = {k: get_writer(k) for k in self.models.keys()}
        self.metrics = metrics
    
    def run_phase(self, phase, epoch_idx=0):
        sampler = self.samplers[phase]
        batch_iter = sampler.iter(self.batch_size, phase)
        for i, data_i in enumerate(batch_iter):
            for name, m in self.models.items():
                method = getattr(m, f"{phase}_step")
                method(data_i, i)


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
            
        def get_data(phase):
            if phase in self.samplers:
                _, data = next(enumerate(self.samplers[phase].iter(self.batch_size, phase)))
                x = data["x"].numpy()
                y = data["y_pred"].numpy()
                return Pool(x, y)
            return None

        if "train" in self.samplers:
            for i in range(start + 1, epoch):
                for name, m in self.models.items():
                    m.model.fit(
                        get_data("train"),
                        # eval_set = get_data("eval"),
                        # log_cout=TensorBoardLogger(f"./metric/{name}"),
                    )
                    importance = m.model.get_feature_importance().tolist()
                    feature_columns = self.samplers["train"].feature_columns()
                    importance = list(zip(importance, feature_columns))
                    importance= sorted(importance,
                        key = lambda x: -x[0]
                    )
                    for score, name in importance:
                        print(f"{name} ==> {'#' * int(10 * score)} {score}")
                
                if not save_last or i == epoch - 1:
                    save(i)

        if "predict" in self.samplers:
            phase = "predict"
            iter = self.samplers[phase].iter(self.batch_size, phase)
            ys = []
            yps = []
            for i, data in enumerate(iter):
                for name, m in self.models.items():
                    pred = m.model.predict(
                        data["x"]
                    )
                    y = data['y_pred'].reshape([-1])
                    # print(pred)
                    # print(data['y_pred'])
                    print(f"{i} ==> {np.corrcoef(pred, y)[0, 1]}")
                    ys.append(y)
                    yps.append(pred)
            print("corrcoef:", np.corrcoef(np.concatenate(ys), np.concatenate(yps))[0, 1]) 

def get_samplers_cpp(label_gen, date_ranges, csi=None):
    # loader = QlibDataloader(os.path.expanduser("~/output/qlib_bin"), [label_gen])
    # loader = QlibDataloader(os.path.expanduser("~/output/qlib_bin"), [label_gen], "csi300")
    loader = FtDataloader("tmp", [label_gen])
    return {k: SamplersCpp(loader, v) for k, v in date_ranges.items()}

if __name__ == "__main__":

    with Context() as ctx:

        def label_gen(data):
            l = data["close"].shape[0]
            return {
                "pred": np.concatenate(
                    [np.log(data["open"][2:] / data["close"][1:-1]), [float("nan")] * 2]
                )[:l],
                # "pred": np.concatenate(
                #     [data["close"][2:] / data["close"][1:-1] - 1, [float("nan")] * 2]
                # )[:l],
                # "cls": np.concatenate([data["limit_flag"][1:], [float("nan")]])[:l],
                #  "cls": get_label(data),
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
        # saved_models = from_cache(f"models.pkl")
        saved_models = None
        # for save_name in ["cls", "reg"]:
        # for save_name in ["reg", "cls"]:
        for save_name in ["cat"]:
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
                # for i in schedule:
                model_name = f"s_{save_name}"
                models[model_name] = RegCat(
                        #   iterations=200,
                        #   depth=20,
                        #   learning_rate=1,
                            samplers[stages[0]].feature_columns(),
                            loss_function='MAE',
                            eval_metric='MAE',
                            
                        )

                epoch_idx = -1

            trainer = CatTrainer(int(1e9), samplers, models)

            trainer.run(epoch_idx, 1, save_name)
