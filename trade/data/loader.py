import torch
from torch import nn
import pandas as pd
import numpy as np
import os
from bisect import bisect_left, bisect_right
from tqdm import tqdm
import ray


def get_inst(path, type="all"):
    instruments = os.path.join(path, "instruments", f"{type}.txt")
    ins = {}
    with open(instruments) as fp:
        for p in fp:
            tmp = p.strip().split("\t")
            if tmp[0] not in ins:
                ins[tmp[0]] = tmp[1:]
            else:
                ins[tmp[0]] += tmp[1:]
    sorted_keys = sorted(list(ins.keys()))
    return {k: ins[k] for k in sorted_keys}


@ray.remote
def parallel_get_stock_features(loader, stock, day_range):
    return loader.get_stock_features(stock, day_range), stock


class Summarizer:
    def __init__(self, data):
        self.high = np.nanquantile(data, 0.99, axis=0, keepdims=True)
        self.low = np.nanquantile(data, 0.01, axis=0, keepdims=True)
        tmp = np.clip(data, self.low, self.high)
        self.mean = np.nanmean(tmp)
        self.std = np.nanstd(tmp)

    def __call__(self, data):
        data = np.clip(data, self.low, self.high)
        data = (data - self.mean) / self.std
        return data.astype("float32")


class Dataloader:

    def get_all_days(self):
        days_path = os.path.join(self.path, "calendars", "day.txt")
        days = []
        with open(days_path) as fp:
            for p in fp:
                days.append(p.strip())
        return days

    def get_stock_days(self, stock, day_range):
        begin, end = day_range
        b = bisect_left(self.days, begin)
        if self.days[b] != begin:
            print(f"begin not in be careful {stock} [{begin}, {end}]")
            b += 1
        e = bisect_right(self.days, end)
        if self.days[e - 1] != end:
            e -= 1
        return self.days[b:e]

    def get_stock_features(self, stock, day_range):
        data = {}
        f_days = np.array(self.get_stock_days(None, day_range))
        feature_path = os.path.join(self.path, "features", stock.lower())
        for file_name in os.listdir(feature_path):
            feature = np.fromfile(
                os.path.join(feature_path, file_name), dtype=np.float32
            )
            feature_name = file_name.split(".")[0]
            data[feature_name] = feature[1:]  # first element is stock id?
            assert (
                f_days.shape[0] == data[feature_name].shape[0]
            ), f" shape not equal feature:{feature_name} shape:{f_days.shape[0]} vs {data[feature_name].shape[0]}"

        data["datetime"] = f_days
        data["instrument"] = np.full_like(f_days, stock)
        data = self.add_columns(data)
        return data

    def get_base_columns(self):
        path = os.path.join(self.path, "features")
        # print(os.listdir(path))
        feature_path = os.path.join(path, os.listdir(path)[0])
        # print(feature_path)
        # feature_path = os.path.join(self.path, "features", "SH000300")
        return [file_name.split(".")[0] for file_name in os.listdir(feature_path)]

    @property
    def feature_columns(self):
        return [
            i
            for i in self.features.columns
            if i not in (self.base_columns + self.labels + self.indices)
        ]

    def get_inst(self, type="all"):
        return get_inst(self.path, type=type)

    def get_features(self, n_parallel=16, type="all"):
        ins = self.get_inst(type)
        data = []
        l = 0
        tasks = []
        for i, (stock, day_range) in tqdm(enumerate(ins.items())):
            tasks.append(parallel_get_stock_features.remote(self, stock, day_range))
            while len(tasks) >= n_parallel or (i == len(ins.items()) - 1 and tasks):
                rs_np = None
                try:
                    ready, tasks = ray.wait(tasks)
                    rs_np, stock_get = ray.get(ready)[0]
                    single = pd.DataFrame.from_dict(rs_np)
                except BaseException as e:
                    assert rs_np is not None, str(e)
                    shape = {k: v.shape for k, v in rs_np.items()}
                    print(f" {stock}: shapes {shape} {e}")
                data.append(single)
                r = l + len(data[-1])
                self.inst_count[stock_get] = l, r
                l = r
            # ### debug
            # if len(data) > 10:
            #     break
            # ### debug
        return pd.concat(data)

    def select_by_csi(self, csi="csi300"):
        ins = self.get_inst(csi)
        ind = np.zeros(len(self.features), dtype=bool)
        date = self.features["datetime"]
        for stock, day_ranges in ins.items():
            if stock not in self.inst_count:
                print(f"error , stock not found {stock}")
                continue
            ind_start, ind_end = self.inst_count[stock]
            datetime = date[ind_start:ind_end]
            for i in range(0, len(day_ranges), 2):
                begin, end = day_ranges[i], day_ranges[i + 1]
                begin = ind_start + np.searchsorted(datetime, begin, "left")
                end = ind_start + np.searchsorted(datetime, end, "right")
                assert end <= ind_end
                ind[begin:end] = True

        return self.features[ind]

    def __init__(self, path, label_generators=[]):
        self.path = path
        self.inst_count = {}
        self.label_generators = label_generators
        self.indices = ["instrument", "datetime"]
        self.base_columns = self.get_base_columns()
        self.days = self.get_all_days()
        self.features = self.get_features()
        # self.labels = self.labels()
        # self.feature_columns = self.feature_columns()

    def add_columns(self, data):
        from .feature.feature import Feature

        data = Feature(data=data)()
        labels = {}
        for gen in self.label_generators:
            labels.update({f"y_{k}": v for k, v in gen(data).items()})
        data.update(labels)
        return data
    
    @property
    def labels(self):
        return [i for i in self.features.columns if i.startswith("y_")]
