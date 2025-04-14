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
def parallel_get_stock_features(loader, *args):
    return loader.get_stock_features(*args)


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


class BaseDataloader:

    def __init__(self, path, label_generators=[]):
        self.path = path
        self.label_generators = label_generators
        self.indices = ["instrument", "datetime"]
        self.base_columns = []
        self.features = None
    
    @property
    def feature_columns(self):
        return [
            i
            for i in self.features.columns
            if i not in (self.base_columns + self.labels + self.indices)
        ]

    def get_stock_params(self):
        raise NotImplementedError

    def get_features(self, n_parallel=16):
        params = self.get_stock_params()
        data = []
        l = 0
        tasks = []
        for i, p in tqdm(enumerate(params)):
            tasks.append(parallel_get_stock_features.remote(self, *p))
            while len(tasks) >= n_parallel or (i == len(params) - 1 and tasks):
                rs_np = None
                try:
                    ready, tasks = ray.wait(tasks)
                    rs_np = ray.get(ready)[0]
                    single = pd.DataFrame.from_dict(rs_np)
                except BaseException as e:
                    assert rs_np is not None, str(e)
                    shape = {k: v.shape for k, v in rs_np.items()}
                    print(f" {stock}: shapes {shape} {e}")
                data.append(single)
        return pd.concat(data)

    def add_columns(self, data):
        from .feature.feature import Feature

        if not self.base_columns:
            self.base_columns = list(data.keys())

        data = Feature(data=data)()
        labels = {}
        for gen in self.label_generators:
            labels.update({f"y_{k}": v for k, v in gen(data).items()})
        data.update(labels)
        return data
    
    @property
    def labels(self):
        return [i for i in self.features.columns if i.startswith("y_")]


class QlibDataloader(BaseDataloader):

    def __init__(self, path, label_generators=[], csi = None):
        super(QlibDataloader, self).__init__(path, label_generators)
        self.csi = csi
        self.csi_ins = {}
        if self.csi:
            self.csi_ins = get_inst(self.path, self.csi)
        self.days = self.get_all_days()
        self.features = self.get_features()


    def get_stock_params(self):
        d = get_inst(self.path)
        return list(d.items())[1000:1050]

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
        if self.csi_ins and stock not in self.csi_ins:
            return data
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
        
        if self.csi_ins:
            ind = np.zeros(len(f_days), dtype=bool)
            day_ranges = self.csi_ins[stock]
            ind_start, ind_end = 0, len(f_days)
            datetime = f_days
            for i in range(0, len(day_ranges), 2):
                begin, end = day_ranges[i], day_ranges[i + 1]
                begin = ind_start + np.searchsorted(datetime, begin, "left")
                end = ind_start + np.searchsorted(datetime, end, "right")
                assert end <= ind_end, f"{ datetime, begin, end}" 
                ind[begin:end] = True
            data = {
                k: v[ind] for k,v in data.items()
            }

        return data

class FtDataloader(BaseDataloader):

    def __init__(self, path, label_generators=[]):
        super(FtDataloader, self).__init__(path, label_generators)
        self.features = self.get_features()
        self.days = self.features["datetime"].unique()
    

    def get_stock_params(self):
        files = os.listdir(self.path)
        params = [(os.path.join(self.path, p),) for p in files if p.endswith(".csv")]
        return params

    def get_stock_features(self, path):
        path = [path] if not isinstance(path, list) else path
        df = pd.concat([pd.read_csv(p) for p in path])
        # columns = "code,name,time_key,open,close,high,low,pe_ratio,turnover_rate,volume,turnover,change_rate,last_close"
        columns = "code,time_key,open,close,high,low,volume,change_rate".split(",")
        columns_rename = "instrument,datetime,open,close,high,low,volume,change".split(",")
        df = df[columns]
        df.columns = columns_rename
        data = {k: df[k].to_numpy() for k in df.columns}
        data = self.add_columns(data)
        return data