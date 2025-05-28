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

def list_dir(path = "./qmt"):
    files = os.listdir(path)
    # files = [f.split("-")[0].replace(".", "") for f in files]
    files = ["".join(f.split(".")[:2][::-1]).replace(".", "") for f in files]
    return files


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

    def __init__(self, path, label_generators=[], extend_feature = True):
        self.path = path
        self.extend_feature = extend_feature
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
                    base_columns, rs_np = ray.get(ready)[0]
                    if not self.base_columns:
                        self.base_columns = base_columns
                    if isinstance(rs_np, dict):
                        single = pd.DataFrame.from_dict(rs_np)
                    else:
                        single = rs_np
                    # single = single.dropna(subset=[c for c in single.columns if c !="y_pred"])
                    single = single.dropna()
                except BaseException as e:
                    assert rs_np is not None, str(e)
                    shape = {k: v.shape for k, v in rs_np.items()}
                    print(f" {p}: shapes {shape} {e}")
                data.append(single)
        return pd.concat(data)

    def add_columns(self, data, add_label=True):
        base_columns = list(data.keys())
        if self.extend_feature:
            from .feature.feature import Feature
            data = Feature(data=data)()
        if add_label:
            labels = {}
            for gen in self.label_generators:
                labels.update({f"y_{k}": v for k, v in gen(data).items()})
            data.update(labels)
        return base_columns, data
    
    @property
    def labels(self):
        return [i for i in self.features.columns if i.startswith("y_")]


class QlibDataloader(BaseDataloader):

    def __init__(self, path, label_generators=[], csi = None, extend_feature = True, insts=[]):
        super(QlibDataloader, self).__init__(path, label_generators, extend_feature)
        self.csi = csi
        self.csi_ins = {}
        self.insts = insts
        if self.csi:
            self.csi_ins = get_inst(self.path, self.csi)
        self.days = self.get_all_days()
        self.features = self.get_features()


    def get_stock_params(self):
        d = get_inst(self.path)
        keep = []
        if self.insts:
            keep = list({ k:v for k,v in d.items() if k in self.insts}.items())
            d = {k:v for k,v in d.items() if k not in self.insts}
            
        # keys = pd.read_csv("sci.csv")["instrument"].to_list()
        # d = { k:v for k,v in d.items() if k in keys}
        # if True:
        #     inst  = get_inst(self.path, "csi300")
        #     keys = inst.keys()
        #     end = max([
        #         v[-1] for v in inst.values()
        #     ])
        #     keys = [
        #         k for k,v in inst.items() if v[-1] == end
        #     ][:100]
            
        #     # print(inst["SH600811"], end, "SH600811" in keys)
        #     # 1/0
            
        #     keys = list_dir()
        #     keys = set(keys).intersection(set(d.keys()))
        #     keys = sorted(list(keys), key=lambda x:x[2:])[:100]
            
            
        #     d = { k:v for k,v in d.items() if k in keys}
        #     rs = list(d.items())
        #     index = np.arange(len(rs))
        #     # index =  np.random.choice(index, min(100, len(rs)), replace=False)
        #     rs =  [rs[i] for i in index]
        #     print("selected stocks", rs)
        #     return rs
        #     # return [:50]
        rs =  list(d.items())
        # import random
        # random.shuffle(rs)
        # return keep + rs[:100 - len(keep)]
        return rs

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
        base_columns, data = self.add_columns(data)
        
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

        return base_columns, data

class FtDataloader(BaseDataloader):

    def __init__(self, path, label_generators=[], extend_feature = True):
        super(FtDataloader, self).__init__(path, label_generators, extend_feature)
        self.down_sample = True
        self.features = self.get_features()
        self.days = self.features["datetime"].unique()
        

    def get_stock_params(self):
        files = os.listdir(self.path)
        params = [(os.path.join(self.path, p),) for p in files if p.endswith(".feather") or p.endswith(".csv")]
        return params

    def get_stock_features(self, path):
        path = [path] if not isinstance(path, list) else path
        df = pd.concat([pd.read_csv(p) if p.endswith(".csv") else pd.read_feather(p) for p in path])
        # columns = "code,name,time_key,open,close,high,low,pe_ratio,turnover_rate,volume,turnover,change_rate,last_close"
        columns = "code,time_key,open,close,high,low,volume,change_rate".split(",")
        columns_rename = "instrument,datetime,open,close,high,low,volume,change".split(",")
        others =[p for p in df.columns if p not in columns and "Unnamed" not in p]
        columns += others
        columns_rename += others
        
        df = df[columns]
        df.columns = columns_rename
        
        
        # print(df[["datetime","open", "high","close", "low"]].tail(20))
        # 1/0
        if not self.down_sample:
            data = {k: df[k].to_numpy() for k in df.columns if k in columns_rename}
            return self.add_columns(data)
        
        # print(df)

        datetime = pd.to_datetime(df["datetime"])
        df["datetime"] = datetime
        df.set_index("datetime", inplace=True)
        # df["Y"] = datetime.dt.isocalendar().year
        # df["W"] = datetime.dt.isocalendar().week
        # df["D"] = datetime.dt.isocalendar().day
        if datetime.diff().iloc[-1].total_seconds() == 60:
            resample_range = [None, "5min"]
        #     df["H"] = datetime.dt.hour
        #     df["5min"] = datetime.dt.minute //  5 + datetime.dt.minute % 5 != 0
        else:
            resample_range = [None, "W"]
        
        resample_range = resample_range[1:]
        
        total = None
        def change_func(series):
            # if len(series) == 1:
            #     return float("nan")
            v = 1
            for p in series:
                v *= (1 + p)
            return v - 1
        for min_id, min in enumerate(resample_range):
            if min is not None:
                agg = {
                    "open":"first",
                    "close":"last",
                    "high":"max",
                    "low":"min",
                    "volume":"sum",
                    "change":change_func,
                }
                agg = {
                    p: agg[p] if p in agg else "last" for p in df.columns if p not in min
                }
                
                
                if min.endswith('min'):
                    import datetime
                    mask = (df.index.time == datetime.time(9, 30))
                    df.index = df.index.where(~mask, df.index + pd.Timedelta(minutes=1))
                df_i = df.resample(min, origin="end").agg(agg).dropna()
                df_i.reset_index(names = ["datetime"], inplace=True)
                # if min.endswith('min'):
                #     df_i["datetime"] = df_i["datetime"] + pd.Timedelta(minutes=int(min.split("min")[0]))
                
                # print(df_i[df_i["datetime"] >= "2025-05-27"][["datetime","open", "high","close", "low"]].head(10))
                print(df_i[df_i["datetime"] >= "2025-02-24"][["datetime","open", "high","close", "low"]].head(2))
                print(df_i[df_i["datetime"] >= "2025-02-24"][["datetime","open", "high","close", "low"]].tail(2))
                # print(df[df.index >= "2025-05-27"][["open", "high","close", "low"]].head(10))
                print(df[df.index >= "2025-02-24"][["open", "high","close", "low"]].head(20))
                import time
                time.sleep(1)
                1/0
                # [48, 49, 46, 45] 15:30
                # [49,49,47,46] :15:25
            else:
                df_i = df

            if len(df_i) == 0 :
                break
            
            data = {k: df_i[k].to_numpy() for k in df_i.columns if k in columns_rename}
            base_columns, data = self.add_columns(data, min_id == 0)
            data = pd.DataFrame.from_dict(data)
            
            if total is None:
                total = data.reset_index(names=["datetime"])
            else:
                columns = [c for c in data.columns if c not in base_columns or c in self.indices]
                data = data[columns]
                data.columns = [f"{col}_{min[-1]}" if col not in self.indices else col for col in data.columns]
                total = total.merge(data, how="left", on=self.indices)
                columns = [c for c in data.columns if c not in self.indices]
                total.loc[:,columns].ffill(inplace=True)

        return base_columns, total