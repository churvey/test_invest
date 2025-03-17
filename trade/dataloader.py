import torch
from torch import nn
import pandas as pd
import numpy as np
import os
from bisect import bisect_left, bisect_right


def get_inst(path, type="all"):
    instruments = os.path.join(path, "instruments", f"{type}.txt")
    ins = {}
    with open(instruments) as fp:
        for p in fp:
            tmp = p.strip().split("\t")
            # print(tmp)
            if tmp[0] not in ins:
                ins[tmp[0]] = tmp[1:]
            else:
                ins[tmp[0]] += tmp[1:]
    return ins

class Summarizer:
    def __init__(self, data):
        self.high = np.nanquantile(data, 0.99, axis = 0, keepdims = True )
        self.low  = np.nanquantile(data, 0.01, axis = 0, keepdims = True )
        tmp = np.clip(data, self.low, self.high)
        self.mean = np.nanmean(tmp)
        self.std = np.nanstd(tmp)
        # data = (data - mean) / std
        # return data
    
    def __call__(self, data):
        data = np.clip(data, self.low, self.high)
        data =  (data - self.mean) / self.std
        return data.astype("float32")
        


class Dataloader:
    def __init__(self, path, csi="csi300", baseline="SH000300", range_hint=None):
        self.path = path
        self.csi = csi
        self.baseline = baseline
        self.range_hint = range_hint
        days = os.path.join(path, "calendars", "day.txt")
        self.days = []
        with open(days) as fp:
            for p in fp:
                self.days.append(p.strip())
                

        ins = get_inst(path)
        csi = get_inst(path, csi)

        def list_features(k):
            files = os.listdir(os.path.join(path, "features", k))
            return files

        self.base_data = {}
        self.base_feature_names = []
        dfs = []
        feature_dfs = []
        base_dfs = []
        has_base = False
        
        keys = [self.baseline] + [stock for stock in ins.keys() if stock != self.baseline]
        for stock in keys:
            d_range = ins[stock]
            if stock not in csi and stock != self.baseline:
                continue

            if self.range_hint and (
                self.range_hint[1] <= d_range[0] or self.range_hint[0] >= d_range[-1]
            ):
                print(f"skip not in range {stock} {self.range_hint} vs {d_range}")
                continue

            csi_days = None
            if stock != self.baseline:
                csi_days = set()
                for d_range_csi in range(0, len(csi[stock]), 2):
                    csi_days.update(
                        self.get_days(stock, *csi[stock][d_range_csi : d_range_csi + 2])
                    )
                csi_days = np.array(sorted(list(csi_days)))

                if self.range_hint and (
                    self.range_hint[1] <= csi_days[0] or self.range_hint[0] >= csi_days[-1]
                ):
                    print(f"skip not in csi range {stock} {self.range_hint} vs {csi_days[0], csi_days[-1]}")
                    continue

            files = os.listdir(os.path.join(path, "features", stock))
            data = {}
            f_days = np.array(self.get_days(None, *d_range))
            for file_name in list_features(stock):
                feature = np.fromfile(
                    os.path.join(path, "features", stock, file_name), dtype=np.float32
                )
                feature_name = file_name.split(".")[0]
                if feature_name not in self.base_feature_names:
                    self.base_feature_names.append(feature_name)
                data[feature_name] = feature[1:]  # first element is stock id?
                assert (
                    f_days.shape[0] == data[feature_name].shape[0]
                ), f"{f_days.shape[0]} vs {data[feature_name].shape[0]}"

            
            data["datetime"] = f_days
            data["instrument"] = np.full_like(f_days, stock)
   

            if stock != self.baseline:
                self.exetend_features(data)
                csi_indices = np.searchsorted(f_days, csi_days)
                if csi_indices[-1] >= f_days.shape[0]:
                    print("days excede ", stock, f_days[-30:], csi_days[-30:])
                    csi_indices = csi_indices[csi_indices < f_days.shape[0]]
                feature_df = pd.DataFrame.from_dict(
                    {
                        k: v[csi_indices]
                        for k, v in data.items()
                        if k not in self.base_feature_names
                    }
                )
                feature_dfs.append(feature_df)
                base_df = pd.DataFrame.from_dict(
                    {
                        k: v[csi_indices]
                        for k, v in data.items()
                        if k in self.base_feature_names + ["datetime", "instrument"]
                    }
                )
                base_dfs.append(base_df)
            else:
                has_base = True

            self.base_data[stock] = {
                k: v
                for k, v in data.items()
                if k in self.base_feature_names or k == "datetime"
            }

            # if len(feature_dfs) > 50 and has_base:
            #     break
        self.feature_df = pd.concat(feature_dfs).sort_values(by=["datetime"])
        self.base_df = pd.concat(base_dfs).sort_values(by=["datetime"])

    def exetend_features(self, data):
        # data[self.label()] = np.concatenate(
        #     [data["close"][2:] / data["close"][1:-1] - 1, [float("nan")] * 2]
        # )
        base = self.base_data[self.baseline]["close"]
        base_days = self.base_data[self.baseline]["datetime"]
        close = data["close"]
        days = data["datetime"]
        # base_indices = np.searchsorted(days, base_days)
        base_indices = np.searchsorted(base_days, days)
        
        base_days2 = base_days[base_indices]
        np.testing.assert_equal(days, base_days2)
        
        base_close2 = base[base_indices]
        
        
        data["y_c1"] = np.concatenate(
            [data["close"][1:] / data["close"][0:-1] - 1, [float("nan")] * 1]
        )
        
        data["y_c2"] = np.concatenate(
            [data["close"][2:] / data["close"][1:-1] - 1, [float("nan")] * 2]
        )
        
        tmp = np.concatenate(
            [base_close2[2:] / base_close2[1:-1] - 1, [float("nan")] * 2]
        )
        
        data["y_c2_b"] = data["y_c2"] - tmp
        
        
        data["y_o2"] = np.concatenate(
            [data["open"][2:] / data["open"][1:-1] - 1, [float("nan")] * 2]
        )

     
        
        from .features import Feature

        data =  Feature(data=data)()
        
        data["y_f"] = np.concatenate(
            [data["limit_flag"][1:] ,[float("nan")] * 1]
        )
        
        return data

    def label(self):
        # return "y_f"
        return "y_c2"

    def indices(self):
        return ["datetime", "instrument"]

    def get_days(self, stock, begin, end):
        b = bisect_left(self.days, begin)
        if self.days[b] != begin:
            print(f"begin not in be careful {stock} [{begin}, {end}]")
            b += 1
        e = bisect_right(self.days, end)
        if self.days[e - 1] != end:
            # print(f"end not in be careful {stock} [{begin}, {end}]")
            e -= 1
        return self.days[b:e]

    def load(
        self,
        dates={
            "train": [
                "2008-01-01",
                "2014-12-31",
            ],
            "valid": ["2015-01-01", "2016-12-31"],
            "predict": ["2017-01-01", "2018-12-31"],
        },
        label = None
    ):
        label = label if label else self.label()
        x_cols = [
            p
            for p in self.feature_df.columns
            if p not in self.indices() and p != self.label() and not p.startswith("y")
        ]
        all_f = x_cols + [label] + self.indices()
        
        # print(f"label {label}")
        # xs = '\n'.join(sorted(x_cols))
        # print(f"x_cols:\n{xs}")

        np_label = self.feature_df[label].to_numpy()
        
        weights = np.array([np.sum(np_label == i) for i in range(3)]).astype("float32")
        weights= weights / weights.sum()
        print("weights", weights)

        # print("label 0",  np.sum(np_label == 0))
        # print("label 1",  np.sum(np_label == 1))
        # print("label -1",  np.sum(np_label == 2))
        
        
        rs = {k: {} for k in ["x", "y", "indices"]}
        for k, v in dates.items():
            df = self.feature_df
            df = df[(df["datetime"] < v[1]) & (df["datetime"] >= v[0])]
            if k == "train":
                sampler = np.random.rand((len(df)))
                indexer = np.logical_or(df[label] != 0 , sampler > weights[0])
                df = df[indexer]
            
            # df = df[]
            df = df[all_f]
            # print(f"before drop nan {k}: {len(df)}")
            df = df.dropna()
            # print(f"after drop nan {k}: {len(df)}")
            rs["x"][k] = df[x_cols].to_numpy().astype("float32")
            rs["y"][k] = df[label].to_numpy().astype("float32")
            if len(rs["y"][k].shape) == 1:
                rs["y"][k] = rs["y"][k].reshape([-1, 1])
            rs["indices"][k] = df[self.indices()]
    
        if not os.path.exists("features.csv"):
            self.feature_df.to_csv("features.csv")
        print(f"save len of features: {len(self.feature_df)}")
        # summarizer = Summarizer(rs["x"]["train"])
        # for k, v in rs["x"].items():
        #     rs["x"][k] = summarizer(v)

        rs["base_data"] = self.base_data

        return rs

    def rolling(
        self,
        train,
        valid,
        step = 1,
        label = None
    ):

        valid_days = self.get_days(None, *valid) 
        train_days = self.get_days(None, *train)
        
        days = train_days + valid_days
        # base_data = self.load({"train": train, "valid": [valid_days[0],  valid_days[step]]})
        # yield base_data
        for i in range(step + 1, len(days), 1):
            t = [days[i - step - 1],  days[i-1]]       
            if i < len(train_days):
                v = [valid[-1], valid[-1]]
            elif i < len(days) - step:
                v = [days[i], days[i + step]]
            else:
                v = [days[i], valid[-1]]
        
            base_data = self.load(
                {"train": t, "valid":v},
                label = label
            )
            # if base_data["x"]["valid"].shape[0] > 0:
                # print(f"valid dates {i} {valid}")
            # print(t, v)
            yield base_data
         
    def rolling_v2(
        self,
        date_range,
        train_days = 250 * 6,
        valid_days = 250 // 2,
        step = 250 // 2,
        label = None
    ):

        days = self.get_days(None, *date_range)
        i = len(days) - train_days - valid_days
        while i >= train_days // 2:
            train_begin = i - train_days // 2
            train_end = i + train_days
            valid_begin = train_end
            valid_end = valid_begin + valid_days
            i -= step
            yield {"train": [days[train_begin], days[train_end -1]], "valid":[days[valid_begin] , days[valid_end-1]], "predict":[days[-1] , days[-1]+"1"]}
            # base_data = self.load(
            #     ,
            #     label = label
            # )
            # yield base_data
         
            
            
