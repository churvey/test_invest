import torch
from torch import nn
import pandas as pd
import numpy as np
import os
from numpy.lib.stride_tricks import sliding_window_view



a = np.array([0.37214212135762714, 1.9769927480283929, float("nan"), float("nan"), 0.34202851190906275]).reshape([1, -1])
b = np.array([0.0, -0.038674160838127136, float("nan"), 0.0011484858114272356, 0.0]).reshape([1, -1])
    

    
def cov(x, y, valid):
    # m = x * y
    # valid = ~(np.isnan(x) | np.isnan(y))
    # print(valid)
    return np.mean(x * y, axis=-1, where=valid) - np.mean(
        y, axis=-1, where=valid
    ) * np.mean(x, axis=-1, where=valid)


def corr(x, y):
    # x = x.astype("float64")
    # y = y.astype("float64")
    # m = x * y    
    valid = ~(np.isnan(x) | np.isnan(y))
    c = np.sqrt(cov(x, x, valid)) * np.sqrt(cov(y, y, valid))
    v = cov(x, y, valid)
    rs = np.zeros_like(c)
    vs = c != 0
    rs[vs] = v[vs] / c[vs]
    # rs = np.divide(cov(x, y, valid), c, where=c != 0)
    # rs = np.where(c != 0, cov(x,y,valid) / c, 0)

    # print(c)
    # rs =  np.divide(cov(m/x, m/y), c, where=c!=0)
    # print(rs)
    # l = np.abs(rs[~np.isnan(rs)]) <= 1.0001
    
    # if not np.all(
    #     np.abs(rs[~np.isnan(rs)]) <= 1.0001
    # ):
    #     import pdb
    #     pdb.set_trace()
    

    # assert np.all(
    #     np.abs(rs) <= 1.0001
    # ), f" {rs[~l]} {x[~l][0].tolist()} {y[~l][0].tolist()}"
    return rs

# def corr(x, y):
#     # x = x.astype("float64")
#     # y = y.astype("float64")
#     # m = x * y    
#     print("corr1",corr1(a,b))

#     # c = np.sqrt(cov(m/y, m/y) * cov(m/x, m/x))
#     c = np.sqrt(cov(x, x)) * np.sqrt(cov(y, y))
#     rs = np.divide(cov(x, y), c, where=c != 0)

#     # print(c)
#     # rs =  np.divide(cov(m/x, m/y), c, where=c!=0)
#     # print(rs)
#     l = np.abs(rs) <= 1.0001

#     assert np.all(
#         np.abs(rs) <= 1.0001
#     ), f" {rs[~l]} {x[~l][0].tolist()} {y[~l][0].tolist()}"
#     return rs


class Feature:
    def __init__(self, data, window=5, rolling_window=[5, 10, 20, 30, 60]):
        self.data = data
        self.window = window
        self.rolling_window = rolling_window

    def __call__(self):
        import inspect

        data = self.data
        method_list = inspect.getmembers(Feature, predicate=inspect.isfunction)
        method_list = sorted(method_list, key=lambda x: x[0])
        # print([m[0] for m in method_list])
        for m in method_list:
            # print(m[0])
            if m[0].startswith("__"):
                continue
            else:
                r = m[1](self, data)
                if r is not None:
                    data[m[0]] = r
        return data

# 1、沪市主板：以600、601或603开头

# 2、深市主板：以000、001、002、003开头

# 3、创业板：以300开头，属于深交所

# 4、科创板：以688开头，属于上交所

# 5、北交所：以8开头

    def limit_flag(self, data):
        limit = 0.099
        if data["instrument"][0][2:5] in ["300", "688"]:
            limit = 0.199
        if data["instrument"][0][2:3] in ["8"]:
            limit = 0.299
        
        rs = np.zeros(data["close"].shape)

        up_flag = (data["change"] >= limit) & (data["high"] - data["close"] < 1e-5)

        rs[up_flag] = 1

        low_flag = (data["change"] <= -limit) & (data["close"] - data["low"]  < 1e-5)

        rs[low_flag] = 2

        return rs

    def kmid(self, data):
        return (data["close"] - data["open"]) / data["open"]

    def klen(self, data):
        return (data["high"] - data["low"]) / data["open"]

    def kmid2(self, data):
        return (data["close"] - data["open"]) / (data["high"] - data["low"] + 1e-12)

    def kup(self, data):
        return (data["high"] - np.maximum(data["open"], data["close"])) / data["open"]

    def kup2(self, data):
        return (data["high"] - np.maximum(data["open"], data["close"])) / (
            data["high"] - data["low"] + 1e-12
        )

    def klow(self, data):
        return (np.minimum(data["open"], data["close"]) - data["low"]) / data["open"]

    def klow2(self, data):
        return (np.minimum(data["open"], data["close"]) - data["low"]) / (
            data["high"] - data["low"] + 1e-12
        )

    def ksft(self, data):
        return (data["close"] * 2 - data["high"] - data["low"]) / data["open"]

    def ksft2(self, data):
        return (data["close"] * 2 - data["high"] - data["low"]) / (
            data["high"] - data["low"] + 1e-12
        )

    # def price(self, data):
    #     for i in range(self.window):
    #         for key in ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP"]:
    #             key = key.lower()
    #             if key in data:
    #                 if data["close"].shape[0] < i:
    #                     val = np.array([float("nan")] * data["close"].shape[0])
    #                 else:
    #                     val = (
    #                         data[key][:-i] / data["close"][i:]
    #                         if i > 0
    #                         else data[key] / data["close"]
    #                     )
    #                     val = np.concatenate([[float("nan")] * i, val])
    #                 data[f"{key}_{i}"] = val
    #     return None

    # def volume(self, data):
    #     for i in range(1, self.window):
    #         for key in ["volume"]:
    #             if data["close"].shape[0] < i:
    #                 val = np.array([float("nan")] * data["close"].shape[0])
    #             else:
    #                 val = data[key][:-i] / (data["volume"][i:] + 1e-12)
    #                 val = np.concatenate([[float("nan")] * i, val])
    #             data[f"{key}_{i}"] = val
    #     return None

    def corr(self, data):
        name = "corr"
        for i in self.rolling_window:
            v = np.concatenate([[float("nan")] * (i - 1), data["volume"]])
            c = np.concatenate([[float("nan")] * (i - 1), data["close"]])
            v = np.log(v + 1)

            c = sliding_window_view(c, i)
            v = sliding_window_view(v, i)
            val = np.full(data["close"].shape, float("nan"))
            is_valid = ~np.isnan(data["volume"]) & ~np.isnan(data["close"])
            val[is_valid] = corr(v[is_valid], c[is_valid])

            assert val.shape == data["close"].shape
            data[f"{name}_{i}"] = val
        return None

    def cord(self, data):
        name = "cord"
        for i in self.rolling_window:
            v = data["volume"][1:] / (data["volume"][:-1])
            # c = data["change"]
            v = np.concatenate([[float("nan")] * i, v])
            c = np.concatenate([[float("nan")] * (i - 1), data["change"]])
            v = np.log(v + 1)

            is_valid = ~np.isnan(v[i - 1 :]) & ~np.isnan(c[i - 1 :])

            c = sliding_window_view(c, i)
            v = sliding_window_view(v, i)
            val = np.full(data["close"].shape, float("nan"))

            val[is_valid] = corr(v[is_valid], c[is_valid])

            assert val.shape == data["close"].shape
            data[f"{name}_{i}"] = val
        return None

    def sump(self, data):
        name = "sump"
        self.__sumx__(data, name, lambda a, b: a, "close")
        return None

    def sumn(self, data):
        name = "sumn"
        self.__sumx__(data, name, lambda a, b: b, "close")
        return None

    def sumd(self, data):
        name = "sumd"
        self.__sumx__(data, name, lambda a, b: a - b, "close")
        return None

    def __sumx__(self, data, name, func, column):
        #   name = "sumd"
        # "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
        for i in self.rolling_window:
            c = data[column][1:] - data[column][:-1]
            c1 = func(np.maximum(c, 0.0), np.maximum(-c, 0.0))
            c2 = np.abs(c)
            c1 = np.concatenate([[float("nan")] * i, c1])
            c2 = np.concatenate([[float("nan")] * i, c2])
            c1 = sliding_window_view(c1, i)
            c2 = sliding_window_view(c2, i)

            val = np.nansum(c1, axis=-1) / (np.nansum(c2, axis=-1) + 1e-12)
            assert val.shape == data["close"].shape
            data[f"{name}_{i}"] = val
        return None

    def vma(self, data):
        def func2(a, b):
            return a / (b + 1e-12)

        return self.__rolling__(data, "vstd", np.nanstd, "volume", func2, "volume")
    
    def vstd(self, data):
        def func2(a, b):
            return a / (b + 1e-12)

        return self.__rolling__(data, "vma", np.nanmean, "volume", func2, "volume")

    def vsump(self, data):
        name = "vsump"
        self.__sumx__(data, name, lambda a, b: a, "volume")
        return None

    def vsumn(self, data):
        name = "vsumn"
        self.__sumx__(data, name, lambda a, b: b, "volume")
        return None

    def vsumd(self, data):
        name = "vsumd"
        self.__sumx__(data, name, lambda a, b: a - b, "volume")
        return None

    def roc(self, data):
        for i in self.rolling_window:
            if data["close"].shape[0] < i:
                val = np.array([float("nan")] * data["close"].shape[0])
            else:
                val = data["close"][:-i] / data["close"][i:]
                val = np.concatenate([[float("nan")] * i, val])
            data[f"roc_{i}"] = val
        return None

    def __rolling__(self, data, name, func, ref="close", func2=np.divide, ref2="close"):
        for i in self.rolling_window:
            tmp = np.concatenate([[float("nan")] * (i - 1), data[ref]])
            view = sliding_window_view(tmp, i)
            val = np.full(data[ref].shape, float("nan"))
            is_valid = ~np.isnan(data[ref])
            val[is_valid] = func(view[is_valid], axis=1)
            if func2:
                val = func2(val, data[ref2])
            data[f"{name}_{i}"] = val

    def ma(self, data):
        return self.__rolling__(data, "ma", np.nanmean)

    def std(self, data):
        return self.__rolling__(data, "std", np.nanstd)

    def max(self, data):
        return self.__rolling__(data, "max", np.nanmax, "high")

    def min(self, data):
        return self.__rolling__(data, "min", np.nanmin, "low")

    def qtlu(self, data):
        return self.__rolling__(
            data,
            "qtlu",
            lambda data, axis: torch.nanquantile(
                torch.asarray(data), 0.8, dim=axis
            ).numpy(),
        )

    def qtld(self, data):
        return self.__rolling__(
            data,
            "qtld",
            lambda data, axis: torch.nanquantile(
                torch.asarray(data), 0.2, dim=axis
            ).numpy(),
        )

    def rank(self, data):
        def func(val, axis):
            close = data["close"]
            is_valid = ~np.isnan(close)
            close = close[is_valid]
            view = np.sort(val, axis=axis)
            return np.array(
                [
                    (np.searchsorted(view[j], close[j]) + 1) / view.shape[1]
                    for j in range(view.shape[0])
                ]
            )

        self.__rolling__(data, "rank", func, func2=None)

    def z_rsv(self, data):
        name = "rsv"
        for i in self.rolling_window:
            val = (1 - data[f"min_{i}"]) / (data[f"max_{i}"] - data[f"min_{i}"] + 1e-12)
            data[f"{name}_{i}"] = val
        return None
    
    def z_bolling(self, data):
        name = "bollinger"
        for i in self.rolling_window:
            up = data[f"ma_{i}"] + 2 * data[f"std_{i}"]
            down = data[f"ma_{i}"] - 2 * data[f"std_{i}"]
            data[f"{name}_u_{i}"] = up
            data[f"{name}_d_{i}"] = down
        return None   

    def imax(self, data):
        name = "imax"
        for i in self.rolling_window:
            if data["close"].shape[0] < i:
                break
            tmp = np.concatenate([[float("-inf")] * (i - 1), data["high"]])
            offset = np.concatenate(
                [np.arange(i - 1, -1, -1), np.zeros(data["close"].shape[0] - i)]
            )
            val = np.argmax(sliding_window_view(tmp, i), axis=1) + 1 - offset  # n-i +1
            val = val.astype("float32") / i
            assert val.shape == data["close"].shape
            data[f"{name}_{i}"] = val
        return None

    def imin(self, data):
        name = "imin"
        for i in self.rolling_window:
            if data["close"].shape[0] < i:
                break
            tmp = np.concatenate([[float("inf")] * (i - 1), data["low"]])
            offset = np.concatenate(
                [np.arange(i - 1, -1, -1), np.zeros(data["close"].shape[0] - i)]
            )
            val = np.argmin(sliding_window_view(tmp, i), axis=1) + 1 - offset  # n-i +1
            val = val.astype("float32") / i
            assert val.shape == data["close"].shape
            data[f"{name}_{i}"] = val
        return None

    def imxd(self, data):
        name = "imxd"
        for i in self.rolling_window:
            h = np.concatenate([[float("-inf")] * (i - 1), data["high"]])
            l = np.concatenate([[float("inf")] * (i - 1), data["low"]])
            val = np.argmax(sliding_window_view(h, i), axis=1) + 1  # n-i +1
            val -= np.argmin(sliding_window_view(l, i), axis=1) + 1  # n-i +1
            val = val.astype("float32") / i
            assert val.shape == data["close"].shape
            data[f"{name}_{i}"] = val
        return None

    def cntp(self, data):
        name = "cntp"
        increase = np.concatenate(
            [[float("nan")], (data["close"][1:] > data["close"][:-1]).astype("float32")]
        )
        for i in self.rolling_window:
            if data["close"].shape[0] < i:
                val = np.array([float("nan")] * data["close"].shape[0])
            else:
                val = np.mean(sliding_window_view(increase, i), axis=1)  # n-i +1
                val = np.concatenate([[float("nan")] * (i - 1), val])
                assert val.shape == data["close"].shape
            data[f"{name}_{i}"] = val
        return None

    def cntn(self, data):
        name = "cntn"
        decrease = np.concatenate(
            [[float("nan")], (data["close"][1:] < data["close"][:-1]).astype("float32")]
        )
        for i in self.rolling_window:
            if data["close"].shape[0] < i:
                val = np.array([float("nan")] * data["close"].shape[0])
            else:
                val = np.mean(sliding_window_view(decrease, i), axis=1)  # n-i +1
                val = np.concatenate([[float("nan")] * (i - 1), val])
                assert val.shape == data["close"].shape
            data[f"{name}_{i}"] = val
        return None

    def cntd(self, data):
        name = "cntd"

        for i in self.rolling_window:
            increase = np.concatenate(
                [
                    [float("nan")] * (i),
                    (data["close"][1:] > data["close"][:-1]).astype("float32"),
                ]
            )
            decrease = np.concatenate(
                [
                    [float("nan")] * (i),
                    (data["close"][1:] < data["close"][:-1]).astype("float32"),
                ]
            )
            val = np.mean(sliding_window_view(increase, i), axis=1)  # n-i +1
            val2 = np.mean(sliding_window_view(decrease, i), axis=1)  # n-i +1
            val -= val2
            assert val.shape == data["close"].shape
            data[f"{name}_{i}"] = val
        return None

        # return self.__rolling__(
        #     data, "cntd", lambda data, axis: np.mean((data[...,1:] > data[...,:-1]).astype("float32"), axis = axis) - np.mean((data[...,1:] < data[...,:-1]).astype("float32"), axis = axis), func2 = None
        # )

        # if use("CNTD"):
        #     # The diff between past up day and past down day
        #     fields += ["Mean($close>Ref($close, 1), %d)-Mean($close<Ref($close, 1), %d)" % (d, d) for d in windows]
        #     names += ["CNTD%d" % d for d in windows]

        # if "rolling" in config:
        #     windows = config["rolling"].get("windows", [5, 10, 20, 30, 60])
        #     include = config["rolling"].get("include", None)
        #     exclude = config["rolling"].get("exclude", [])
        #     # `exclude` in dataset config unnecessary filed
        #     # `include` in dataset config necessary field
        #     if use("BETA"):
        #         # The rate of close price change in the past d days, divided by latest close price to remove unit
        #         # For example, price increase 10 dollar per day in the past d days, then Slope will be 10.
        #         fields += ["Slope($close, %d)/$close" % d for d in windows]
        #         names += ["BETA%d" % d for d in windows]
        #     if use("RSQR"):
        #         # The R-sqaure value of linear regression for the past d days, represent the trend linear
        #         fields += ["Rsquare($close, %d)" % d for d in windows]
        #         names += ["RSQR%d" % d for d in windows]
        #     if use("RESI"):
        #         # The redisdual for linear regression for the past d days, represent the trend linearity for past d days.
        #         fields += ["Resi($close, %d)/$close" % d for d in windows]
        #         names += ["RESI%d" % d for d in windows]

        #     if use("CORR"):
        #         # The correlation between absolute close price and log scaled trading volume
        #         fields += ["Corr($close, Log($volume+1), %d)" % d for d in windows]
        #         names += ["CORR%d" % d for d in windows]
        #     if use("CORD"):
        #         # The correlation between price change ratio and volume change ratio
        #         fields += ["Corr($close/Ref($close,1), Log($volume/Ref($volume, 1)+1), %d)" % d for d in windows]
        #         names += ["CORD%d" % d for d in windows]

        #     if use("SUMP"):
        #         # The total gain / the absolute total price changed
        #         # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
        #         fields += [
        #             "Sum(Greater($close-Ref($close, 1), 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
        #             for d in windows
        #         ]
        #         names += ["SUMP%d" % d for d in windows]
        #     if use("SUMN"):
        #         # The total lose / the absolute total price changed
        #         # Can be derived from SUMP by SUMN = 1 - SUMP
        #         # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
        #         fields += [
        #             "Sum(Greater(Ref($close, 1)-$close, 0), %d)/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d)
        #             for d in windows
        #         ]
        #         names += ["SUMN%d" % d for d in windows]
        #     if use("SUMD"):
        #         # The diff ratio between total gain and total lose
        #         # Similar to RSI indicator. https://www.investopedia.com/terms/r/rsi.asp
        #         fields += [
        #             "(Sum(Greater($close-Ref($close, 1), 0), %d)-Sum(Greater(Ref($close, 1)-$close, 0), %d))"
        #             "/(Sum(Abs($close-Ref($close, 1)), %d)+1e-12)" % (d, d, d)
        #             for d in windows
        #         ]
        #         names += ["SUMD%d" % d for d in windows]
        #     if use("VMA"):
        #         # Simple Volume Moving average: https://www.barchart.com/education/technical-indicators/volume_moving_average
        #         fields += ["Mean($volume, %d)/($volume+1e-12)" % d for d in windows]
        #         names += ["VMA%d" % d for d in windows]
        #     if use("VSTD"):
        #         # The standard deviation for volume in past d days.
        #         fields += ["Std($volume, %d)/($volume+1e-12)" % d for d in windows]
        #         names += ["VSTD%d" % d for d in windows]
        #     if use("WVMA"):
        #         # The volume weighted price change volatility
        #         fields += [
        #             "Std(Abs($close/Ref($close, 1)-1)*$volume, %d)/(Mean(Abs($close/Ref($close, 1)-1)*$volume, %d)+1e-12)"
        #             % (d, d)
        #             for d in windows
        #         ]
        #         names += ["WVMA%d" % d for d in windows]
        #     if use("VSUMP"):
        #         # The total volume increase / the absolute total volume changed
        #         fields += [
        #             "Sum(Greater($volume-Ref($volume, 1), 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
        #             % (d, d)
        #             for d in windows
        #         ]
        #         names += ["VSUMP%d" % d for d in windows]
        #     if use("VSUMN"):
        #         # The total volume increase / the absolute total volume changed
        #         # Can be derived from VSUMP by VSUMN = 1 - VSUMP
        #         fields += [
        #             "Sum(Greater(Ref($volume, 1)-$volume, 0), %d)/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)"
        #             % (d, d)
        #             for d in windows
        #         ]
        #         names += ["VSUMN%d" % d for d in windows]
        #     if use("VSUMD"):
        #         # The diff ratio between total volume increase and total volume decrease
        #         # RSI indicator for volume
        #         fields += [
        #             "(Sum(Greater($volume-Ref($volume, 1), 0), %d)-Sum(Greater(Ref($volume, 1)-$volume, 0), %d))"
        #             "/(Sum(Abs($volume-Ref($volume, 1)), %d)+1e-12)" % (d, d, d)
        #             for d in windows
        #         ]
        #         names += ["VSUMD%d" % d for d in windows]

        # return fields, names
