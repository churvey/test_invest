# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from trade.model.env import TradeEnv
import pandas as pd


from gymnasium.envs.registration import register

# register(
#     id="TradeEnv-v0",
#     entry_point=make_trade_env,  # 关键！entry_point 指向可调用对象
#     kwargs={},  # 可选：默认参数
# )

def _get_data():
    path = [
        "tmp/SH.513050-中概互联网ETF.csv",
        # "data/K_DAY/SH.515290-银行ETF天弘.csv",
    ]

    df = pd.concat([pd.read_csv(p) for p in path])
    # columns = "code,name,time_key,open,close,high,low,pe_ratio,turnover_rate,volume,turnover,change_rate,last_close"
    columns = "code,time_key,open,close,high,low,volume".split(",")
    columns_rename = "instrument,datetime,open,close,high,low,volume".split(",")
    df = df[columns]
    df.columns = columns_rename

    keys = "open,close,high,low,volume".split(",")
    all_inputs = {
        k: (
            df[k].to_numpy(dtype="double")
            if k not in ["instrument", "datetime"]
            else df[k].to_numpy()
        )
        for k in df.columns
    }
    inputs = {k: all_inputs[k] for k in keys}

    import talib
    from talib import abstract

    print(talib.get_functions())

    print(len(talib.get_functions()))

    for f in talib.get_functions():
        # print(f)
        if f in ["AD", "OBV"]:
            continue

        func = abstract.Function(f)
        if "timeperiod" in func.info["parameters"]:
            outputs = func(inputs, timeperiod=20)
        elif "periods" not in func.info["input_names"]:
            outputs = func(inputs)
        else:
            # skip
            print(f"skip func {f}")
            continue
        # outputs = dict(zip(func.info["output_names"], outputs))
        output_names = [f"{f}_{o}" for o in func.info["output_names"]]
        outputs = dict(zip(output_names, outputs))
        outputs = {
            k: v
            for k, v in outputs.items()
            if not np.all(np.isnan(v)) and v.dtype == np.float64
        }
        # print(output_names)
        all_inputs.update(outputs)
    df = pd.DataFrame(all_inputs)
    df = df.dropna()
    print(f"columns:{len(df.columns)}")
    print(df)
    cols = [c for c in df.columns if ("BBANDS" in c or "MA_" in c or "z_pos" in c)]
    cols += ["datetime", "high", "low"]
    print(df[cols][df["datetime"] >= "2017-05-04 00:00:00"])
    return df

    # df = df[columns]


def get_data():
    path = [
        "tmp/SH.513050-中概互联网ETF.csv",
        # "data/K_DAY/SH.515290-银行ETF天弘.csv",
    ]

    df = pd.concat([pd.read_csv(p) for p in path])
    # columns = "code,name,time_key,open,close,high,low,pe_ratio,turnover_rate,volume,turnover,change_rate,last_close"
    columns = "code,time_key,open,close,high,low,volume,change_rate".split(",")
    columns_rename = "instrument,datetime,open,close,high,low,volume,change".split(",")
    df = df[columns]
    df.columns = columns_rename

    # keys = "open,close,high,low,volume".split(",")
    all_inputs = {k: df[k].to_numpy() for k in df.columns}

    from trade.data.feature.feature import Feature
    data = Feature(all_inputs)()
    df = pd.DataFrame(data)
    df = df.dropna()
    print(df["z_pos_20"].to_list())
    # 1/0
    print(f"columns:{len(df.columns)}")
    # print(df)
    cols = [c for c in df.columns if "20" in c and ("bollinger" in c or "ma" in c or "z_pos" in c)]
    cols += ["datetime", "high", "low"]
    print(df[cols][df["datetime"] >= "2017-05-04 00:00:00"])
    return df


if __name__ == "__main__":
    get_data()