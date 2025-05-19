import argparse
import datetime
import random
import os
import pandas as pd
import numpy as np
from trade.data.loader import QlibDataloader,FtDataloader
# from trade.data.sampler import *
# from trade.model.reg_dnn import RegDNN
# from trade.model.cls_dnn import ClsDNN
from trade.train.utils import *

def label_gen(data):
    limit = 0.099
    if data["instrument"][0][2:5] in ["300", "688"]:
        limit = 0.199
    if data["instrument"][0][2:3] in ["8"]:
        limit = 0.299

    rs = np.zeros(data["close"].shape)

    up_flag = (data["change"] >= limit) & (data["high"] - data["close"] < 1e-5)

    cannot_buy = (data["change"] >= limit) & (data["high"] - data["low"] < 1e-5)

    cannot_sell = (data["change"] <= -limit) & (data["high"] - data["low"] < 1e-5)

    rs[up_flag] = 1

    pred = np.zeros(rs.shape)
    pred[:] = float("nan")

    lrs = np.concatenate([[float("nan")], rs[:-1]])
    nrs = np.concatenate([rs[1:], [float("nan")]])

    first_up = (lrs == 0) & (rs == 1)

    pred[first_up & (nrs == 1)] = 1
    pred[first_up & (nrs == 0)] = 0
    if pred[-1] != 0 and pred[-1] != 1 and first_up[-1]:
        pred[-1] = 0

    # pred = np.concatenate(
    #         [
    #             (data["open"][2:] / data["open"][1:-1] - 1) * 100,
    #             [float("nan")] * 2,
    #         ]
    # )[:l]

    # pred = np.concatenate(
    #         [
    #             (data["open"][2:] / data["close"][1:-1] - 1) * 100,
    #             [float("nan")] * 2,
    #         ]
    # )[:l]

    # inc = np.concatenate(
    #         [
    #             (data["open"][1:] / data["close"][:-1] - 1) * 100,
    #             [float("nan")] * 1,
    #         ]
    # )[:l]
    # valid = (np.abs(pred) <= 0.098) & (np.abs(data["change"]) < 0.098)
    # valid = (np.abs(inc) <= 0.098 * 100) # 去掉涨停板/跌停板
    # valid = (inc <= 0.098 * 100) # 去掉涨停板
    # pred[~valid] = float("nan")

    # print(f'valid {valid.sum()} vs {len(valid)}' )
    # 1/0

    # for k in data.keys():
    #     data[k] = data[k][valid]

    return {
        # "pred": np.concatenate(
        #     [np.log(data["open"][1:] / data["close"][:-1]), [float("nan")] * 1]
        # )[:l],
        # "pred": pred,
        "cs": ~cannot_sell,
        "cb": ~cannot_buy,
        # "inc": inc,
        # "pred": np.concatenate(
        #     [np.log(data["open"] / data["close"]), []]
        # )[:l],
        # "pred": np.concatenate(
        #     [np.abs(np.log(data["open"][2:] / data["open"][1:-1])), [float("nan")] * 2]
        # )[:l],
        # "cls": np.concatenate([data["limit_flag"][1:], [float("nan")]])[:l],
        #  "cls": get_label(data),
    }


def trade():
    with Context() as ctx:
        save_names = [f"r_0_ClsDNN_exp_{i}" for i in range(10)]
        preds = [from_cache(f"{s}/predict.pkl") for s in save_names]
        preds = [p for p in preds if p is not None]
        print(preds[0])
        agg_dict = {
            k:["mean"] for k in preds[0].columns if "y" in k
        }
        agg_dict["y_p"] = ["sum"]
        preds = pd.concat(preds).groupby(["instrument", "datetime"]).agg(agg_dict).reset_index()
        preds.columns = [col[0] if col[1] != '' else col[0] for col in preds.columns]
        # p = preds.sort_values(["datetime", "instrument"])
        # print(p.tail(20))
        # p = p[(p["y_p"] >= 1) | (p["y"]==1)]
        # print(p)
        
        f2 = FtDataloader("./qmt", [label_gen], extend_feature=False).features
        
        preds = preds.merge(f2.reset_index(drop=True), how="left", on = ["instrument", "datetime"])
        
    print(preds)
        
    cash = 20000.0
    holds = {}
    dates = np.unique(np.sort(preds["datetime"].to_numpy()))
    for i, date in enumerate(dates):
        if i < len(dates) - 1:
            p = preds[preds["datetime"]==date]
            buy_date = dates[i+1]
            buy_p = f2[f2["datetime"]==buy_date]
            candi = p[(p["y_p"] >=1) & (p["y_cb"] >=1)].sort_values("y_p", ascending=False).head(1)["instrument"].to_list()
            candi_sell = holds.copy()
            buy_caches = cash // len(candi)
            
            def get_price(c, p = "open"):
                print(buy_p, c)
                p_c = buy_p[buy_p["instrument"] == c]
                open = (p_c[p] / p_c["factor"]).iloc[0]
                open = round(open, 2)
                return open 
            
            for c in candi:
                if c in holds:
                    candi_sell.pop(c)
                    continue
                open = get_price(c)
                
                # buy(buy_caches, c, open)
                
                amount = buy_caches * 100 // (open * 100 * 100) * 100
                cash -= open *amount
                holds[c] = amount
                
                print(f"buy {c} {buy_date} " )
            
            for c,v in candi_sell.items():
                p_c = buy_p[[buy_p["instrument"] == c]]
                if p_c["y_cs"] == 0:
                    continue
                else:
                    open = get_price(c)
                cash += open * v
                holds.pop(c)
                
            total_value = cash
            for c, v in holds.items():
                total_value += v * get_price(c, "close")    
            print(f"predict on {date} action on {buy_date} total_value when close {total_value}")
    
trade()