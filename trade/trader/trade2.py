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
        preds = [p["instrument,datetime,y,y_p,y_pred".split(",")] for p in preds if p is not None] 
        
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
        
        # f2 = FtDataloader("./qmt", extend_feature=False).features
        insts = preds["instrument"].drop_duplicates().to_list()
        print(f"insts len:{len(insts)}")
        
        f2 = QlibDataloader(os.path.expanduser("~/output/qlib_bin"), [label_gen], extend_feature=False, insts=insts).features
        preds = preds.merge(f2.reset_index(drop=True), how="left", on = ["instrument", "datetime"])
        
    print(preds)
        
    cash = 200000.0
    holds = {}
    dates = np.sort(np.unique(preds["datetime"].to_numpy()))
    for i, date in enumerate(dates):
        if i < len(dates) - 1:
            # print(f"begin to prcocess {date}")
            p = preds[preds["datetime"]==date]
            buy_date = dates[i+1]
            buy_p = f2[f2["datetime"]==buy_date]
            candi_f = p[(p["y_p"] >=2)].sort_values("y_p", ascending=False)
            print(candi_f)
            candi = candi_f.head(10)["instrument"].to_list()
            
            candi_sell = holds.copy()
            
            
            def get_price(c, p = "open"):
                
                p_c = buy_p[buy_p["instrument"] == c]
                price = (p_c[p] / p_c["factor"]).iloc[0]
                # print(f"{price} vs {round(price, 2)}")
                price = round(price, 2)
                return price 
            
            if len(candi) > 0:
                buy_caches = cash // len(candi)
                for c in candi:
                    if c in holds:
                        candi_sell.pop(c)
                        continue
                    # print(buy_p, c)
                    pc = buy_p[buy_p["instrument"] == c]
                    if len(pc) == 0 or pc["y_cb"].iloc[0] == False:
                        print(f"cannot by {date} {c} ")
                        continue
                    
                    open = get_price(c)
                    
                    # buy(buy_caches, c, open)
                    
                    amount = buy_caches * 100 // (open * 100 * 100) * 100
                    if amount > 0:
                        cash -= open * amount
                        holds[c] = amount, open
                        
                        print(f"buy {buy_caches} {c} {date} {open} {open * amount}" )
                        # assert open * amount > 0
            
            for c,v in candi_sell.items():
                pc = buy_p[buy_p["instrument"] == c]
                if len(pc) == 0 or pc["y_cs"].iloc[0] == False:
                    continue
                else:
                    open = get_price(c)
                    value = open * v[0]
                    cash += value
                    holds.pop(c)
                    print(f"sell {c} {date} {open}-{v[1]}={open - v[1]} {(open-v[1])/v[1]*100 :2f}%" )
            
            try:
                total_value = cash
                for c, v in holds.items():
                    total_value += v[0] * get_price(c, "close")    
                print(f"predict on {date} action on {buy_date} total_value when close {total_value}")
            except BaseException as e:
                print(e)
                pass
    
trade()