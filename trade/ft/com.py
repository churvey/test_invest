from futu import *
import pandas as pd
import os

# import ray
import time
from trade.data.loader import QlibDataloader, FtDataloader


def update_latest_price(df):
    cols = ["instrument", "vwap", "open", "close","high", "low", "volume", "datetime", "change", "factor"]
    df = df[cols]

    inst = df["instrument"].tolist()
    factor = df["factor"].tolist()
    
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    
    codes = [p[:2]+"."+p[2:] for p in inst]

    ret, data = quote_ctx.get_market_snapshot(codes)
# open_price	float	今日开盘价
# high_price	float	最高价格
# low_price
# volume
    if ret == RET_OK:
        data["change"] = (data["last_price"] - data["prev_close_price"]) / data["prev_close_price"]
        data["instrument"] = inst
        data["factor"] = factor
        data = data[["instrument", "avg_price","open_price", "last_price","high_price", "low_price", "volume", "update_time", "change","factor"]]
        data.columns = ["instrument", "vwap", "open", "close","high", "low", "volume", "datetime", "change", "factor"]
        data["datetime"] = data["datetime"].str[:len("2025-01-02")]
        for k in [ "low","high", "close", "open", "vwap"]:
            data[k] *= data["factor"]
        data["volume"] /= data["factor"] / 100
        data = data[df.columns]
        
        # m = dict(zip(
        #     data["code"].tolist(),
        #     data["last_price"].tolist()
        # ))
        # print(data)
        # print(data['code'][0])    # 取第一条的股票代码
        # print(data['code'].values.tolist())   # 转为 list
    else:
        print('error:', data)
    quote_ctx.close() # 结束后记得关闭当条连接，防止连接条数用尽

    print(df)
    
    print(data)
    


if __name__ == "__main__":
    df = QlibDataloader(os.path.expanduser("~/output/qlib_bin"), [], extend_feature=False).features
    df = df[df["datetime"] == "2025-05-13"]
    # print(df)
    update_latest_price(df)