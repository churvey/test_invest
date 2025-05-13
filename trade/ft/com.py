from futu import *
import pandas as pd
import os

# import ray
import time
from trade.data.loader import QlibDataloader, FtDataloader


def update_latest_price(df):
    inst = df["instrument"].tolist()
    ratio = df["factor"].tolist()
    
    quote_ctx = OpenQuoteContext(host='127.0.0.1', port=11111)
    
    codes = [p[:2]+"."+p[2:] for p in inst]

    ret, data = quote_ctx.get_market_snapshot(codes)
    if ret == RET_OK:
        data = data[["code", "last_price"]]
        m = dict(zip(
            data["code"].tolist(),
            data["last_price"].tolist()
        ))
        print(data)
        # print(data['code'][0])    # 取第一条的股票代码
        # print(data['code'].values.tolist())   # 转为 list
    else:
        print('error:', data)
    quote_ctx.close() # 结束后记得关闭当条连接，防止连接条数用尽

    value = [m[codes[i]] for i in range(len(codes))]
    print(df)
    df["close"] = value
    print(df)
    


if __name__ == "__main__":
    df = QlibDataloader(os.path.expanduser("~/output/qlib_bin"), [], extend_feature=False).features
    df = df[df["datetime"] == "2025-05-13"]
    # print(df)
    update_latest_price(df)