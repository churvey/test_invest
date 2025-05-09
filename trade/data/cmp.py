from trade.data.loader import QlibDataloader, FtDataloader
import os
import numpy as np


if __name__ == "__main__":
    f1 = QlibDataloader(os.path.expanduser("~/output/qlib_bin"), [], extend_feature=False).features
    f2 = FtDataloader("./qmt", [], extend_feature=False).features
    
    f1["instrument"] = f1["instrument"].str[2:]
    
    for k in [ "low","high", "close", "open"]:
        f1[k] = (f1[k] / f1["factor"]).round(2)
    
    f1["change"] *= 100
    f1["volume"] *= f1["factor"]
    
    f2["instrument"] = f2["instrument"].str[:6]
    f2["datetime"] = f2["datetime"].str[:len("2025-01-02")]
    
    f1 = f1[(f1["datetime"]<="2025") & (f1["datetime"]>="2024")][["instrument","datetime", "low","high", "change", "close", "open", "volume"]]
    f2 = f2[(f2["datetime"]<="2025") & (f2["datetime"]>="2024")][["instrument","datetime", "low","high", "change", "close", "open", "volume"]]

    f1 = f1.sort_values(["instrument","datetime"]).reset_index(drop=True)
    f2 = f2.sort_values(["instrument","datetime"]).reset_index(drop=True)
    
    f3 = f1.merge(f2, on=["instrument", "datetime"])
    # f3["v2"] = f3["volume_x"] / f3["volume_y"]
    
    
    index = f3["low_x"] != f3["low_x"]
    
     
    
    print(f3[index])
    
    # print(f1)
    # print(f2)
    
    for k in [ "low","high", "close", "open"]:
        v1 = f1[k].to_numpy().round(decimals=2)
        v2 = f2[k].to_numpy()
        np.testing.assert_allclose(v1, v2)
        print(f"passed {k}")
        
#          adjclose       vwap        volume    factor        low       high    change      close       open        amount    datetime instrument
# 4858   569.859985  19.067497  2.579018e+05  1.840783  18.573500  19.825232 -0.052190  18.720762  19.788416  4.917542e+05  2025-01-02   SH600588

#      instrument             datetime   open  close   high    low  volume    change
# 4133  600588.SH  2025-01-02 00:00:00  10.75  10.17  10.77  10.09  474741 -5.219012