from trade.data.loader import QlibDataloader, FtDataloader
import os
import numpy as np


if __name__ == "__main__":
    # f1 = QlibDataloader(os.path.expanduser("~/output/qlib_bin"), [], extend_feature=False).features
    f2 = FtDataloader("./tmp2", []).features
    print(f2)
    
    val = f2.iloc[-1].to_dict()
    for k, v in val.items():
        print(k, "==>", v)
    # columns = [c for c in f2.columns if "bollinger" in c ]
    # columns = ["bollinger_d_20", "bollinger_u_60" ,"bollinger_d_60", "bollinger_u_60"]
    # columns = ["instrument","datetime"] + columns
    # print(f2[columns])