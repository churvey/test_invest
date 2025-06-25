from trade.data.loader import QlibDataloader, FtDataloader
import os
import numpy as np
import pandas as pd

def volume_up(data):
    i = 20
    from numpy.lib.stride_tricks import sliding_window_view
    name = "bollinger"
    
    def _get(n="high"):
        return data[n] / data["close"]

    high = _get()
    low = _get("low")

    def _pos(name):
        p0 = low > data[name]
        p1 = (low <= data[name]) & (high > data[name])
        if "_d_" in name:
            p2 = (high <= data[name])
            return [p0, p1, p2]
        return [p0, p1]

    rs = []
    for n in [f"{name}_u_{i}", f"ma_{i}", f"{name}_d_{i}"]:
        rs += _pos(n)
    v = np.zeros(data["close"].shape)
    for p in rs:
        v = v * 2.0 + ((v == 0) & p)
     
    slide = 20
    open = data["open"]
    if len(open) <  slide:
        return {}   
        
    # low_pos = np.all(sliding_window_view(v, slide // 2) <= 4, axis=-1)
    low_pos = np.mean(sliding_window_view(v, slide), axis=-1) <= 4.8
    low_pos = np.concatenate(
        [[False] * (len(data["close"]) - len(low_pos)), low_pos]
    )
    
    candi = low_pos & (data["vma_5"] < 0.8) & (data["change"] > 0)
    
    
    if np.sum(candi) == 0 :
        return {}
    
    i_limit = 0.1
    d_limit = -0.05
    
    open_slide = sliding_window_view(open, slide)
    
    max_profile = sliding_window_view(data["high"], slide)[2:] / open[1:len(open_slide) - len(data["open"]) - 1].reshape([-1, 1]) - 1
    min_profile = sliding_window_view(data["low"], slide)[2:] / open[1:len(open_slide) - len(data["open"]) - 1].reshape([-1, 1]) - 1
    
    next_open = open[1:] / data["close"][:-1] -1
    
    
    
    # profile = (open_slide[2:] / open[1:len(open_slide) - len(data["open"]) - 1].reshape([-1, 1]) - 1)
    max_profile[np.isnan(max_profile)] = 0.0
    min_profile[np.isnan(min_profile)] = 0.0
    
    candi[:len(min_profile)] &= ~(np.any(np.isnan(min_profile) | np.isnan(max_profile), axis = -1))
    
    argmax = np.argmax(max_profile, axis = -1)
    argmin = np.argmin(min_profile, axis = -1)
    
    max = np.max(max_profile, axis = -1)
    min = np.min(min_profile, axis = -1)
    
   
    can_profile = (max >= i_limit) & ((min > d_limit) | (argmax < argmin))
    profile_v = np.where(
        can_profile, i_limit, np.where(min >= d_limit, min_profile[:, -1], d_limit)
    )
    
    can_profile = profile_v >= i_limit
    can_profile = np.concatenate(
       [can_profile, np.full(len(open) - len(can_profile), False)]
    )
    profile_v = np.concatenate(
       [profile_v, np.full(len(open) - len(profile_v), 0.0)]
    )
    
    next_open = np.concatenate(
       [next_open, np.full(len(open) - len(next_open), 0.0)]
    )
    max_profile = np.concatenate(
       [max_profile, np.full((len(open) - len(max_profile), max_profile.shape[-1]), 0.0)]
    )
    min_profile = np.concatenate(
       [min_profile, np.full((len(open) - len(min_profile), min_profile.shape[-1]), 0.0)]
    )

    pred = np.zeros(open.shape)
    pred[can_profile] = 1
    not_profile = ~can_profile
    pred[not_profile] = 0
    pred[~candi] = float("nan")
    

    assert pred.shape == data["close"].shape
    assert profile_v.shape == data["close"].shape, (profile_v.shape , data["close"].shape)
    
    return {
        "pred":pred,
        "profile_v":profile_v,
        "next_open":next_open,
        **{f"max_{i}":max_profile[:,i] for i in range(max_profile.shape[-1])},
        **{f"min_{i}":min_profile[:,i] for i in range(min_profile.shape[-1])}
    }
    
    
if __name__ == "__main__":
    # f1 = QlibDataloader(os.path.expanduser("~/output/qlib_bin"), [volume_up], extend_feature=["vma", "ma", "std", "z_bollinger"]).features
    f1 = FtDataloader("./qmt", [volume_up], extend_feature=["vma", "ma", "std", "z_bollinger"]).features
    
    print(f1)
    
    print(f1["y_pred"].mean())
    
    f1 = f1[f1["y_pred"] > 0]
    print(len(f1))
    # f1 = f1[f1["y_min_19"]< -0.05]
    # f1 = f1.head(1).to_dict()
    # for k,v in f1.items():
    #     print(k,"==>", v)

    
    print(f1)
    
    for i in range(20):
        k = f"y_min_{i}"
        k2 = f"y_max_{i}"
        print(k, "==>", f1[k].mean(), f1[k2].mean())
        
   
    for i in range(20):
        k = f"y_min_{i}"
        k2 = f"y_max_{i}"
        print(k, "==>", f1[k].quantile([0.1, 0.3, 0.5, 0.7,0.9]).to_numpy())
        # print(k2, "==>", f1[k2].quantile([0.1, 0.3, 0.5, 0.7,0.9]).to_numpy())
    
    print("next", "==>", f1["y_next_open"].quantile([0.1, 0.3, 0.5, 0.7,0.9]).to_numpy())
    print("next", "==>", f1["y_next_open"].mean())
    
    # f2 = FtDataloader("./tmp2", []).features
    # print(f2)
    
    # val = f2.iloc[-1].to_dict()
    # for k, v in val.items():
    #     print(k, "==>", v)
    # columns = [c for c in f2.columns if "bollinger" in c ]
    # columns = ["bollinger_d_20", "bollinger_u_60" ,"bollinger_d_60", "bollinger_u_60"]
    # columns = ["instrument","datetime"] + columns
    # print(f2[columns])