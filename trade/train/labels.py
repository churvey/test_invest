import numpy as np

def one_into_two(data):
        limit = 0.099
        if data["instrument"][0][2:5] in ["300", "688"]:
            limit = 0.199
        if data["instrument"][0][2:3] in ["8"]:
            limit = 0.299
        rs = np.zeros(data["close"].shape)
        up_flag = (data["change"] >= limit) & (data["high"] - data["close"] < 1e-5)
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

        return {
            "pred": pred,
        }
        
def down_to_up(data):
    i = 20
    name = "bollinger"
    # down = data["bollinger_d_20"]
    # up = data["bollinger_u_20"]
    
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
        v = v * 2 + ((v == 0) & p)
    print(v.min(), v.max())
    
    down = (v == 4)
    from numpy.lib.stride_tricks import sliding_window_view
    
    v_slide = sliding_window_view(v, 20)
    
    open_slide = sliding_window_view(data["open"], 20)
    profile = (open_slide[1:] - data["open"][:len(open_slide) - len(data["open"]) - 1]) >= 0.04
    profile &= (v_slide[1:] > 4)
    
    no_upper_before_down = np.all(v_slide <= 4, axis = -1)
    pad = np.full(len(down) - len(no_upper_before_down), False)
    # print(f"down: {down.shape}, v_slide:{v_slide.shape}, no_upper:{no_upper_before_down.shape} {pad.shape}")
    no_upper_before_down = np.concatenate(
       [pad , no_upper_before_down]
    )
    
    # print(f"after concat down: {down.shape}, v_slide:{v_slide.shape}, no_upper:{no_upper_before_down.shape}")
    # print(no_upper_before_down)
    candi = down & no_upper_before_down
    
    upper_after_down = np.any(v_slide > 4, axis = -1)
    upper_after_down = np.concatenate(
       [upper_after_down, np.full(len(down) - len(upper_after_down) + 1, False)]
    )[1:].astype(bool)
    
    pred = np.zeros(v.shape)
    pred[:] = float("nan")
    pred[candi & upper_after_down ] = 1
    pred[candi & ~upper_after_down] = 0
    
    return {
        "pred":pred
    }
    