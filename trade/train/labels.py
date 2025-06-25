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
    # print(v.min(), v.max())
    
    down = (v == 4)
    from numpy.lib.stride_tricks import sliding_window_view
    
    slide = 20
    v_slide = sliding_window_view(v, slide)
    
    open_slide = sliding_window_view(data["close"], slide)
    profile = (open_slide[1:] - data["close"][:len(open_slide) - len(data["close"]) - 1].reshape([-1, 1]))
    argmax = np.argmax(profile, axis = -1)
    argmin = np.argmin(profile, axis = -1)
    
    max = np.max(profile, axis = -1)
    min = np.min(profile, axis = -1)
    
    print(f"profile {profile.max()} {profile.min()} {profile.mean()} {np.quantile(profile, 0.99)} {np.quantile(profile, 0.5)} ")
    
    i_limit = 0.01
    d_limit = -0.005
    # 1/0
   
    # print(f"shapes {(max >= i_limit).shape} {(min >= d_limit).shape } {(argmax > argmin).shape}")
    # print(f"values {profile[0]} {profile[:,argmax][0]} {argmax[0]}")
    can_profile = (max >= i_limit) & ((min >= d_limit) | (argmax > argmin))
    profile_v = np.where(
        can_profile, i_limit, np.where(min >= d_limit, profile[:, -1], d_limit)
    )
    
    can_profile = profile_v >= i_limit / 2
    can_profile = np.concatenate(
       [can_profile, np.full(len(down) - len(can_profile), False)]
    )
    profile_v = np.concatenate(
       [profile_v, np.full(len(down) - len(profile_v), 0.0)]
    )
    # profile &= (v_slide[1:] > 4)
    
    no_upper_before_down = np.all(v_slide <= 4, axis = -1)
    pad = np.full(len(down) - len(no_upper_before_down), False)
    # print(f"down: {down.shape}, v_slide:{v_slide.shape}, no_upper:{no_upper_before_down.shape} {pad.shape}")
    no_upper_before_down = np.concatenate(
       [pad , no_upper_before_down]
    )
    
    # print(f"after concat down: {down.shape}, v_slide:{v_slide.shape}, no_upper:{no_upper_before_down.shape}")
    # print(no_upper_before_down)
    candi = down & no_upper_before_down
    
    # upper_after_down = np.any(v_slide > 4, axis = -1)
    # upper_after_down = np.concatenate(
    #    [upper_after_down, np.full(len(down) - len(upper_after_down) + 1, False)]
    # )[1:].astype(bool)
    
    # candi = data["std_5"] >= i_limit / 2
    
    
    
    pred = np.zeros(v.shape)
    # pred[:] = float("nan")
    pred[can_profile] = 1
    not_profile = ~can_profile
    # a = (np.random.rand(len(can_profile)) <= np.mean(can_profile.astype('float32')) * 2)
    # not_profile &= a
    pred[not_profile] = 0
    pred[~candi] = float("nan")
    
    print(f"pred mean {np.nanmean(pred)} {np.nansum(pred)}/{len(pred)} {np.mean(can_profile.astype('float32'))} {np.sum(candi)}")
    
    z_pos = {int(i):(int(np.sum(v[can_profile]==i)), int(np.sum(v == i)))  for i in np.unique(v)}
    
    print("z_pos", z_pos)
    
    assert pred.shape == data["close"].shape
    assert profile_v.shape == data["close"].shape, (profile_v.shape , data["close"].shape)
    
    return {
        "pred":pred,
        "profile_v":profile_v,
        "z_pos":v,
    }
    
    


def volume_up_wrapper(config):

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
                p2 = high <= data[name]
                return [p0, p1, p2]
            return [p0, p1]

        rs = []
        for n in [f"{name}_u_{i}", f"ma_{i}", f"{name}_d_{i}"]:
            rs += _pos(n)
        v = np.zeros(data["close"].shape)
        for p in rs:
            v = v * 2.0 + ((v == 0) & p)

        slide = int(config["slide"])
        open = data["open"]
        if len(open) < slide:
            return {}

        low_pos = np.all(sliding_window_view(v, slide // 2) <= 4, axis=-1)

        # print(config["low"])
        # low_pos = np.mean(sliding_window_view(v, slide // 2), axis=-1) <= int(config["low"])
        low_pos = np.concatenate(
            [[False] * (len(data["close"]) - len(low_pos)), low_pos]
        )

        # candi = low_pos & (data["vma_5"] <= config["vma"]) & (data["change"] > 0)

        candi =  (data["vma_5"] <= config["vma"]) & (data["change"] > 0)

        if np.sum(candi) == 0:
            return {}

        i_limit = config["limit"]
        d_limit = -i_limit / 2

        open_slide = sliding_window_view(open, slide)

        max_profile = (
            sliding_window_view(data["high"], slide)[2:]
            / open[1 : len(open_slide) - len(data["open"]) - 1].reshape([-1, 1])
            - 1
        )
        min_profile = (
            sliding_window_view(data["low"], slide)[2:]
            / open[1 : len(open_slide) - len(data["open"]) - 1].reshape([-1, 1])
            - 1
        )

        # profile = (open_slide[2:] / open[1:len(open_slide) - len(data["open"]) - 1].reshape([-1, 1]) - 1)
        max_profile[np.isnan(max_profile)] = 0.0
        min_profile[np.isnan(min_profile)] = 0.0

        candi[: len(min_profile)] &= ~(
            np.any(np.isnan(min_profile) | np.isnan(max_profile), axis=-1)
        )

        argmax = np.argmax(max_profile, axis=-1)
        argmin = np.argmin(min_profile, axis=-1)

        max = np.max(max_profile, axis=-1)
        min = np.min(min_profile, axis=-1)

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

        pred = np.zeros(open.shape)
        pred[can_profile] = 1
        not_profile = ~can_profile
        pred[not_profile] = 0
        pred[~candi] = float("nan")

        assert pred.shape == data["close"].shape
        assert profile_v.shape == data["close"].shape, (
            profile_v.shape,
            data["close"].shape,
        )

        return {f"pred": pred, f"profile_v": profile_v, "z_pos":v}

    return volume_up