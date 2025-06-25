from trade.data.loader import QlibDataloader, FtDataloader
import os
import numpy as np
import pandas as pd


def test(exp, config):

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

        # low_pos = np.all(sliding_window_view(v, int(config["low"])) <= 4, axis=-1)

        # print(config["low"])
        low_pos = np.mean(sliding_window_view(v, slide), axis=-1) <= int(config["low"])
        low_pos = np.concatenate(
            [[False] * (len(data["close"]) - len(low_pos)), low_pos]
        )

        candi = low_pos & (data["vma_5"] <= config["vma"]) & (data["change"] > 0)

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

        return {f"pred_{exp}": pred, f"profile_v{exp}": profile_v}

    return volume_up


if __name__ == "__main__":
    # config = [
    #     {"slide": 20, "low": 4.8, "limit": 0.1},
    # ]

    slides = [5, 10, 20]
    lows = [4, 4.8, 5.6]
    limit = [0.1, 0.06, 0.04, 0.02]
    vma = [0.6, 0.7, 0.8]

    def configs():
        for s in slides:
            for l in lows:
                for li in limit:
                    for v in vma:
                        yield {"slide": s, "low": l, "limit": li, "vma":v}

    config = list(configs())
    print(config)
    f1 = FtDataloader(
        "./qmt",
        [test(i, config[i]) for i in range(len(config))],
        extend_feature=["vma", "ma", "std", "z_bollinger"],
    ).features.sort_values(["datetime"])

    f1 = f1[f1["datetime"] <= "2025-05-09"]
    print(f1)

    # print(f1[f1["datetime"] <= "2025-04-01"][["datetime","instrument","y_pred", "y_profile_v"]].tail(20))
    rs = []
    for i, c in enumerate(config):
        c_str = ""
        for k, v in c.items():
            c_str += f"{k}:{v}__"
        c_str = c_str[:-2]
        rs.append(
            (f1[f"y_pred_{i}"].dropna().mean(), len(f1[f"y_pred_{i}"].dropna()), c_str)
        )
        # print(f1[f"y_pred_{i}"].dropna().mean(), len(f1[f"y_pred_{i}"].dropna()), c)

    for r in reversed(sorted(rs)):
        print(r)

    # print(f1[f1["y_pred"] > 0][["datetime","instrument"]].tail(20))

    # 12 {'limit': 0.05, 'low': 20.0, 'slide': 10.0} 0.3974317136050648 61286
    # 8 {'limit': 0.05, 'low': 15.0, 'slide': 10.0} 0.3879064976371407 110671

    # f2 = FtDataloader("./tmp2", []).features
    # print(f2)

    # val = f2.iloc[-1].to_dict()
    # for k, v in val.items():
    #     print(k, "==>", v)
    # columns = [c for c in f2.columns if "bollinger" in c ]
    # columns = ["bollinger_d_20", "bollinger_u_60" ,"bollinger_d_60", "bollinger_u_60"]
    # columns = ["instrument","datetime"] + columns
    # print(f2[columns])
