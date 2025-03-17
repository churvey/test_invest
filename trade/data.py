import torch
from torch import nn
import pandas as pd
import numpy as np
import os


def load_data_from_model(path="dnn.tensor2"):
    data = torch.load(path, map_location="cpu")
    return data


def load_from_processed_data(path="processed.csv", return_df=False):
    df = pd.read_csv(path)
    key = np.concatenate([df.iloc[1].to_numpy()[:2], df.iloc[0].to_numpy()[2:]])
    df = df.iloc[2:]
    df.columns = key

    features = "CNTD5,CNTD10,CNTD20,CNTD30,CNTD60,CNTN5,CNTN10,CNTN20,CNTN30,CNTN60,CNTP5,CNTP10,CNTP20,CNTP30,CNTP60,IMAX5,IMAX10,IMAX20,IMAX30,IMAX60,IMIN5,IMIN10,IMIN20,IMIN30,IMIN60,IMXD5,IMXD10,IMXD20,IMXD30,IMXD60,KLEN,KLOW,KLOW2,KMID,KMID2,KSFT,KSFT2,KUP,KUP2,MA5,MA10,MA20,MA30,MA60,MAX5,MAX10,MAX20,MAX30,MAX60,MIN5,MIN10,MIN20,MIN30,MIN60,QTLD5,QTLD10,QTLD20,QTLD30,QTLD60,QTLU5,QTLU10,QTLU20,QTLU30,QTLU60,RANK5,RANK10,RANK20,RANK30,RANK60,ROC5,ROC10,ROC20,ROC30,ROC60,STD5,STD10,STD20,STD30,STD60,RSV5,RSV10,RSV20,RSV30,RSV60"
    features += ",HIGH0,LOW0,OPEN0"

    features = features.split(",")
    all_columns = [p for p in df.columns if p != "VWAP0"]
    features = all_columns[2:-1]

    df = df[(df["datetime"] <= "2016-12-31") & (df["datetime"] >= "2008-01-01")]

    if return_df:
        return df

    df = df[all_columns].dropna()

    data = {}
    merge = False
    if merge:
        lable = "y"
        df0 = load_data_from_qlib(return_df=True)
        df = pd.merge(df, df0, how="left", on=["datetime", "instrument"])
    else:
        lable = df.columns[-1]

    print(f"label :{lable}")
    train = df[(df["datetime"] < "2014-12-31") & (df["datetime"] >= "2008-01-01")]
    valid = df[(df["datetime"] >= "2014-12-31") & (df["datetime"] <= "2016-12-31")]

    data["x"] = {
        "train": np.ascontiguousarray(train[features].to_numpy(), dtype=np.float32),
        "valid": np.ascontiguousarray(valid[features].to_numpy(), dtype=np.float32),
    }

    data["y"] = {
        "train": np.ascontiguousarray(
            train[lable].to_numpy(), dtype=np.float32
        ).reshape([-1, 1]),
        "valid": np.ascontiguousarray(
            valid[lable].to_numpy(), dtype=np.float32
        ).reshape([-1, 1]),
    }

    return data


def load_from_segs(segs=["train", "valid"]):
    data = {"x": {}, "y": {}}
    for seg in segs:
        df = pd.read_pickle(f"{seg}.csv")
        data["x"][seg] = np.ascontiguousarray(
            df["feature"].to_numpy(), dtype=np.float32
        )
        data["y"][seg] = np.ascontiguousarray(df["label"].to_numpy(), dtype=np.float32)
    return data


class Summarizer:
    def __init__(self, data):
        self.high = np.nanquantile(data, 0.99, axis = 0, keepdims = True )
        self.low  = np.nanquantile(data, 0.01, axis = 0, keepdims = True )
        tmp = np.clip(data, self.low, self.high)
        self.mean = np.nanmean(tmp)
        self.std = np.nanstd(tmp)
        # data = (data - mean) / std
        # return data
    
    def __call__(self, data):
        data = np.clip(data, self.low, self.high)
        data =  (data - self.mean) / self.std
        return data.astype("float32")
        
        

# def standarize(data):
#     high = np.nanquantile(data, 0.99, axis = -1)
#     low  = np.nanquantile(data, 0.01, axis = -1)
#     data = np.clip(low, high)
#     mean = np.nanmean(data)
#     std = np.nanstd(data)
#     data = (data - mean) / std
#     return data

def exetend_features(data):
    from .features import Feature

    return Feature(data=data)()


def get_inst(path, type="all"):
    instruments = os.path.join(path, "instruments", f"{type}.txt")
    ins = {}
    with open(instruments) as fp:
        for p in fp:
            tmp = p.strip().split("\t")
            # print(tmp)
            if tmp[0] not in ins:
                ins[tmp[0]] = tmp[1:]
            else:
                ins[tmp[0]] += tmp[1:]
    return ins


def load_data_from_qlib(
    path="/home/churvey/.qlib/qlib_data/cn_data",
    return_df=False,
    dates={
        "train": [
            "2008-01-01",
            "2014-12-31",
        ],
        "valid": ["2015-01-01", "2016-12-31"],
        "predict": ["2017-01-01", "2018-12-31"],
    },
    csi="csi300",
):

    ins = get_inst(path)
    csi = get_inst(path, csi)
    
    csi["SH000300"] = ins["SH000300"]
    
    days = os.path.join(path, "calendars", "day.txt")
    d = []
    with open(days) as fp:
        for p in fp:
            d.append(p.strip())
    from bisect import bisect_left, bisect_right

    def get_days(r):
        b = bisect_left(d, r[0])
        if d[b] != r[0]:
            print("begin not in be careful", r[0])
            b += 1
        e = bisect_right(d, r[1])
        if d[e - 1] != r[1]:
            print("end not in be careful", r[1])
            e -= 1
        return d[b:e]

    def list_features(k):
        files = os.listdir(os.path.join(path, "features", k))
        return files

    index = 0
    keys = ["change", "close", "factor", "high", "low", "open", "volume"]
    arr_dict = {k: [] for k in dates}
    index_dict = {k: [] for k in dates}
    origin_data = {}
    dfs = []
    for k, v in ins.items():
        # if k not in csi or k != "SH600015":
        # if index > 100:
        #     break
        # if k not in csi or k != "SH000300":
        if k not in csi:
            continue
        data = {}
        day = np.array(get_days(v))
        if not len(day):
            print(f"empty: {v}")
            continue

        data[k] = {}
        for f in list_features(k):
            # if f.split(".")[0] not in keys:
            #     continue
            f1 = np.fromfile(os.path.join(path, "features", k, f), dtype=np.float32)
            data[k][f.split(".")[0]] = f1[1:]
            # if f.split(".")[0] not in keys:
            #     print(f, f1)

        if data[k]["close"].shape[0] <= 2 or data[k]["close"].shape[0] != len(day):
            print(f'ignore { data[k]["close"].shape[0] } vs {len(day)}')
            continue

        origin_data[k] = {f: data[k][f] for f in keys}
        origin_data[k]["datetime"] = np.array(day)
        
        # if k == "SH601989":
        #     print(k, day[-1], day.shape, data[k]["close"].shape)
        
        data[k]["y"] = np.concatenate(
            [data[k]["close"][2:] / data[k]["close"][1:-1] - 1, [float("nan")] * 2]
        )
        data[k] = exetend_features(data[k])
        if index % 100 == 0:
            print(index, k, data[k].keys(), data[k]["close"].shape)

        # print(f"{k} {len(csi[k])}")
        
        x = None
        for i in range(0, len(csi[k]), 2):
            for d_k, d_v in dates.items():
                b = csi[k][i]
                e = csi[k][i + 1]
                begin, end = d_v
                b = b if b > begin else begin
                e = e if e < end else end
                begin_index = bisect_left(day, b)
                end_index = bisect_left(day, e)
                # if index % 100 == 0:
                #     print(d_k, begin_index, end_index, d_v, csi[k], [b, e])
                if return_df:
                    data[k]["datetime"] = day
                    data[k]["instrument"] = np.array([k] * len(day))
                    # data[k] = {k:v for k,v in data[k].items() if k not in keys}
                    df = pd.DataFrame.from_dict(data[k])
                    df = df[(df["datetime"] <= e) & (df["datetime"] >= b)]
                    dfs.append(df)
                else:
                    if x is None:
                        l = [
                            data[k][f] for f in data[k].keys() if f != "y" and f not in ["volume", "amount"]
                        ] + [data[k]["y"]]
                        index_d = [day, np.array([k] * len(day))]
                        try:
                            x = np.stack(l, axis=-1)
                            index_d = np.stack(index_d, axis=-1)
                            assert x.shape[0] == index_d.shape[0]
                        except BaseException as e:
                            s = [k.shape for k in l]
                            print(s)
                            raise e
                    arr_dict[d_k].append(x[begin_index : end_index + 1])
                    index_dict[d_k].append(index_d[begin_index : end_index + 1])
                    # if k == "SH601989":
                    #      print(d_k, "SH601989", index_dict[d_k][-1][-1])
        index += 1
    if return_df:
        return pd.concat(dfs)

    rs = {
        "x": {},
        "y": {},
        "datetime": {},
        "instrument": {},
        "origin_data": origin_data,
    }

    for k, v in arr_dict.items():
        v = np.concatenate(v)
        print(f"before nan {v.shape}")
        notna = ~np.isnan(v).any(axis=1)
        v = v[notna, :]
        print(f"after nan {v.shape}")
        # print(v)
        index = np.concatenate(index_dict[k])
        index = index[notna, :]
        # print(index)
        rs["x"][k] = np.ascontiguousarray(v[:, :-1], dtype=np.float32)
        rs["y"][k] = np.ascontiguousarray(v[:, -1:], dtype=np.float32)
        rs["datetime"][k] = index[:, 0]
        rs["instrument"][k] = index[:, 1]
        
    summarizer = Summarizer(rs["x"]["train"])
    for k, v in rs["x"].items():
        rs["x"][k] = summarizer(v)
        
    # a = rs["datetime"]["predict"] <= "2024-09-02" 
    # a1 = rs["datetime"]["predict"] >= "2024-08-28"
    
    # a = a&a1
    
    # b = rs["instrument"]["predict"] == "SH601989"
    
    # c = rs["datetime"]["predict"] == "2024-09-03"
    
    # print(
    #     "rs",
    #     np.sum(a),
    #     np.sum(b),
    #     np.sum(c),
    #     np.any(a & b),
    #     np.any(b & c)
    # )
    
    # print("instruct", rs["instrument"]["predict"][c].tolist())
    

    return rs

    # train = np.concatenate(train)
    # train = train[~np.isnan(train).any(axis=1), :]
    # validate = np.concatenate(validate)
    # validate = validate[~np.isnan(validate).any(axis=1), :]

    # y_all = np.concatenate([train[:, -1:], validate[:, -1:]])
    # y_mean = np.mean(y_all)
    # y_std = np.std(y_all)

    # return {
    #     "x": {
    #         "train": np.ascontiguousarray(train[:, :-1], dtype=np.float32),
    #         "valid": np.ascontiguousarray(validate[:, :-1], dtype=np.float32),
    #     },
    #     "y": {
    #         "train": (np.ascontiguousarray(train[:, -1:], dtype=np.float32) - y_mean)
    #         / y_std,
    #         "valid": (np.ascontiguousarray(validate[:, -1:], dtype=np.float32) - y_mean)
    #         / y_std,
    #     },
    # }


def cmp_data():
    dates = {
        "train": [
            "2008-01-01",
            "2014-12-31",
        ],
        "valid": ["2015-01-01", "2016-12-31"],
    }
    begin, end = None, None
    for k, v in dates.items():
        begin = v[0] if not begin or v[0] < begin else begin
        end = v[1] if not end or v[1] > end else end

    df0 = load_data_from_qlib(return_df=True, dates=dates)

    close = df0["close"].to_numpy(dtype="float")

    dfk = df0.dropna()
    print(len(dfk))

    print("close nan", np.sum(np.isnan(close)))
    # 1/0
    print(df0)
    df0["hh"] = df0["datetime"]
    path = "loaded_l.csv"
    df = load_from_processed_data(path, return_df=True)
    print(f"len left {len(df0)}")
    print(f"len right {len(df)}")
    dd = pd.merge(df0, df, how="inner", on=["datetime", "instrument"], sort=True)
    print(dd)

    tmp = {}
    already_in = []
    for column in dd.columns:
        tmp[column] = {}
        rs = tmp[column]
        # for column in ["ma_5"]:
        if column != column.upper():
            c2 = column.upper().replace("_", "")
            if c2 in dd.columns:
                already_in.append(c2)
                v = dd[column].to_numpy(dtype="float")
                v2 = dd[c2].to_numpy(dtype="float")
                v_i = np.isnan(v)
                v2_i = np.isnan(v2)
                rs["c"] = [np.sum(v_i), np.sum(v2_i)]

                # print(dd[["datetime", "instrument", column, c2, "close"]][v_i])
                # print(dd[["datetime", "instrument", column, c2, "close"]][v2_i])

                i = ~(v_i | v2_i)
                v = v[i]
                v2 = v2[i]
                try:
                    np.testing.assert_allclose(v, v2, 1e-5, 1e-5)
                    rs["eq"] = True
                except Exception as e:
                    rs["eq"] = False
                    print(e)
                    print(f"###########{column}###########")
            else:
                print("not in ", column)

    for k, v in tmp.items():
        if k not in already_in:
            print(f"{k}==>{v}")


if __name__ == "__main__":
    cmp_data()
