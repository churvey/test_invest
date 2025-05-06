import numpy as np
from tqdm import tqdm
import logging
import trade_cpp
from ctypes import *
import math
import torch
from numpy.lib.stride_tricks import sliding_window_view


def numpy_unfold(x, dimension, size, step):
    # 滑动窗口视图
    window_view = sliding_window_view(x, window_shape=size, axis=dimension)
    # 根据步长选择窗口
    x = window_view[..., ::step, :].transpose(0, 1, 3, 2)
    x = x.reshape([-1, *x.shape[-2:]])
    return x


def torch_unfold(x, dim, size, step):
    x = x.unfold(dim, size, step).permute(0, 1, 3, 2)
    x = x.reshape([-1, *x.shape[-2:]])
    return x


class Sampler:

    def __init__(self, loader, day_range=None, seq_col="instrument"):
        print(f"loader features:{len(loader.features)}")
        self.label_name = "y_pred"
        features = loader.features
        self.seq_col = seq_col
        # features.
        if day_range:
            begin, end = day_range
            features = features[
                (features["datetime"] >= begin) & (features["datetime"] < end)
            ]
            print(f"after select {day_range} features:{len(features)}")
        self.labels = loader.labels
        self.days = loader.days
        self._feature_columns = loader.feature_columns
        columns = loader.indices + loader.feature_columns + loader.labels
        features = features[columns]
        l = len(features)
        na_count = features.isna().sum() / l
        t = dict(zip(features.columns, na_count.to_numpy()))
        logging.info(f"na_count { {k: v for k, v in t.items() if v > 0.05} }")
        if not seq_col:
            features = features.dropna()
            shape = [len(features), -1]
            self.seq_len = 1
        else:
            assert seq_col in loader.indices
            index_col = [p for p in loader.indices if p != seq_col][0]
            index = features[[index_col]].drop_duplicates()
            indices = index.merge(features[[seq_col]].drop_duplicates(), how="cross")
            # shape = [len(index), -1]
            
            self.seq_len = len(indices) // len(index)
            features = features.merge(indices, how="right", on=loader.indices)
            features = features.sort_values([index_col, seq_col])
            if self.seq_col == "instrument":
                features[self._feature_columns] = features[
                    self._feature_columns
                ].fillna(0)
            print(features)
            shape = [len(features), -1]

        self.features_np = np.ascontiguousarray(
            features[self._feature_columns].to_numpy(dtype="float32")
        ).reshape(shape)
        print("features_np", self.features_np.shape)
        self.label_np = features[loader.labels].to_numpy(dtype="float32").reshape(shape)
        self.datetime_np = features["datetime"].to_numpy().reshape(shape)
        self.instrument_np = features["instrument"].to_numpy().reshape(shape)
        self.datetime = np.sort(np.unique(self.datetime_np))
        self.instrument = np.sort(np.unique(self.instrument_np))
        self.w = self.weight(features)
        self.index = self.init_seed()

    def feature_columns(self):
        return self._feature_columns

    def label_weight(self):
        label = self.label_np[:, self.labels.index(self.label_name)]
        if "cls" in self.label_name:
            # label = self.label_np[:, self.labels.index(label)]
            label_unique = np.sort(np.unique(label))
            weight = {
                l: math.sqrt(1 - (label == l).sum() / self.label_np.shape[0])
                for l in label_unique
            }
            weight_s = np.zeros_like(label)
            for l in label_unique:
                weight_s[label == l] = weight[l]
            return weight_s
        else:
            # print(label)
            # return (label * label)
            # return np.abs(label)
            return 1
            # weight = np.abs(label)
            # bins = 200

            # bucket = np.arange(bins) / bins

            # def get_value(w):
            #     for i in range(len(bucket)):
            #         if bucket[i] > w:
            #             return bucket[i]
            #     return 1

            # weight = np.array([
            #     get_value(w) for w in weight
            # ])
            # return weight

    def weight(self, features):
        if self.seq_col:
            return 1
        w = np.arange(1, len(self.days) + 1, dtype="float64")
        weight_mapping = dict(zip(self.days, w))
        weight = features["datetime"].map(weight_mapping)
        # weight = weight * self.label_weight()
        weight = weight / weight.mean()
        weight = weight.to_numpy(dtype="float32").reshape([-1])
        assert weight.shape[0] == self.features_np.shape[0]
        return weight.astype("float64")

    def sample(self, batch_size, i):
        return self.index[batch_size * i : batch_size * (i + 1)]

    def to_numpy(self, index):
        import pdb

        # pdb.set_trace()
        return {
            "x": self.features_np[index, ...],
            "datetime": self.datetime_np[index, ...],
            "instrument": self.instrument_np[index, ...],
            **{
                self.labels[i]: self.label_np[
                    index, ..., i * self.seq_len : (i + 1) * self.seq_len
                ]
                for i in range(len(self.labels))
            },
        }

    def init_seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.index = np.arange(self.features_np.shape[0])
        np.random.shuffle(self.index)
        return self.index

    def gen(self, batch_size, i, phase="train"):
        if phase == "train":
            return self.to_numpy(self.sample(batch_size, i))
        elif phase == "valid":
            return self.to_numpy(self.sample(int(1e8), i))
        else:
            if self.seq_col != "datetime":
                date = self.datetime[i]
                samples = (
                    self.datetime_np.reshape([len(self.datetime_np), -1])[:, -1] == date
                )
            else:
                ins = self.instrument[i]
                samples = (
                    self.instrument_np.reshape([len(self.instrument_np), -1])[:, -1]
                    == ins
                )
            return self.to_numpy(samples)

    def iter(self, batch_size, phase="train", ratio=1.0):
        total_len = self.len(batch_size, phase, ratio)

        def gen():
            for i in range(total_len):
                yield self.gen(batch_size, i, phase)

        return tqdm(gen(), total=total_len)

    def len(self, batch_size, phase, ratio=1.0):
        if phase == "train":
            l = self.features_np.shape[0]
            return (int(l * ratio) + batch_size - 1) // batch_size
        elif phase == "valid":
            return 1
        else:
            return (
                self.datetime.shape[0]
                if self.seq_col != "datetime"
                else self.instrument.shape[0]
            )


class SamplersCpp(Sampler):

    def __init__(self, loader, day_range=None, seq_col="instrument", max_seqlen= 256):
        super(SamplersCpp, self).__init__(loader, day_range, seq_col)
        self.max_seqlen = max_seqlen
        self.data = {
            "x": self.features_np,
            **{
                self.labels[i]: np.ascontiguousarray(
                    self.label_np[:, i * self.seq_len : (i + 1) * self.seq_len]
                )
                for i in range(len(self.labels))
            },
        }

    #         print(f"flags {self.features_np.flags} {(self.features_np.ctypes.data)} {self.features_np.dtype}")
    #         print( {k: v.ctypes.data for k, v in self.data.items()}, self)
    #         print( {k: cast(v.ctypes.data, POINTER(c_float))[0] for k, v in self.data.items()}, self)
    #         print( {k: v[0,0] for k, v in self.data.items()}, self)
    #         print(trade_cpp.get_array_ptr(self.data))

    def wrapper_iter(self, iter):
        if self.seq_col != "datetime":
            return iter

        def g():
            for data in iter:
                batch_size = len(data["x"])
                def unfold(x, dim):
                    x = x.reshape(len(x), -1, dim)  # [N, T, F]
                    # print("x.shape", x.shape)
                    x = (
                        numpy_unfold(x, 1, self.max_seqlen, 1)
                        if not isinstance(x, np.ndarray)
                        else numpy_unfold(x, 1, self.max_seqlen, 1)
                    )
                    return x

                # x = unfold(x, self.len(self.features))
                for k in data.keys():
                    # print(f"k:{k}")
                    data[k] = unfold(
                        data[k], 1 if k != "x" else len(self.feature_columns())
                    )
                    # if k == "datetime" or k == "instrument":
                    #     print(k, data[k][-1].reshape([-1]))

                shapes = {k: v.shape for k, v in data.items()}
                print("shapes", shapes)
                x = data["x"]
                y = data["y_pred"]
                
                if not isinstance(x, np.ndarray):
                    x = x.numpy()
                    y = y.numpy()
               
                valid_x = ~np.any(np.any(np.isnan(x), axis=-1), axis=-1)
                valid_y = ~np.isnan(y[:, -1, -1])
                valid = valid_x & valid_y
                # print("valid", valid.shape)
                total_size = np.sum(valid)
                print("valid count", total_size, valid.shape, batch_size)
                

                
                new_data = {
                    k: v[valid] if k == "x" else v[valid][:, -1, ...]
                    for k, v in data.items()
                }
                if total_size > 1:
                    if batch_size == 1:
                        yield new_data
                    else:
                        # if batch_size < 16 and total_size >= 16:
                        #     batch_size = 16
                        for i in range(total_size // batch_size):
                            new_data_i = {
                                k: v[i * batch_size + total_size % batch_size : (i + 1) * batch_size + total_size % batch_size]
                                for k, v in new_data.items()
                            }
                            yield new_data_i

        return tqdm(g(), total=iter.total * len(self.datetime))

    def iter(self, batch_size, phase="train", ratio=1.0):
        seqlen = 1
        index = np.arange(self.features_np.shape[0])
        
        if self.seq_col == "instrument":
            seqlen = len(self.instrument)
            index = np.arange(self.datetime.shape[0]) * seqlen
        elif self.seq_col == "datetime":
            seqlen = self.max_seqlen
            index = index[index % len(self.datetime) <= len(self.datetime) - seqlen]
        if phase == "train":
            n = int(index.shape[0] * ratio)
            weight = (np.zeros(index.shape) + 1) * self.w
            weight *= self.label_weight()
            weight /= np.sum(weight)
            if self.seq_col == "datetime":
                n = n * len(self.datetime) // (seqlen * len(self.datetime - seqlen))
            index = np.random.choice(index, n, replace=True, p=weight)
        elif phase == "valid":
            batch_size = int(8096 * 16)
            pass
        else:
            batch_size = int(8096 * 16)
        #     return super(SamplersCpp, self).iter(batch_size, phase, ratio)
        sampler = trade_cpp.NumpyDictSampler(self.data, batch_size, index.tolist(), seqlen)

        def sample():
            for data in sampler:

                indices = data.pop("indices").numpy()
                yield {
                    "datetime": self.datetime_np[indices],
                    "instrument": self.instrument_np[indices],
                    **data,
                }
        total_len = (len(index) + batch_size - 1) // batch_size
        return tqdm(sample(), total=total_len)

    # def iter(self, batch_size, phase="train", ratio=1.0):
    #     return self.wrapper_iter(self.iter_internal(batch_size, phase, ratio))
