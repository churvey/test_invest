import numpy as np
from tqdm import tqdm
import logging
import trade_cpp
from ctypes import *
import math


class Sampler:

    def __init__(self, loader , day_range=None):
        print(f"loader features:{len(loader.features)}")
        self.label_name = "y_pred"
        features = loader.features
        if day_range:
            begin, end = day_range
            features = features[
                (features["datetime"] >= begin) & (features["datetime"] < end)
            ]
            print(f"after select {day_range} features:{len(features)}")
        self.labels = loader.labels
        self.days = loader.days

        columns = loader.indices + loader.feature_columns + loader.labels
        features = features[columns]
        l = len(features)
        na_count = features.isna().sum() / l
        t = dict(zip(features.columns, na_count.to_numpy()))
        logging.info(f"na_count { {k: v for k, v in t.items() if v > 0.05} }")
        features = features.dropna()
        logging.info(f"features:{len(features)}")

        self._feature_columns = loader.feature_columns
        self.features_np = np.ascontiguousarray(
            features[self._feature_columns].to_numpy(dtype="float32")
        )
        self.label_np = features[loader.labels].to_numpy(dtype="float32")
        self.datetime_np = features["datetime"].to_numpy()
        self.datetime = np.sort(np.unique(self.datetime_np))
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
            print(label)
            return (label * label)

    def weight(self, features):
        w = np.arange(1, len(self.days) + 1, dtype="float64")
        weight_mapping = dict(zip(self.days, w))
        weight = features["datetime"].map(weight_mapping)
        # weight = weight * self.label_weight()
        weight = weight / weight.mean()
        weight = weight.to_numpy(dtype="float32").reshape([-1])
        assert weight.shape[0] == self.features_np.shape[0]
        return weight

    def sample(self, batch_size, i):
        return self.index[batch_size * i : batch_size * (i + 1)]

    def to_numpy(self, index):
        return {
            "x": self.features_np[index, :],
            **{
                self.labels[i]: self.label_np[index, i : i + 1]
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
            date = self.datetime[i]
            samples = self.datetime_np == date
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
            return self.datetime.shape[0]


class SamplersCpp(Sampler):

    def __init__(self, loader, day_range=None):
        super(SamplersCpp, self).__init__(loader, day_range)
        self.data = {
            "x": self.features_np,
            **{
                self.labels[i]: np.ascontiguousarray(self.label_np[:, i : i + 1])
                for i in range(len(self.labels))
            },
        }

    #         print(f"flags {self.features_np.flags} {(self.features_np.ctypes.data)} {self.features_np.dtype}")
    #         print( {k: v.ctypes.data for k, v in self.data.items()}, self)
    #         print( {k: cast(v.ctypes.data, POINTER(c_float))[0] for k, v in self.data.items()}, self)
    #         print( {k: v[0,0] for k, v in self.data.items()}, self)
    #         print(trade_cpp.get_array_ptr(self.data))

    def iter(self, batch_size, phase="train", ratio=1.0):
        index = np.arange(self.features_np.shape[0])
        if phase == "train":
            n = int(index.shape[0] * ratio)
            weight = self.w.astype("float64")
            weight *= self.label_weight()
            weight /= weight.sum()
            index = np.random.choice(index, n, replace=True, p=weight)
        elif phase == "valid":
            batch_size = int(8096 * 16)
            pass
        else:
            return super(SamplersCpp, self).iter(batch_size, phase, ratio)
        sampler = trade_cpp.NumpyDictSampler(self.data, batch_size, index.tolist())
        return tqdm(sampler, total=self.len(batch_size, phase, ratio))
