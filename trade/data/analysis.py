
import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import cloudpickle

from torch.utils.tensorboard import SummaryWriter
import datetime

from trade.data.loader import QlibDataloader,FtDataloader
# from trade.data.sampler import *
# from trade.model.reg_dnn import RegDNN
# from trade.model.cls_dnn import ClsDNN
# from trade.train.utils import *
import numpy as np
import torch


def label_gen(data):
    l = data["close"].shape[0]
    return {
        "pred": np.concatenate(
            [np.log(data["open"][1:] / data["close"][:-1]), [float("nan")] * 1]
        )[:l],
        # "pred": np.concatenate(
        #     [np.abs(np.log(data["open"][1:] / data["close"][:-1])), [float("nan")] * 1]
        # )[:l],
        # "cls": np.concatenate([data["limit_flag"][1:], [float("nan")]])[:l],
        #  "cls": get_label(data),
    }

def plot_label(label_gen):
    loader = QlibDataloader(os.path.expanduser("~/output/qlib_bin"), [label_gen])
    data = loader.features.dropna()["y_pred"].to_numpy()
    
    mu = np.mean(data)
    sigma = np.std(data)
    # l = loader.features["y_pred"] < 0
    # print(loader.features["y_pred"][l])
    import seaborn as sns
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=100, density=True, alpha=0.6, color='blue', edgecolor='black', label='Data Histogram')
    import seaborn as sns
    sns.kdeplot(data, color='red', linewidth=2, label='Data KDE')

    # 绘制理论上的高斯分布
    x = np.linspace(min(data), max(data), 100)
    gaussian = norm.pdf(x, loc=mu, scale=sigma)
    plt.plot(x, gaussian, 'g--', linewidth=2, label='Gaussian Fit')

    # 添加标注和样式
    plt.title("Data Distribution vs Gaussian Distribution", fontsize=14)
    plt.xlabel("Value", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    
    
    
    
if __name__ == "__main__":
    plot_label(label_gen)