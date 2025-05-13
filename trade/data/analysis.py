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
from trade.train.utils import *
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm


def label_gen(data):
    l = data["close"].shape[0]
    pred = np.concatenate(
            [
                (data["open"][1:] / data["close"][:-1] - 1),
                [float("nan")] * 1,
            ]
    )[:l]
    
    valid = (np.abs(pred) <= 0.098) & (np.abs(data["change"]) < 0.098)
    pred = pred[valid]
    
    for k in data.keys():
        data[k] = data[k][valid]            
    
    return {
        # "pred": np.concatenate(
        #     [np.log(data["open"][1:] / data["close"][:-1]), [float("nan")] * 1]
        # )[:l],
        "pred": pred,
        # "pred": np.concatenate(
        #     [np.log(data["open"] / data["close"]), []]
        # )[:l],
        # "pred": np.concatenate(
        #     [np.abs(np.log(data["open"][2:] / data["open"][1:-1])), [float("nan")] * 2]
        # )[:l],
        # "cls": np.concatenate([data["limit_flag"][1:], [float("nan")]])[:l],
        #  "cls": get_label(data),
    }

def plot_label(label_gen):
    loader = QlibDataloader(os.path.expanduser("~/output/qlib_bin"), [label_gen], extend_feature=False)
    data = loader.features.dropna()["y_pred"].to_numpy()
    
    
    weight = np.abs(data)
    bins = 200
    
    bucket = np.arange(bins) / bins

    def get_value(w):
        for i in range(len(bucket)):
            if bucket[i] > w:
                return bucket[i]
        return 1
    
    weight = np.array([
        get_value(w) for w in weight
    ])
    
    weight /= weight.sum()
    data_resample = np.random.choice(data, len(data), replace=True, p=weight)
    
    weight2 = data * data
    weight2 /= weight2.sum()
    data_resample2 = np.random.choice(data, len(data), replace=True, p=weight2)
    
    mu = np.mean(data)
    sigma = np.std(data)
    # l = loader.features["y_pred"] < 0
    # print(loader.features["y_pred"][l])

    plt.figure(figsize=(10, 6))
    # plt.hist(data, bins=100, density=True, alpha=0.6, color='blue', edgecolor='black', label='Data Histogram')
    plt.hist(data_resample, bins=bins, density=True, alpha=0.6, label='Data Resample')
    # plt.hist(data_resample2, bins=100, density=True, alpha=0.6, label='Data Resample2')
    
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


def select_inst():
    with Context() as ctx:

        # valid_stocks = None

        test_ins = ["BJ872374"]

        insts_map = {"mae":[], "corr":[], "profit":[]}
        for i in range(50):
            df = from_cache(f"RegDNN_e_{i}/predict.pkl")
            if df is not None:
                # print(i, len(df))
                df["mae"] = (
                    (df["y"] - df["y_p"]) / (df["y"] + 1e-8)
                ).abs()
                # if valid_stocks is None:
                max_date = df[~df["y"].isna()]["datetime"].max()
                # print(df[df["instrument"] == "SZ300495"])
                # print("max_date", max_date)
                valid_stocks = df[(df["datetime"] == max_date) & (~df["y"].isna())]["instrument"].to_list()
                corr = df[df['instrument'].isin(valid_stocks)].dropna().groupby("instrument")[["y", "y_p"]].corr().loc[(slice(None), "y_p"), "y"].reset_index()[["instrument", "y"]]

                yp90 = df[df['instrument'].isin(valid_stocks)].dropna()["y_p"].to_numpy()
                yp90 = np.quantile(yp90, 0.9)
                print("yp90", yp90)
                # profit = df[df['instrument'].isin(valid_stocks)].dropna().groupby("instrument").apply(
                #     lambda x: x[x['y_p'] >= yp90]["y"].sum()
                # ).to_frame(name="profit").reset_index().sort_values(["profit"], ascending=False)

                profit = df[df['instrument'].isin(valid_stocks)].dropna().groupby("instrument").apply(
                    lambda x: len(x[(x['y_p'] >= yp90) & (x['y'] > 0)]) * 1.0 / (len(x[(x['y_p'] >= yp90)]) + 1e-8)
                ).to_frame(name="profit").reset_index().sort_values(["profit"], ascending=False)

                # print(profit)

                # print()
                # df = df[df["instrument"].isin(valid_stocks)]
                corr = corr.sort_values(["y"], ascending=False)
                corr.columns = ["instrument", "corr"]
                # print(corr)
                mae = df.groupby('instrument')["mae"].mean().to_frame(name="mae").reset_index().sort_values(["mae"])
                mae = mae[mae['instrument'].isin(valid_stocks)]

                # print(mae)
                insts_map["mae"].append(mae)
                insts_map["corr"].append(corr)
                insts_map["profit"].append(profit)

        insts_rs  = {}
        for k, insts in insts_map.items():
            # for i in insts:
            #     print(i[i["instrument"].isin(test_ins)])
#   col1_sum=('col1', 'sum'),
#     col1_mean=('col1', 'mean'),
#     col2_max=('col2', 'max')
            # insts = [p.head(10) for p in insts]
            insts = pd.concat(insts).sort_values(k, ascending=(k == "mae"))
            # print(insts[insts["instrument"].isin(test_ins)])
            insts = insts.groupby("instrument").agg({k: 
                
                
                ["min", "max", "mean", "var", "count"]}).reset_index()
            
            # print(insts.columns)
            insts.columns = ['_'.join(col).strip() if col[1] != '' else col[0] for col in insts.columns]
            

            insts = insts.sort_values([f"{k}_mean"], ascending=(k == "mae")).reset_index(drop=True)
            # print(insts)
            insts.to_csv(f"{k}.csv")
            insts_rs[k] = insts
        l = []
        for v in insts_rs.values():
            l.extend(
                v["instrument"].tolist()
            )
        print(l)
        print(len(l),"vs", len(set(l)))
        # count = {}
        rs = list(insts_rs.values())
        for v in rs[1:]:
            rs[0] = pd.merge(rs[0], v, on = "instrument", how="outer")
        mean = rs[0].mean()
        print(mean)
        # print(rs[0].dropna())
        
        r = rs[0]
        r = r[(r["corr_mean"] > mean["corr_mean"])]
        
        print(r.sort_values(["profit_mean"], ascending=False))
        
        


def plot_pred(save_names = ["RegDNN", "RegTransformer"]):
# def plot_pred(save_names = ["RegDNN"]):
    with Context() as ctx:
        
        preds = [from_cache(f"RegDNN_e_{i}/predict.pkl") for i in range(50)]
        preds = [p for p in preds if p is not None]
        # preds = [ from_cache(f"{save_name}/predict.pkl") for save_name in save_names]
        
        
        profit  = pd.read_csv("profit.csv")["instrument"].tolist()
        
        preds = [
            p[p["instrument"].isin(profit)] for p in preds
        ]
        
        if len(save_names) > 10:
            pred = preds[0].copy()
            
            direct = (preds[1]["y"] * preds[0]["y"] > 0)
            pred["y"] = ((preds[1]["y"] + preds[0]["y"]) / 2)
            
            pred = pred[direct]
            
            save_names.append("mix_0")
            preds.append(pred)
        
        # pred = preds[1].copy()
        # pred["y"] = preds[0]["y_p"]
        # save_names.append("mix_1")
        # preds.append(pred)
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        
        for idx, pred1 in enumerate(preds):
            
            
            # print(pred[pred["y"].isna()])
            
            pred = pred1.dropna()
            
            top_n = pred.groupby('datetime').apply(
                lambda x: x.sort_values('y_p', ascending=False).head(5)
            ).reset_index(drop=True)
            labels_t = top_n["y"].to_numpy()
            preds_t = top_n["y_p"].to_numpy()
            
            d_sig = np.sum(labels_t * preds_t >0)
            # print(
            #     "d_sig top_n", d_sig / len(labels_t)
            # )
            
            # print(top_n)
            
            labels = pred["y"].to_numpy()
            preds = pred["y_p"].to_numpy()
            np.random.seed(42)

            # mae = np.abs(labels-preds)
            diff = labels-preds
            # print(
            #     "mean", np.mean(diff), len(pred)
            # )
            d_sig = np.sum(labels * preds >0)
            # print(
            #     "d_sig", d_sig / len(labels)
            # )
            
            q_rs = []
            qs = [0.75, 0.85, 0.95, 0.99, 0.995, 0.999]
            for q in qs:
                q_v = np.quantile(preds, q)
                select = preds >= q_v
                d_sig = np.sum(labels[select] * preds[select] > 0)
                # d_sig_n = np.sum(labels[select] * preds[select] < 0)
                
                # if q == qs[-1]:
                #     print(pred[select])
                
                d_sig_2 = np.sum(labels[select] >= 1.0 )
                sum_label = np.sum(labels[select]) / len(labels[select])
                print(
                    f"d_sig:{d_sig / len(labels[select]):.3f} {d_sig_2/ len(labels[select]):.3f} q:{q:.3f}, q_v{q_v:.3f} sum_label {sum_label:.3f}"
                )
                q_rs.append(
                    [labels[select] , preds[select]]
                )
            print("#"*10)
            
        
            
        #     plt.figure(figsize=(10, 6))
        # # plt.hist(data, bins=100, density=True, alpha=0.6, color='blue', edgecolor='black', label='Data Histogram')
        #     plt.hist(diff, bins=100, density=True, alpha=0.6, label='Data Resample')
        #     plt.title("Data Distribution vs Gaussian Distribution", fontsize=14)
        #     plt.xlabel("Value", fontsize=12)
        #     plt.ylabel("Density", fontsize=12)
        #     plt.legend()
        #     plt.grid(True, alpha=0.3)
        #     plt.show()
            
        #     labels, preds = q_rs[-1]
        # # 计算指标
        #     from scipy.stats import pearsonr, spearmanr
        #     from sklearn.metrics import r2_score
        #     r_pearson, p_pearson = pearsonr(labels, preds)
        #     r_spearman, p_spearman = spearmanr(labels, preds)
        #     r2 = r2_score(labels, preds)

        #     # 绘制散点图 + 回归线
        #     # plt.figure(figsize=(10, 6))
        #     # plt.subplot(len(preds) + 1, 1, idx+1) 
        #     sns.regplot(x=labels, y=preds, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=axs[idx, 0])
        #     # sns.regplot(x=q_rs[-1][0], y=q_rs[-1][1], scatter_kws={'alpha':0.5})
        #     # plt.plot([-0.1, 0.1], [-0.1, 0.1], '--', color='grey')  # 理想对角线
        #     axs[idx, 0].plot([-10, 10], [-10, 10], '--', color='grey')  # 理想对角线
        #     # axs[idx, 0].xlabel('True Labels')
        #     # axs[idx, 0].ylabel('Predictions')
        #     axs[idx, 0].set_title(f'{save_names[idx]} Pearson r={r_pearson:.3f}, Spearman ρ={r_spearman:.3f}\nR²={r2:.3f}')
            
        # plt.grid(True)
        # plt.show()
    
    

    # 绘制叠加直方图
        # plt.figure(figsize=(10, 6))
        # plt.hist(y, bins=50, alpha=0.5, label='Labels', color='blue')
        # plt.hist(y_p, bins=50, alpha=0.5, label='Predictions', color='red')
        # plt.xlabel('Value')
        # plt.ylabel('Frequency')
        # plt.title('Label vs. Prediction Distribution')
        # plt.legend()
        # plt.show()
        
        # 计算分位数
        # quantiles = np.linspace(0, 1, 100)
        # label_quantiles = np.quantile(labels, quantiles)
        # pred_quantiles = np.quantile(preds, quantiles)

        # # 绘制Q-Q图
        # plt.figure(figsize=(8, 8))
        # plt.scatter(label_quantiles, pred_quantiles, alpha=0.6)
        # plt.plot([-0.1, 0.1], [-0.1, 0.1], '--', color='red')  # 对角线参考线
        # plt.xlabel('Label Quantiles')
        # plt.ylabel('Prediction Quantiles')
        # plt.title('Q-Q Plot: Labels vs. Predictions')
        # plt.grid(True)
        # plt.show()

if __name__ == "__main__":
    # plot_label(label_gen)
    # plot_pred()
    select_inst()
