import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

file = "data/K_DAY.csv"


files = """data/K_DAY/SH.510050-XD上证50ETF.csv
data/K_DAY/SH.510300-沪深300ETF.csv
data/K_DAY/SH.510760-上证综指ETF.csv
data/K_DAY/SH.510880-红利ETF.csv
data/K_DAY/SH.512010-医药ETF.csv
data/K_DAY/SH.512100-中证1000ETF.csv
data/K_DAY/SH.512200-房地产ETF.csv
data/K_DAY/SH.512480-半导体ETF.csv
data/K_DAY/SH.512670-国防ETF.csv
data/K_DAY/SH.512690-酒ETF.csv
data/K_DAY/SH.512710-军工龙头ETF.csv
data/K_DAY/SH.512880-证券ETF.csv
data/K_DAY/SH.512980-传媒ETF.csv
data/K_DAY/SH.513050-中概互联网ETF.csv
data/K_DAY/SH.513180-恒生科技指数ETF.csv
data/K_DAY/SH.513910-港股央企红利ETF.csv
data/K_DAY/SH.515220-煤炭ETF.csv
data/K_DAY/SH.515290-银行ETF天弘.csv
data/K_DAY/SH.515880-通信ETF.csv
data/K_DAY/SH.516510-云计算ETF.csv
data/K_DAY/SH.516670-畜牧养殖ETF.csv
data/K_DAY/SH.516950-基建ETF.csv
data/K_DAY/SH.518880-黄金ETF.csv
data/K_DAY/SH.560080-中药ETF.csv
data/K_DAY/SH.560980-光伏30ETF.csv
data/K_DAY/SH.561980-半导体设备ETF.csv
data/K_DAY/SH.562500-机器人ETF.csv
data/K_DAY/SH.563300-中证2000ETF.csv
data/K_DAY/SH.588000-科创50ETF.csv
data/K_DAY/SH.588060-科创50ETF龙头.csv
data/K_DAY/SH.588200-科创芯片ETF.csv
data/K_DAY/SH.588800-科创100ETF华夏.csv"""
dfs = []
for file in files.split("\n"):
    df = pd.read_csv(file)
    dfs.append(df)

df = pd.concat(dfs)

print(df)


# key = "change_rate"
key = "time_key"
# time_key

g_df = df.groupby("name")

data = {}

max_len = 0

for code, df_t in g_df:
    arr = df_t["change_rate"].to_numpy()
    data[code] = arr
    max_len = max(max_len, arr.shape[-1])

print(max_len)

data = {
    k: np.pad(
        v,
        (max_len - v.shape[0], 0),
        "constant",
        constant_values=(float("nan"), float("nan")),
    )
    for k, v in data.items()
}

data_c = np.stack([p for p in data.values()])
data_names = np.array([p for p in data.keys()])

mean = np.nanmean(data_c, axis=0)


data_c = np.stack(
    [np.nan_to_num(data_c[:, i], mean[i]) for i in range(mean.shape[0])], axis=-1
)

print(data_c.shape)

# 1/0

# print(data_c)


# print(data_c.shape)

from sklearn.cluster import KMeans
import numpy as np

n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=17, n_init="auto").fit(data_c)

rs = kmeans.predict(data_c)

print(rs)

for i in range(n_clusters):
    c = data_names[rs == i]
    print(i, c)


# kmeans.predict([[0, 0], [12, 3]])
# kmeans.cluster_centers_

# start = g_df[key]["min"].max()

# end = g_df[key]["max"].min()

# print(start, end)

# start = g_df.min("time_key")

# print(start)

# print(df)
