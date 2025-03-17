import torch
import pandas as pd
import numpy as np
import os
import cloudpickle
import json
from .dataloader import Dataloader


from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)


CORS(app)

if os.path.exists("./dataloader.bin"):
    with open("./dataloader.bin", "rb") as f:
        dataloader = cloudpickle.loads(f.read())
else:
    dataloader = Dataloader(
        os.path.expanduser("~/output/qlib_bin"),
        csi="all",
        range_hint=["2024-01-01", "2024-12-31"],
    )
    with open("./dataloader.bin", "wb") as f:
        f.write(cloudpickle.dumps(dataloader))

features = dataloader.feature_df
bf = dataloader.base_df

f = features[features["datetime"] == "2024-12-17"]
b = bf[bf["datetime"] == "2024-12-17"]

m = pd.merge(f, b, on=["datetime", "instrument"])

# b["name"] = b.loc[:,"instrument"]
# b["code"] = b.loc[:,"instrument"]


# low = f[f["limit_flag"]==2.0]

# upp = f[f["limit_flag"]==1.0]

for i in [5, 10, 20, 30, 60]:
    for j in [5, 10, 20, 30, 60]:
        if i < j:
            m[f"std_{i}_{j}"] = m[f"std_{i}"] / m[f"std_{j}"]

metrics = m.columns.tolist()
instruments = b["instrument"].tolist()

from trade.futu.get_stock import get_stocks 
stocks = get_stocks()
names = stocks["code"].tolist()
codes = stocks["name"].tolist()

mapping = {
    k.replace(".", ""):v for k,v in zip(names, codes)
}


@app.route("/feature")
def select_features():
    feature = request.args.get("feature")
    id = request.args.get("id")
    print(feature, id)
    
    code = None
    if id not in instruments:
        if feature not in metrics:
            feature = metrics[0]
        code = m.sort_values(feature)["instrument"].tolist()
        id = code[0]
            
    data = bf[bf["instrument"] == id]
    for col in ["open", "close", "high", "low"]:
        data[col] = data[col] / data["factor"]
    if code:
        name = [mapping[c] if c in mapping else c for c in code]

    # data_json = data.to_json()
    
    # print(data_json)
    # print(metrics)
    # print(instruments)

    txt = data.to_csv(index=False, sep="\t")
    rs = {
        "data": txt,
        "metrics": metrics,
        "stocks": None if not code else {
            "code":code,
            "name":name,
        }
        
    }
    return rs


if __name__ == "__main__":
    app.run(debug=False)
