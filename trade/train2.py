import torch
from torch import nn
import pandas as pd
import numpy as np
import os
import cloudpickle
from .dataloader import Dataloader


class Net(nn.Module):
    def __init__(self, input_dim, output_dim=1, layers=(256,), act="LeakyReLU"):
        super(Net, self).__init__()

        layers = [input_dim] + list(layers)
        dnn_layers = []
        drop_input = nn.Dropout(0.05)
        dnn_layers.append(drop_input)
        hidden_units = input_dim
        for i, (_input_dim, hidden_units) in enumerate(zip(layers[:-1], layers[1:])):
            fc = nn.Linear(_input_dim, hidden_units)
            if act == "LeakyReLU":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
            elif act == "SiLU":
                activation = nn.SiLU()
            elif act == "sigmoid":
                activation = nn.Sigmoid()
            else:
                raise NotImplementedError(f"This type of input is not supported")
            bn = nn.BatchNorm1d(hidden_units)
            seq = nn.Sequential(fc, bn, activation)
            dnn_layers.append(seq)
        drop_input = nn.Dropout(0.05)
        dnn_layers.append(drop_input)
        fc = nn.Linear(hidden_units, output_dim)
        dnn_layers.append(fc)
        # optimizer  # pylint: disable=W0631
        self.dnn_layers = nn.ModuleList(dnn_layers)
        self._weight_init()

    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        cur_output = x
        for i, now_layer in enumerate(self.dnn_layers):
            cur_output = now_layer(cur_output)
        return cur_output


def get_model(input_dim, output_dim = 1):
    model = Net(input_dim, output_dim)
    opt = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0002)
    # opt = torch.optim.Adam(model.parameters(), lr=0.05)
    # opt = torch.optim.SGD(model.parameters(), lr=0.05)

    # lr : 0.002
    # max_steps : 8000
    # batch_size : 8192
    # early_stop_rounds : 50
    # eval_steps : 20
    # optimizer : adam
    # loss_type : mse
    # seed : None
    # device : cuda:0
    # use_GPU : True
    # weight_decay : 0.0002
    # enable data parall : False

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=0.5,
        patience=10,
        verbose=True,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0.00001,
        eps=1e-08,
    )

    def mse(pred, target):
        # sqr_loss = torch.mul(pred - target, pred - target).mean()
        # loss = torch.mul(sqr_loss, w).mean()
        sqr_loss = torch.mul(pred - target, pred - target)
        loss = torch.mul(sqr_loss, 1.0).mean()
        return loss

    # return model, opt, scheduler, torch.nn.MSELoss()
    
# label 0 1404757
# label 1 11636
# label -1 6963
    # w = np.array([1404757.0, 11636.0, 6963.0])
    # w = np.array([1.0, 1.0, 1.0])
    # w = w / w.sum()
    # print("weight", w)
    # weight = torch.tensor(w).cuda().to(torch.float32)
    # return model, opt, scheduler, torch.nn.CrossEntropyLoss(weight)
    return model, opt, scheduler, mse

    

from torch.utils.tensorboard import SummaryWriter
import datetime


def get_writer(date=None):
    now = datetime.datetime.now()
    now = now.strftime("%Y-%m-%d-%H-%M")
    if not date:
        folder = f"./metric/{now}"
    else:
        folder = f"./metric/{date}"
        if os.path.exists(folder):
            folder = f"{folder}-{now}"
    writer = SummaryWriter(folder)
    return writer


# shape of x train/valid (437674, 87) vs  (115511, 87)
# shape of y train/valid (437674, 1) vs  (115511, 1)
# shape of x train/valid (268627, 84) vs  (95357, 84)
# shape of y train/valid (268627, 1) vs  (95357, 1)


# def rolling(
#     dates={
#         "train": [
#             "2012-01-01",
#             "2023-12-31",
#         ],
#         "valid": ["2024-01-01", "2024-12-31"],
#     }
# ):
#     from .dataloader import Dataloader
#     import cloudpickle

#     begin = min([v[0] for v in dates.values()])
#     end = max([v[1] for v in dates.values()])

#     if os.path.exists("./dataloader.bin"):
#         with open("./dataloader.bin", "rb") as f:
#             dataloader = cloudpickle.loads(f.read())
#     else:
#         dataloader = Dataloader(
#             path="~/output/qlib_bin",
#             # range_hint = [begin, end]
#         )
#         with open("./dataloader.bin", "wb") as f:
#             f.write(cloudpickle.dumps(dataloader))

#     writer = get_writer(dates["valid"][0])
#     from .metric import Metric

#     for date_i, data in enumerate(dataloader.rolling_v2(date_range=[begin, end])):
#         roll = date_i
#         train_metric = Metric(writer, "train", roll)
#         eval_metric = Metric(writer, "eval", roll)
#         train_index = 0
#         valid_index = 0
#         dfs = []
#         model = None
#         # for date_i, data in enumerate(dataloader.rolling(**dates, step = roll * 5 + 1)):

#         batch_size = 8096

#         if model is None:
#             model, opt, schedule, loss_fn = get_model(data["x"]["train"].shape[-1], 3)
#         if roll == 0 and date_i == 0:
#             print(model)
#         model.cuda()
#         x_train, x_test = data["x"]["train"], data["x"]["valid"]
#         y_train, y_test = data["y"]["train"], data["y"]["valid"]

#         epcho_num = 50
#         train_len = x_train.shape[0]

#         if train_len <= 1:
#             continue

#         test_len = x_test.shape[0]
#         print(
#             f'shape of origin x train/valid {data["x"]["train"].shape} vs  {data["x"]["valid"].shape}  vs {x_train.shape}'
#         )
#         print(
#             f'shape of origin y train/valid {data["y"]["train"].shape} vs  {data["y"]["valid"].shape} vs {y_train.shape}'
#         )

#         r = np.array(np.arange(train_len))

#         eval_ys = []
#         for epoch in range(epcho_num):
#             model.train()
#             np.random.shuffle(r)
#             x_train = x_train[r]
#             y_train = y_train[r]
#             for i in range((train_len + batch_size - 1) // batch_size):
#                 x = x_train[i * batch_size : i * batch_size + batch_size]
#                 y = y_train[i * batch_size : i * batch_size + batch_size]
#                 x = torch.asarray(x).cuda()
#                 y = torch.asarray(y).cuda()
#                 opt.zero_grad()
#                 y_p = model.forward(x)
#                 loss = loss_fn(y, y_p)
#                 loss.backward()
#                 opt.step()

#                 loss = loss.detach().cpu()
#                 train_index += 1

#                 if train_index % 20 == 0 and train_index > 0:
#                     schedule.step(metrics=loss)
#                     print(
#                         "train",
#                         date_i,
#                         epoch,
#                         train_index,
#                         loss.numpy(),
#                     )
#                 train_metric.update(y.detach(), y_p.detach(), train_index)
#         train_metric.finish(date_i, **{"lr": opt.param_groups[0]["lr"]})
#         if test_len == 0:
#             continue
#         model.eval()
#         eval_ys = []
#         for i in range((test_len + batch_size - 1) // batch_size):
#             x = x_test[i * batch_size : i * batch_size + batch_size]
#             y = y_test[i * batch_size : i * batch_size + batch_size]
#             x = torch.asarray(x).cuda()
#             y = torch.asarray(y).cuda()

#             valid_index += 1
#             with torch.no_grad():
#                 y_p = model.forward(x)
#                 loss = loss_fn(y, y_p)
#                 if valid_index % 20 == 0 and valid_index > 0:
#                     print("eval", date_i, valid_index, loss.cpu().numpy())
#                 eval_metric.update(y, y_p, valid_index)
#                 eval_ys.append(y_p)

#         eval_metric.finish(date_i)
        # pred = torch.cat(eval_ys).detach().cpu().numpy()
        # df = data["indices"]["valid"]
        # df["y_hat"] = pred.reshape([-1]).tolist()
        # df["y_rand"] = np.random.rand(*pred.shape).reshape([-1]).tolist()
        # df = df[df["datetime"] == df["datetime"].iloc[0]]
        # # p = df["datetime"].unique()
        # # assert len(p) == 1
        # dfs.append(df)

        # df = pd.concat(dfs)
        # from .strategy import Strategy

        # baseline = []
        # records_all = {}
        # max_drop  = {}
        # s = Strategy(df, data["base_data"])
        # len_r = 0
        # for top_n in [2]:
        #     # print("top_n", top_n, len(records), records)
        #     records_all[top_n] =  s.run(top_n = top_n)[0]
        #     len_r = len(records_all[top_n])
        #     # records_all[f"{top_n}_t"] =  s.run(top_n = top_n, use_tomorrow = True)[0]

        # for k, v_n  in records_all.items():
        #     max_drop[k] = np.array(v_n)
        #     max_drop[k][0] = 0.0
        #     for i in range(1, len(v_n)):
        #         m = np.max(v_n[:i])
        #         max_drop[k][i] = (v_n[i] - m) / m

        # records_r, baseline = s.run("y_rand", top_n=2, with_baseline=True)

        # # assert len(baseline) == len(records_all[1])

        # # baseline = baseline / baseline[0] - 1

        # back_metric = Metric(writer, "backtest", roll, None)
        # for i in range(len_r):
        #     records = {f"model_{k}": v[i] / 1000000 - 1 for k, v in records_all.items()}
        #     drops = {f"{k}": v[i] for k, v in max_drop.items()}
        #     # incs = {f"model_{k}": v[i]  / v[i -1 ] - 1 if i >0 else 0 for k, v in records_all.items()}
        #     back_metric.finish(
        #         i,
        #         **{
        #             "prof": {
        #                 **records,

        #                 "rand": records_r[i] / 1000000 - 1,
        #                 "baseline": baseline[i] / baseline[0] - 1,
        #             },
        #             "max_drops":{
        #                  **drops
        #             }
        #             # "inc":{
        #             #     **incs,
        #             #     "baseline": baseline[i] / baseline[i-1] - 1 if i > 0 else 0,
        #             # }
        #         },
        #     )


def train(
    dates={
        "train": [
            "2018-01-01",
            "2022-12-31",
        ],
        "valid": ["2023-01-01", "2023-12-31"],
        "predict": ["2024-01-01", "2024-12-31"],
    },
    roll_num = 3
):

    begin = min([v[0] for v in dates.values()])
    end = max([v[1] for v in dates.values()])

    if os.path.exists("./dataloader.bin"):
        with open("./dataloader.bin", "rb") as f:
            dataloader = cloudpickle.loads(f.read())
    else:
        dataloader = Dataloader(
            path="/home/cc90/output/qlib_bin",
            # range_hint = [begin, end]
        )
        with open("./dataloader.bin", "wb") as f:
            f.write(cloudpickle.dumps(dataloader))

    data = dataloader.load(dates)

    writer = get_writer(dates["predict"][0])

    train_len = data["x"]["train"].shape[0]
    test_len = data["x"]["valid"].shape[0]
    print(
        f'shape of x train/valid {data["x"]["train"].shape} vs  {data["x"]["valid"].shape}'
    )
    print(
        f'shape of y train/valid {data["y"]["train"].shape} vs  {data["y"]["valid"].shape}'
    )
    batch_size = 8096

    r = np.array(np.arange(train_len))
    for roll in range(roll_num):
        from .metric import Metric

        train_metric = Metric(writer, "train", roll)
        eval_metric = Metric(writer, "eval", roll)

        model, opt, schedule, loss_fn = get_model(data["x"]["train"].shape[-1], 1)
        if roll == 0:
            print(model)
        model.cuda()
       
        
        # print(f" y_train {y_train.shape}")

        eval_ys = []
        for epoch in range(50):
            data = dataloader.load(dates)
            x_train, x_test = data["x"]["train"], data["x"]["valid"]
            y_train, y_test = data["y"]["train"], data["y"]["valid"]
            train_len = data["x"]["train"].shape[0]
            test_len = data["x"]["valid"].shape[0]
            r = np.array(np.arange(train_len))
            model.train()
            np.random.shuffle(r)
            x_train = x_train[r]
            y_train = y_train[r]
            for i in range((train_len + batch_size - 1) // batch_size):
                x = x_train[i * batch_size : i * batch_size + batch_size]
                y = y_train[i * batch_size : i * batch_size + batch_size]
                x = torch.asarray(x).cuda()
                y = torch.asarray(y).cuda()
                opt.zero_grad()
                y_p = model.forward(x)
                # print(f"y {y.shape, y.dtype}, y_p {y_p.shape, y_p.dtype}")
                loss = loss_fn(y_p, y)
                loss.backward()
                opt.step()

                loss = loss.detach().cpu()
                index = i + epoch * train_len // batch_size

                if index % 20 == 0 and index > 0:
                    schedule.step(metrics=loss)
                    print(
                        "train",
                        epoch,
                        index,
                        loss.numpy(),
                    )
                train_metric.update(y.detach(), y_p.detach(), index)
            train_metric.finish(epoch, **{"lr": opt.param_groups[0]["lr"], "loss":loss.cpu()})
            model.eval()
            eval_ys = []
            for i in range((test_len + batch_size - 1) // batch_size):
                x = x_test[i * batch_size : i * batch_size + batch_size]
                y = y_test[i * batch_size : i * batch_size + batch_size]
                x = torch.asarray(x).cuda()
                y = torch.asarray(y).cuda()

                index = i + epoch * test_len // batch_size
                with torch.no_grad():
                    y_p = model.forward(x)
                    loss = loss_fn(y_p, y)
                    # loss_sum += loss
                    if index % 20 == 0 and index > 0:
                        print("eval", epoch, index, loss.cpu().numpy())
                    eval_metric.update(y, y_p, index)
                    # eval_ys.append(y_p)
            eval_metric.finish(epoch, **{"loss":loss.cpu()})

        x_pred = data["x"]["predict"]
        pred = model.forward(torch.asarray(x_pred).cuda()).detach().cpu().numpy()
        df = data["indices"]["predict"].reset_index()
        # pred_df = pd.DataFrame(pred, columns = ["0","1", "2"])
        # df = pd.concat([df, pred_df], axis=1)
        # df["pred_sum"] = df["0"] + df["1"] + df["2"]
        # df["score"] = 2 * df["1"] - df["2"] + df["0"]
        
        # df = df.sort_values(by="score")
        print(df)
        
        
        
        
        # df["y_hat"] = pred.reshape([-1]).tolist()
        # df["y_rand"] = np.random.rand(*pred.shape).reshape([-1]).tolist()

        # from .strategy import Strategy

        # baseline = []
        # records_all = {}
        # s = Strategy(df, data["base_data"])
        # for top_n in [2]:
        #     # print("top_n", top_n, len(records), records)
        #     records_all[top_n] = s.run(top_n=top_n)[0]
        #     # records_all[f"{top_n}_t"] =  s.run(top_n = top_n, use_tomorrow = True)[0]

        # records_r, baseline = s.run("y_rand", top_n=10, with_baseline=True)

        # # assert len(baseline) == len(records_all[1])

        # # baseline = baseline / baseline[0] - 1

        # back_metric = Metric(writer, "backtest", roll, None)
        # for i in range(len(baseline)):
        #     records = {f"model_{k}": v[i] / 1000000 - 1 for k, v in records_all.items()}
        #     incs = {
        #         f"model_{k}": v[i] / v[i - 1] - 1 if i > 0 else 0
        #         for k, v in records_all.items()
        #     }
        #     back_metric.finish(
        #         i,
        #         **{
        #             "prof": {
        #                 **records,
        #                 "rand": records_r[i] / 1000000 - 1,
        #                 "baseline": baseline[i] / baseline[0] - 1,
        #             },
        #             "inc": {
        #                 **incs,
        #                 "baseline": baseline[i] / baseline[i - 1] - 1 if i > 0 else 0,
        #             },
        #         },
        #     )


if __name__ == "__main__":

    dates = [
        {
            "train": [
                "2008-01-01",
                "2014-12-31",
            ],
            "valid": ["2015-01-01", "2015-12-31"],
            "predict": ["2016-01-01", "2016-12-31"],
        },
        {
            "train": [
                "2008-01-01",
                "2016-12-31",
            ],
            "valid": ["2017-01-01", "2017-12-31"],
            "predict": ["2018-01-01", "2018-12-31"],
        },
        {
            "train": [
                "2008-01-01",
                "2018-12-31",
            ],
            "valid": ["2019-01-01", "2019-12-31"],
            "predict": ["2020-01-01", "2020-12-31"],
        },
        {
            "train": [
                "2008-01-01",
                "2020-12-31",
            ],
            "valid": ["2021-01-01", "2021-12-31"],
            "predict": ["2022-01-01", "2022-12-31"],
        },
        {
            "train": [
                "2008-01-01",
                "2023-12-31",
            ],
            "valid": ["2024-01-01", "2024-12-31"],
            "predict": ["2024-01-01", "2024-12-31"],
        },
    ]
    
    train(dates[0])

    # if os.path.exists("./dataloader.bin"):
    #     with open("./dataloader.bin", "rb") as f:
    #         dataloader = cloudpickle.loads(f.read())
    # else:
    #     dataloader = Dataloader(
    #         os.path.expanduser("~/output/qlib_bin"),
    #         # range_hint = [begin, end]
    #     )
    #     with open("./dataloader.bin", "wb") as f:
    #         f.write(cloudpickle.dumps(dataloader))
    # for d in dataloader.rolling_v2(date_range=["2000-01-01", "2024-12-31"]):
    #     # print(d)
    #     # if d["predict"][0] not in ["2015-12-15", "2022-02-16", "2015-06-11"]:
    #     #     continue
    #     print("train:",d)
    #     train(d)
    #     break
    # train(
    #     {
    #         "train": [
    #             "2008-01-01",
    #             "2016-12-31",
    #         ],
    #         "valid": ["2017-01-01", "2018-12-31"],
    #         "predict": ["2017-01-01", "2018-12-31"],
    #     }
    # )
