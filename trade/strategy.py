import torch
import numpy as np


class Strategy:

    def __init__(self, df, base_data, baseline_code="SH000300"):
        self.df = df.sort_values(by=["datetime"]).reset_index()
        self.base_data = base_data
        self.datetime = self.df["datetime"].unique()
        self.baseline_code = baseline_code
        self.price_use = "close"

    def run(
        self,
        pred_name="y_hat",
        cash=1000000,
        top_n=10,
        with_baseline=False,
        use_tomorrow=True,
    ):
        records = []
        baseline = []
        df_s = self.df.sort_values(by=["datetime", pred_name], ascending=[True, False])
        df_s = df_s.set_index("datetime", drop=False)
        account = {}
        buy_info = {}
        buys = []
        sells = []
        # action_date = self.datetime[1:] if use_tomorrow else self.datetime[:-1]
        # for d, datetime in zip(self.datetime, action_date):
        total_tmp2 = []
        for d_i in range(self.datetime.shape[0]):
            d = self.datetime[d_i]

            buys.append([])
            sells.append([])
            df = df_s.loc[d]
            n_len = len(df)
            # print("n_len", d, n_len)
            instruments = df["instrument"].to_numpy()
            
            print(f" {d} {instruments.shape}")
            
            if instruments.shape[0] > 300:
                print(df)
            
            pred = df[pred_name].to_numpy().reshape([-1])
            to_sell = []
            current_stocks = account.keys()
            # sell and buy use tomorrow's date
            for i in range(n_len - 1, top_n, -1):
                ins = instruments[i]
                if ins in current_stocks:
                    if i <= top_n or len(to_sell) == top_n:
                        break
                    to_sell.append(ins)
                    
            to_buy = [
                i
                for i in range(top_n)
                if pred[i] > 0
            ]
        
            # print(f"action datetime {d} \n sells:{to_sell}\n buys {[instruments[b] for b in to_buy]}")
        
            if d_i == self.datetime.shape[0] - 1:
                break
            datetime = self.datetime[d_i + 1] if use_tomorrow else self.datetime[d_i]
            
            for s in to_sell:
                # s = instruments[i]
                if not self.can_sell(ins, datetime):
                    continue
                sells[-1].append(s)
                amount = account[s]
                price = self.get_price(s, datetime)
                cash += amount * price
                if d_i == self.datetime.shape[0] - 1:
                    next_price = 0  
                else :
                    next_price = self.get_price(s, self.datetime[d_i + 1])
                tmp = np.array([buy_info[s][0], price, next_price])
                tmp2 = (tmp[1:] - tmp[:-1]) / tmp[:-1]
                total_tmp2.append(tmp2)
                print(f"{datetime} {s} {tmp.tolist()} {tmp2} {buy_info[s][1]}")
                account.pop(s)
                buy_info.pop(s)
            
            # print("buy sell", len(to_sell), len(to_buy))

            if len(to_buy) > 0:
                weight = pred[to_buy]
                weight = weight / np.sum(weight)
                last_cash = cash
                value_c = last_cash * weight
                for i, value in zip(to_buy, value_c):
                    buy = instruments[i]
                    if not self.can_buy(instruments[i], datetime):
                        continue
                    price = self.get_price(buy, datetime)
                    amount = int(value / price / 100) * 100
                    if amount >= 100:
                        buys[-1].append(buy)
                        cash -= amount * price
                        account[buy] = (
                            account[buy] + amount if buy in account else amount
                        )
                        buy_info[buy] = [
                            price, pred[i]
                        ]

            total_value = cash
            for k, v in account.items():
                total_value += v * self.get_price(k, datetime)

            inc = (total_value - records[-1]) / records[-1] if len(records) > 0 else 0

            if inc < -0.09:
                print(f"large drop {d} {len(records) } {inc} {buys[-2:]} {sells[-2:]}")

            records.append(total_value)
            if with_baseline:
                baseline.append(self.get_price(self.baseline_code, datetime))
        total_tmp2 = np.stack(total_tmp2, axis = 0)
        print(f"mean {np.mean(total_tmp2, axis=0)}")
        print(f"sum {np.sum(total_tmp2, axis=0)}")
        print(f"meadian {np.median(total_tmp2, axis=0)}")
        print(f"large {np.sum(total_tmp2[:,0] > total_tmp2[:,1]) / total_tmp2.shape[0]}")
        print(f"large or equal {np.sum(total_tmp2[:,0] >= total_tmp2[:,1]) / total_tmp2.shape[0]}")
        
        print(f"finish run {pred_name} {top_n}")
        return records, baseline

    def get_price(self, instrument, datetime):
        data = self.base_data[instrument]
        index = np.searchsorted(data["datetime"], datetime)
        if index >= data["datetime"].shape[0]:
            index -= 1
        while index >= 0 and np.isnan(data[self.price_use][index]):
            index -= 1
        assert index >= 0
        return data[self.price_use][index] / data["factor"][index]

    def can_buy(self, instrument, datetime):
        data = self.base_data[instrument]
        index = np.searchsorted(data["datetime"], datetime)
        if data["datetime"][index] != datetime:
            print(f"cannot buy due to not found {datetime} {instrument}")
            return False

        if np.isnan(data[self.price_use][index]):
            print(f"cannot buy due to close price is nan {datetime} {instrument}")
            return False

        inc = data["change"][index]
        if inc > 0.099 and data["high"][index] - data[self.price_use][index] < 1e-5:
            print(
                f'cannot buy {instrument} {datetime} last {inc} {data[self.price_use][index]} {data["high"][index]} '
            )
            return False

        return True

    def can_sell(self, instrument, datetime):
        data = self.base_data[instrument]
        index = np.searchsorted(data["datetime"], datetime)
        if data["datetime"][index] != datetime:
            print(f"cannot sell due to not found {datetime} {instrument}")
            return False

        if np.isnan(data[self.price_use][index]):
            print(f"cannot sell due to close price is nan {datetime} {instrument}")
            return False

        inc = data["change"][index]
        if inc < -0.099 and data[self.price_use][index] - data["low"][index] < 1e-5:
            print(
                f'cannot sell {instrument} {datetime} last {inc} {data[self.price_use][index]} {data["low"][index]} '
            )
            return False
        return True

    def plot(self):
        pass
