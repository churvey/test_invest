import pandas as pd
import math
from collections import namedtuple
import random
random.seed(43)

# what's a strategy
# event --> buy or sell & amount 
# event --> action --> state
# event --> buy 


class State(object):
    
    
   
    

    def __init__(self, cash, first_buy_ratio=0.3, fee=2.5 / 10000) -> None:
        self._cash = cash
        self._amount = 0
        self._fee = fee
        self._history = []
        self._next_buy_amount = None
        self._next_sell_amount = None
        self._next_buy_price = None
        self._next_sell_price = None
        self._min_value = self._cash
        self._max_value = self._cash
        self._first_buy_ratio = first_buy_ratio
        self._cur_value = self._cash
        self._total_deal = 0.0
        self._min_ratio = 0.005

    def check_valid(self) -> None:
        assert self._amount >= 0 and self._cash >= 0, f"{
            self._amount} - {self._cash}"

    def upper_price(self, price) -> float:
        new_p = int(price * 10000) / 10000.0
        if price > new_p:
            new_p += 1.0/10000
        assert new_p >= price
        return new_p

    def lower_price(self, price) -> float:
        return int(price * 10000) / 10000.0

    def update(self, time, price):
        value = self._cash + self._amount * price * (1 - self._fee)
        self._max_value = max(value, self._max_value)
        self._min_value = min(value, self._min_value)
        self._cur_value = value

        multi = 2
        for i in range(1, len(self._history)):
            if self._history[-i][-1] > 0 ^ self._history[-i-1][-1] > 0:
                break
            else:
                multi *= 2
                
        multi = min(multi, 16)

        if self._history[-1][-1] < 0:
            self._next_buy_price = self.lower_price(price * (1 - self._min_ratio * multi // 2))
            self._next_sell_price = self.upper_price(
                price * (1 + self._min_ratio * multi))
        else:
            self._next_buy_price = self.lower_price(price * (1 - self._min_ratio * multi))
            self._next_sell_price = self.upper_price(price * (1 + self._min_ratio * multi // 2))

        self._next_sell_amount = self._amount // 20
        self._next_buy_amount = int((self._cash / self._next_buy_price) / 20)

        # print(time +"\t"+str(self))
        self.check_valid()

    def buy(self, time, price, amount):
        self._amount += amount
        self._total_deal += amount * price
        self._cash -= amount * price * (1 + self._fee)
        self._history.append(
            (time, price, amount)
        )
        self.update(time, price)

    def sell(self, time, price, amount):
        self._amount -= amount
        self._total_deal += amount * price
        self._cash += amount * price * (1 - self._fee)
        self._history.append(
            (time, price, -amount)
        )
        self.update(time, price)

    def do_trial(self, data) -> None:
        # print(data[0:1],"\n", data[-1:])
        for index, row in data.iterrows():
            if self._amount == 0:
                # first_buy
                self.buy(row["time_key"], row["close"],
                         int(self._cash * self._first_buy_ratio / row["close"]))
                continue

            if row["open"] > row["close"]:  # we assume price is falling
                if row["high"] >= self._next_sell_price:
                    self.sell(row["time_key"], self._next_sell_price,
                              self._next_sell_amount)
                if row["low"] <= self._next_buy_price:
                    self.buy(row["time_key"], self._next_buy_price,
                             self._next_buy_amount)
            else:
                if row["low"] <= self._next_buy_price:
                    self.buy(row["time_key"], self._next_buy_price,
                             self._next_buy_amount)
                if row["high"] >= self._next_sell_price:
                    self.sell(row["time_key"], self._next_sell_price,
                              self._next_sell_amount)

    def __repr__(self) -> str:
        return f"{self._cur_value:{2}.{10}}\t{self._min_value:{2}.{10}}\t{self._max_value:{2}.{10}}\t{self._total_deal:{2}.{10}}"


data = pd.read_csv("SH.510300.csv")
cur = []
min_value = []
max_value = []
total = []
for begin in range(0, len(data) - 60 * 4 * 250, 60 * 4):
    state = State(cash=1000 * 1000.0, first_buy_ratio=0.1)
    # import random
    # begin = random.randint(0, len(data) - 60 * 4 * 250)
    state.do_trial(data[begin: begin + 60 * 4 * 250])
    cur.append(state._cur_value)
    min_value.append(state._min_value)
    max_value.append(state._max_value)
    total.append(state._total_deal)
    print(state)
    
df = pd.DataFrame()
df["cur"] = cur
df["min_value"] = min_value
df["max_value"] = max_value
df["total"] = total
df.to_csv("result.csv")