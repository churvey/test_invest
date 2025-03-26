"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np

# from gymnasium.envs.classic_control import utils
# from gymnasium.error import DependencyNotInstalled
# from gymnasium.vector import AutoresetMode, VectorEnv
# from gymnasium.vector.utils import batch_space
import pandas as pd
from gymnasium import logger, spaces


def get_actions(max_split, n_stock):

    def _inner(remain, records):
        rs = []
        if len(records) == n_stock:
            total = sum(records)
            assert total <= max_split
            return [
                np.array(
                    [r * 1.0 / max_split for r in records]
                    + [1.0 - total * 1.0 / max_split],
                    dtype="float32",
                )
            ]  # norm to 1
        for i in range(max_split + 1):
            if remain - i >= 0:
                rs += _inner(remain - i, records + [i])
        return rs

    return _inner(max_split, [])


class TradeEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """ """

    metadata = {}

    def __init__(
        self,
        df: pd.DataFrame,
        date_range: Tuple[str, str] = None,
        instruments: List[str] = None,
        max_split=10,
        max_length=None,
        fee=0.00001,
        weight_as_feature=True,
        override_action=0,
        eval=False,
        **kwargs,
    ):

        self.fee = fee
        self.eval = eval
        self.weight_as_feature = weight_as_feature
        self.override_action = override_action
        self.df = df.sort_values(["datetime", "instrument"])
        self.dates_all = np.sort(np.unique(self.df["datetime"].to_numpy()))
        if date_range:
            begin, end = date_range
            self.df = self.df[
                (self.df["datetime"] >= begin) & (self.df["datetime"] < end)
            ]
            print("after date_range", date_range, len(self.df))
        if instruments:
            self.instruments = sorted(instruments)
            self.df = self.df[self.df["instrument"].isin(instruments)]
        else:
            self.instruments = np.sort(np.unique(self.df["instrument"].to_numpy()))

        print(f"weight_as_feature: {self.weight_as_feature} fee:{self.fee}")

        self.dates = np.sort(np.unique(self.df["datetime"].to_numpy()))
        self.date_offset = np.argmax(self.dates_all == self.dates[0])
        self.feature_columns = [
            col
            for col in self.df.columns
            if col not in ["datetime", "instrument", "volume"]
        ]
        if max_length:
            self.max_length = max_length
        else:
            self.max_length = self.dates.shape[0]

        self.max_split = max_split

        self.actions = get_actions(self.max_split, len(self.instruments))

        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            (
                len(self.instruments)
                * (len(self.feature_columns) + self.weight_as_feature),
            ),
            dtype=np.float32,
        )

        self.state: np.ndarray | None = None
        # date_df = None
        self.date_index = 0
        self.weight = None
        self.value = 1.0
        self.length = 0
        self.close_v = None
        self.close_init = None
        self.shares = None
        self.total_fee = 0.0
        self.value_history = []

    def rewards(self, old_close_v):
        weights = np.array(self.actions)
        old_shares = self.shares
        values = self.value.sum() * weights
        shares = values / old_close_v
        share_changed = shares - old_shares
        new_value = shares * self.close_v
        fee = np.abs(share_changed) * old_close_v * self.fee
        new_value -= fee

        return new_value, fee

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"
        self.date_index += 1
        self.length += 1
        terminated = self.date_index >= len(self.dates) or self.length > self.max_length
        if not terminated:
            old_close_v = self.close_v
            self._update_state()
            self.value_history.append(self.value.sum())
            values, fees = self.rewards(old_close_v)

            if self.override_action < 0:
                if self.override_action == -1:
                    values_s = np.sum(values, axis=-1)
                    action = np.argmin(values_s)
                else:
                    action = self.np_random.integers(0, len(self.actions))

            elif self.override_action > 0:
                values_s = np.sum(values, axis=-1)
                action = np.argmax(values_s)

            # print(values.shape, action, len(self.actions))
            # print(self.actions[action])
            old_v = self.value.sum()
            self.value = values[action]
            # print(self.value.sum(), old_v)
            reward = math.log(self.value.sum() / old_v)
            self.shares = self.value / self.close_v
            self.weight = self.value / self.value.sum()
            self.total_fee += fees[action].sum()
            if self.weight_as_feature:
                self.state = np.concatenate([self.state, np.array(self.weight[:-1])])

        else:
            reward = 0.0
            # share_changed = self.shares - self.shares

        max_draw_back = (
            max(self.value_history) - self.value_history[-1]
            if self.value_history
            else 0
        )

        return (
            self.state,
            reward,
            terminated,
            False,
            (
                {
                    "value": self.value,
                    "days": self.date_index + self.date_offset,
                    "date": self.dates[self.date_index],
                    "reward": reward,
                    "close": self.close_v,
                    "close_norm": self.close_v / self.close_init,
                    "ratio": self.weight,
                    "shares": self.shares,
                    # "share_changed": share_changed,
                    "total_fee": self.total_fee,
                    "max_draw_back": max_draw_back,
                    "names": self.instruments,
                }
                if not terminated
                else {
                    "value": self.value,
                    "days": self.date_index - 1,
                    "date": self.dates[self.date_index - 1],
                    "reward": reward,
                    "close": self.close_v,
                    "close_norm": self.close_v / self.close_init,
                    "ratio": self.weight,
                    "shares": self.shares,
                    # "share_changed": share_changed,
                    "total_fee": self.total_fee,
                    "max_draw_back": max_draw_back,
                    "names": self.instruments,
                }
            ),
        )

    def _update_state(self):
        date = self.dates[self.date_index]
        date_df = self.df[self.df["datetime"] == date]
        self.state = (
            date_df[self.feature_columns].to_numpy(dtype="float32").reshape([-1])
        )
        close = date_df["close"].to_list()
        close.append(1.0)
        self.close_v = np.array(close)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if self.eval:
            self.date_index = 0
            self.value = np.array([0.0] * len(self.instruments) + [1.0])
        else:
            self.date_index = self.np_random.integers(0, len(self.dates))
            self.value = self.actions[self.np_random.integers(0, len(self.actions))]

        # print("index", self.date_index , len(self.dates))

        self._update_state()
        self.length = 0
        self.shares = self.value / self.close_v
        self.weight = self.value / self.value.sum()
        if self.weight_as_feature:
            self.state = np.concatenate([self.state, np.array(self.weight[:-1])])
        self.close_init = self.close_v.copy()
        self.total_fee = 0.0
        self.value_history = []

        return (
            self.state,
            {
                "value": self.value,
                "days": self.date_index + self.date_offset,
                "date": self.dates[self.date_index],
                "reward": 0,
                "close": self.close_v,
                "close_norm": self.close_v / self.close_init,
                "ratio": self.weight,
                "shares": self.shares,
                "share_changed": self.shares - self.shares,
                "total_fee": self.total_fee,
                "max_draw_back": 0,
                "names": self.instruments,
            },
        )

    def render(self):
        pass

    def close(self):
        pass


def make_trade_env(**kwargs):
    return TradeEnv(config=kwargs)
