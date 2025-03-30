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


class BollTradeEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """ """

    metadata = {}

    def __init__(
        self,
        df: pd.DataFrame,
        date_range: Tuple[str, str] = None,
        instruments: List[str] = None,
        max_split=10,
        max_length=None,
        fee=0.0001,
        weight_as_feature=True,
        override_action=0,
        eval=False,
        **kwargs,
    ):

        self.fee = fee
        self.eval = eval
        self.max_split = max_split
        self.weight_as_feature = weight_as_feature
        self.override_action = override_action
        self.df = df.sort_values(["datetime", "instrument"])
        self.dates = np.sort(np.unique(self.df["datetime"].to_numpy())).tolist()

        self.date_start = 0
        self.date_end = len(self.dates)

        if date_range:
            begin, end = date_range
            self.date_start = self.dates.index(begin)
            self.date_end = self.dates.index(end)

        if instruments:
            self.instruments = sorted(instruments)
            self.df = self.df[self.df["instrument"].isin(instruments)]
        else:
            self.instruments = np.sort(np.unique(self.df["instrument"].to_numpy()))

        print(f"weight_as_feature: {self.weight_as_feature} fee:{self.fee}")

        self.dates = np.sort(np.unique(self.df["datetime"].to_numpy()))

        self.feature_columns = [
            col
            for col in self.df.columns
            if col not in ["datetime", "instrument", "volume"]
        ]
        self.all_state = self.df[self.feature_columns].to_numpy()
        self.all_close_v = self.df["close"].to_numpy()

        self.all_z_pos = self.df["z_pos_20"].to_numpy()

        self.actions = get_actions(self.max_split, len(self.instruments))

        self.action_space = spaces.Discrete(len(self.actions))

        # self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            -np.inf,
            np.inf,
            (
                len(self.instruments)
                * (len(self.feature_columns) + self.weight_as_feature),
            ),
            dtype=np.float32,
        )

        # update
        self.state: np.ndarray | None = None
        self.date_index = 0
        self.value = 1.0
        self.cash = 1.0
        self.shares = 0
        self.close_v = None
        self.close_init = None
        self.z_pos = None
        self.total_fee = 0.0
        self.action_values = []

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        terminated = False

        if self.date_index >= self.date_end - 1:
            terminated = True

        old_value = self.value
        weight = self.actions[action][0]
        new_share = self.value * weight * (1 - self.fee) / self.close_v
        fee = np.abs(new_share - self.shares) * self.close_v * self.fee
        self.shares = new_share
        self.cash = self.value - fee - self.shares * self.close_v
        self.value -= fee
        self.total_fee += fee

        reward = 0
        if not terminated:
            cur_pos = self.z_pos
            while True:
                self._update_state()
                if cur_pos != self.z_pos or self.date_index >= self.date_end - 1:
                    reward = math.log(self.value / old_value)
                    break

        return (
            self.state,
            reward,
            terminated,
            False,
            {
                "value": self.value,
                "days": self.date_index,
                "date": self.dates[self.date_index],
                "reward": reward,
                "close": self.close_v,
                "close_norm": self.close_v / self.close_init,
                "shares": self.shares,
                "total_fee": self.total_fee,
            },
        )

    def _update_state(self):
        self.date_index += 1
        self.close_v = self.all_close_v[self.date_index]
        self.z_pos = self.all_z_pos[self.date_index]
        self.state = self.all_state[self.date_index]
        if self.weight_as_feature:
            self.state = np.concatenate([self.state, [self.shares]])
        self.value = self.shares * self.close_v + self.cash

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self.total_fee = 0.0
        if self.eval:
            self.date_index = self.date_start - 1
            self.value = 1
            self.cash = 1
            self.shares = 0
        else:
            self.date_index = (
                self.np_random.integers(self.date_start, self.date_end) - 1
            )
            self.value = 1

        self._update_state()
        if not self.eval and self.np_random.integers(self.date_start, self.date_end) % 2 == 1:
            self.cash = self.np_random.random()
            print(f"cash init {self.cash}")
            self.shares = (self.value - self.cash) * (1 - self.fee) / self.close_v
            fee = self.shares * self.close_v * self.fee
            self.cash -= fee
            self.value = self.cash + self.shares * self.close_v
            self.total_fee = fee

        self.close_init = self.close_v.copy()
        self.action_values = []

        return (
            self.state,
            {
                "value": self.value,
                "days": self.date_index,
                "date": self.dates[self.date_index],
                "reward": 0,
                "close": self.close_v,
                "close_norm": self.close_v / self.close_init,
                "shares": self.shares,
                "total_fee": self.total_fee,
            },
        )

    def render(self):
        pass

    def close(self):
        pass
