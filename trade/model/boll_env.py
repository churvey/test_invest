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

        self.action_space = spaces.Discrete(3)

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
        self.shares = 0
        self.hold = False
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
        else:
            if self.hold and action == 1:
                terminated = True
            if not self.hold and action == -1:
                terminated = True
            if terminated and self.eval:
                action = 0 #
                terminated = False

        
        reward = 0
        if not terminated:
            cur_pos = self.z_pos
            self.action_values.append(self.close_v)
            if action != 0:
                self.total_fee += self.value * self.fee
                self.hold = self.hold or action == 1

            cur_close_v = self.action_values[-1]
            cur_value = self.value
            while True:
                self._update_state()
                if cur_pos != self.z_pos or self.date_index >= self.date_end - 1:
                    if self.hold:
                        if action == 1:
                            self.shares = cur_value * (1 - self.fee) / cur_close_v
                            self.value = self.shares * self.close_v
                        else:
                            self.value = self.shares * self.close_v

                        reward = self.value - cur_value
                    elif action == -1:
                        self.shares = 0
                        reward = -self.value * self.fee
                        self.value *= 1 - self.fee
        
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

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        if self.eval:
            self.date_index = self.date_start -1
            self.value = 1
            self.hold = False
        else:
            self.date_index = (
                self.np_random.integers(self.date_start, self.date_end) - 1
            )
            self.value = 1
            self.hold = self.np_random.integers(self.date_start, self.date_end) % 2 == 1

        self._update_state()
        if self.hold:
            self.value *= (1 - self.fee)
            self.shares = self.value / self.close_v
   
        self.close_init = self.close_v.copy()
        self.total_fee = 0.0
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
            }
        )

    def render(self):
        pass

    def close(self):
        pass
