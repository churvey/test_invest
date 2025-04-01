# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from trade.model.env import TradeEnv
import pandas as pd


from gymnasium.envs.registration import register

# register(
#     id="TradeEnv-v0",
#     entry_point=make_trade_env,  # 关键！entry_point 指向可调用对象
#     kwargs={},  # 可选：默认参数
# )


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "TradeEnv-v0"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 20
    """the number of parallel game environments"""
    num_steps: int = 20
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    weight_as_feature: bool = True

    fee: float = 0.0001


def _get_data():
    path = [
        "tmp/SH.513050-中概互联网ETF.csv",
        # "data/K_DAY/SH.515290-银行ETF天弘.csv",
    ]

    df = pd.concat([pd.read_csv(p) for p in path])
    # columns = "code,name,time_key,open,close,high,low,pe_ratio,turnover_rate,volume,turnover,change_rate,last_close"
    columns = "code,time_key,open,close,high,low,volume".split(",")
    columns_rename = "instrument,datetime,open,close,high,low,volume".split(",")
    df = df[columns]
    df.columns = columns_rename

    keys = "open,close,high,low,volume".split(",")
    all_inputs = {
        k: (
            df[k].to_numpy(dtype="double")
            if k not in ["instrument", "datetime"]
            else df[k].to_numpy()
        )
        for k in df.columns
    }
    inputs = {k: all_inputs[k] for k in keys}

    import talib
    from talib import abstract

    print(talib.get_functions())

    print(len(talib.get_functions()))

    for f in talib.get_functions():
        # print(f)
        if f in ["AD", "OBV"]:
            continue

        func = abstract.Function(f)
        if "timeperiod" in func.info["parameters"]:
            outputs = func(inputs, timeperiod=20)
        elif "periods" not in func.info["input_names"]:
            outputs = func(inputs)
        else:
            # skip
            print(f"skip func {f}")
            continue
        # outputs = dict(zip(func.info["output_names"], outputs))
        output_names = [f"{f}_{o}" for o in func.info["output_names"]]
        outputs = dict(zip(output_names, outputs))
        outputs = {
            k: v
            for k, v in outputs.items()
            if not np.all(np.isnan(v)) and v.dtype == np.float64
        }
        # print(output_names)
        all_inputs.update(outputs)
    df = pd.DataFrame(all_inputs)
    df = df.dropna()
    print(f"columns:{len(df.columns)}")
    print(df)
    return df
    # df = df[columns]


def get_data():
    path = [
        "tmp/SH.513050-中概互联网ETF.csv",
        "tmp/SH.515290-银行ETF天弘.csv",
    ]

    df = pd.concat([pd.read_csv(p) for p in path])
    # columns = "code,name,time_key,open,close,high,low,pe_ratio,turnover_rate,volume,turnover,change_rate,last_close"
    columns = "code,time_key,open,close,high,low,volume,change_rate".split(",")
    columns_rename = "instrument,datetime,open,close,high,low,volume,change".split(",")
    df = df[columns]
    df.columns = columns_rename

    # keys = "open,close,high,low,volume".split(",")
    all_inputs = {k: df[k].to_numpy() for k in df.columns}

    from trade.data.feature.feature import Feature

    data = Feature(all_inputs)()
    df = pd.DataFrame(data)
    df = df.dropna()
    print(df["z_pos_20"].to_list())
    # 1/0
    print(f"columns:{len(df.columns)}")
    print(df)
    return df


def make_env(env_id="TradeEnv-v0", df=None, **kwargs):

    def thunk():
        env = gym.make(env_id, df=df, **kwargs)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        from trade.model.zoo import Net

        self.critic = Net(np.array(envs.single_observation_space.shape).prod())

        self.actor = Net(
            np.array(envs.single_observation_space.shape).prod(),
            envs.single_action_space.n,
        )

    def get_value(self, x):
        x = x.reshape([-1, x.shape[-1]])
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        x = x.reshape([-1, x.shape[-1]])
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)


def train(args, df, agent, date_range, run_name):
    agent.train()

    writer = SummaryWriter(f"runs/{run_name}")

    # TRY NOT TO MODIFY: seeding

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                args.env_id,
                df=df,
                # date_range=("2021-02-01 00:00:00", "2024-01-01 00:00:00"),
                # date_range=("2024-01-01 00:00:00", "2025-01-01 00:00:00"),
                date_range=date_range,
                max_length=360,
                weight_as_feature=args.weight_as_feature,
                fee=args.fee,
                # eval=True,
            )
            for i in range(args.num_envs)
        ],
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Discrete
    ), "only discrete action space is supported"

    #
    print(f"agent {agent}")
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    # ALGO Logic: Storage setup
    obs = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape
    ).to(device)
    actions = torch.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape
    ).to(device)

    print(
        "envs.single_action_space.shape",
        envs.single_action_space.shape,
        envs.single_action_space.n,
    )

    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                # print("action", action.shape, actions.shape)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(
                next_done
            ).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(
                            f"global_step={global_step}, episodic_return={info['episode']['r']}"
                        )
                        writer.add_scalar(
                            "charts/episodic_return", info["episode"]["r"], global_step
                        )
                        writer.add_scalar(
                            "charts/episodic_length", info["episode"]["l"], global_step
                        )
                        # writer.add_scalar(
                        #     "charts/addition", info["addition"], global_step
                        # )
                        # writer.add_scalar(
                        #     "charts/reward_without_addition", info["episode"]["r"] - info["addition"], global_step
                        # )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar(
            "charts/learning_rate", optimizer.param_groups[0]["lr"], global_step
        )
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print(
            "SPS:",
            int(global_step / (time.time() - start_time)),
            f" {iteration} / {args.num_iterations + 1}",
        )
        writer.add_scalar(
            "charts/SPS", int(global_step / (time.time() - start_time)), global_step
        )

    envs.close()


def eval(args, df, agent, date_range, run_name):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    agent.eval()
    # for action, action_name in {0: "norm", 1: "max", -1: "min", -2: "rand"}.items():
    for action, action_name in {0: "norm"}.items():
        writer = SummaryWriter(f"runs/{run_name}_a_{action_name}")
        eval_env = make_env(
            args.env_id,
            df=df,
            # date_range=("2024-01-01 00:00:00", "2025-01-01 00:00:00"),
            date_range=date_range,
            weight_as_feature=args.weight_as_feature,
            fee=args.fee,
            override_action=action,
            eval=True,
        )()

        obs, infos = eval_env.reset()
        termindate = False
        while not termindate:
            days = infos["days"]
            if days % 10 == 0:
                print(infos)
            # names = infos["names"]
            tag_scalar_dict = {
                "value": infos["value"],
            }
            if action == 0:
                tag_scalar_dict.update({f"close_norm": infos["close_norm"]})
            writer.add_scalars(
                main_tag="cmp",  # 主标签（图表标题）
                tag_scalar_dict=tag_scalar_dict,
                global_step=infos["days"],
            )

            writer.add_scalar("return/reward", infos["reward"], infos["days"])
            writer.add_scalar("return/total_fee", infos["total_fee"], infos["days"])
            # writer.add_scalar(
            #     "return/max_draw_back", infos["max_draw_back"], infos["days"]
            # )

            # writer.add_scalars(
            #     main_tag="eval",  # 主标签（图表标题）
            #     tag_scalar_dict={
            #         **{
            #             f"ratio_{names[i]}": infos["ratio"][i]
            #             for i in range(len(names))
            #         },
            #         **{
            #             "addition": infos["addition"]
            #         }
            #     },
            #     global_step=infos["days"],
            # )
            actions, _, _, _ = agent.get_action_and_value(torch.Tensor(obs).to(device))
            next_obs, _, termindate, _, infos = eval_env.step(actions.cpu().numpy()[0])
            obs = next_obs
        eval_env.close()
        writer.close()


if __name__ == "__main__":
    register(
        id="TradeEnv-v0",
        # entry_point="trade.model.env:TradeEnv",
        entry_point="trade.model.boll_env:BollTradeEnv",
    )
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    df = get_data()
    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                df=df.copy(),
                # date_range=("2021-02-01 00:00:00", "2024-01-01 00:00:00"),
                max_length=360,
                weight_as_feature=args.weight_as_feature,
                fee=args.fee,
            )
            for i in range(args.num_envs)
        ],
    )
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    dates = make_env(
        args.env_id,
        df=df.copy(),
        max_length=360,
        weight_as_feature=args.weight_as_feature,
        fee=args.fee,
    )().dates

    train_len = 360 * 4
    eval_len = 360
    
    if train_len + eval_len >= len(dates):
        train_len = len(dates) - eval_len - 1

    roll_num = (len(dates) - train_len) // eval_len
    print("roll_num", roll_num)
    time_start = int(time.time())
    for i in range(roll_num):
        agent = Agent(envs).to(device)

        def to_date(p):
            return (dates[p[0]], dates[p[1]])

        train_range = i * eval_len, i * eval_len + train_len
        train_range = to_date(train_range)
        eval_ranges = [
            (i * eval_len + train_len - eval_len, i * eval_len + train_len),
            (i * eval_len + train_len, i * eval_len + train_len + eval_len),
        ]
        eval_ranges = [to_date(r) for r in eval_ranges]
        print("eval ranges:", eval_ranges)
        run_name = (
            f"{args.env_id}__{args.exp_name}__{args.seed}__roll_{i}__{time_start}"
        )
        train(args, df.copy(), agent, train_range, run_name)

        for j, eval_range in enumerate(eval_ranges):
            run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__roll_{i}_eval_{j}__{time_start}"
            eval(args, df.copy(), agent, eval_range, run_name)
            # break
        # break
    envs.close()
