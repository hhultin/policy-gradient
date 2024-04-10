import argparse
import datetime
import os
import pickle

import gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

from option_hedging import add_parameters, step, end_value, compute_objective

torch.set_default_dtype(torch.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--num_iterations', type=int, default=int(1e5))
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_evaluate', type=int, default=int(1e5))
    parser.add_argument('--device', type=str, default="0", choices=["0", "1", "-1"])
    parser.add_argument('--objective', type=str, default="cost", choices=["cost", "exponential"])
    parser.add_argument('--exp_param', type=float, default=2e-7)
    return parser.parse_args()


class HedgingVecEnv(VecEnv):
    """
    Environment for order execution using the structure of the gym package

    :param args: Namespace of all parameters
    """

    def __init__(self, args, num_envs=1):
        self.current_step = None
        self.shares = None
        self.balance = None
        self.price = None
        self.value = None
        self.args = args
        self.num_envs = num_envs

        self.action_space = gym.spaces.Box(low=-1e3, high=1e3, shape=(1,))
        low = -np.inf * np.ones(3)
        high = np.inf * np.ones(3)
        self.observation_space = gym.spaces.Box(low=low, high=high)
        self.dtype = self.action_space.dtype

    def _get_observation(self):
        obs = np.stack((self.current_step / self.args.n, (self.price - self.args.s0) / self.args.s0,
                        self.shares / self.args.units),
                       axis=-1)
        return obs.astype(self.dtype)

    def step(self, action):
        action = action.squeeze()
        action = (self.args.amax - self.args.amin) / (1 + np.exp(-self.args.sigmoid_param * action)) + self.args.amin
        done = False
        s1, b1, alpha1 = step(self.price, self.balance, self.shares, action, self.args, using_torch=False)
        v1 = compute_objective(end_value(alpha1, s1, b1, self.args, using_torch=False), self.args, using_torch=False)
        reward = -v1 + self.value
        self.current_step += 1
        self.price = s1
        self.balance = b1
        self.shares = alpha1
        self.value = v1
        if self.current_step[0] == self.args.n:
            endval = end_value(self.shares, self.price, self.balance, self.args, using_torch=False)
            reward += -compute_objective(endval, self.args, using_torch=False) + self.value
            self.shares *= 0
            self.balance = endval
            done = True
            last_obs = self._get_observation()
            infos = [{"terminal_observation": last_obs[i, :]} for i in range(self.num_envs)]
            obs = self.reset()
        else:
            infos = [{} for _ in range(self.num_envs)]
            obs = self._get_observation()
        return obs, reward.astype(self.dtype), done * np.ones(self.num_envs, dtype=self.dtype), infos

    def reset(self):
        self.price = self.args.s0 * np.ones(self.num_envs, dtype=self.dtype)
        self.balance = self.args.cash0 * np.ones(self.num_envs, dtype=self.dtype)
        self.shares = self.args.alpha0 * np.ones(self.num_envs, dtype=self.dtype)
        self.current_step = 0 * np.ones(self.num_envs, dtype=self.dtype)
        self.value = compute_objective(end_value(self.shares, self.price, self.balance, self.args, using_torch=False),
                                       self.args, using_torch=False)
        return self._get_observation()

    def close(self):
        return

    def env_is_wrapped(self, *args):
        return [False] * self.num_envs

    def env_method(self):
        return

    def get_attr(self):
        return

    def seed(self):
        return

    def set_attr(self):
        return

    def step_async(self):
        return

    def step_wait(self):
        return


def main(args):
    # Setting the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    # Define problem setup variables
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args = add_parameters(args)

    print('TRAINING')
    models = []
    num_envs = args.batch_size
    env = HedgingVecEnv(args, num_envs=num_envs)
    # Separate evaluation env
    eval_env = HedgingVecEnv(args, num_envs=args.num_evaluate)
    num_timesteps = []

    for j in range(args.num_runs):
        # Stop training if there is no improvement after more than 5 evaluations
        stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=5, verbose=1)
        eval_callback = EvalCallback(eval_env, eval_freq=args.n * 500,
                                     best_model_save_path="./results/models/ppo_hedging_{}_{}".format(timestamp, j),
                                     log_path="./results/models/ppo_hedging_{}_{}".format(timestamp, j), verbose=1,
                                     n_eval_episodes=args.num_evaluate, callback_after_eval=stop_train_callback)
        models.append(
            PPO("MlpPolicy", env, verbose=1, gamma=1, batch_size=args.batch_size,
                tensorboard_log="./ppo_hedging_tensorboard/", n_steps=args.n, n_epochs=1, learning_rate=3e-5))
        print(models[-1].policy)
        models[-1].learn(total_timesteps=args.n * args.num_iterations * args.batch_size, callback=eval_callback,
                         log_interval=100)
        models[-1].set_parameters("./results/models/ppo_hedging_{}_{}/best_model.zip".format(timestamp, j))
        num_timesteps.append(models[-1].num_timesteps)
        models[-1].save("./results/models/ppo_hedging_{}_{}/best_model".format(timestamp, j))

    print('EVALUATION')
    prices = np.zeros((args.num_runs, args.n + 1, args.num_evaluate), dtype=np.float32)
    prices[:, 0, :] = args.s0
    cash = np.zeros((args.num_runs, args.n + 1, args.num_evaluate), dtype=np.float32)
    cash[:, 0, :] = args.cash0
    quantity = np.zeros((args.num_runs, args.n + 1, args.num_evaluate), dtype=np.float32)
    quantity[:, 0, :] = args.alpha0
    endvals = np.zeros((args.num_runs, args.num_evaluate), dtype=np.float32)
    gs = np.zeros(args.num_runs, dtype=np.float32)

    for j in range(args.num_runs):
        print(j + 1)
        for t in range(args.n):
            obs = np.stack(
                [(t / args.n) * np.ones(args.num_evaluate), (prices[j, t, :] - args.s0) / args.s0,
                 quantity[j, t, :] / args.units],
                axis=-1).astype(np.float32)
            actions, _states = models[j].predict(obs, deterministic=True)
            actions = (args.amax - args.amin) / (1 + np.exp(-args.sigmoid_param * actions)) + args.amin
            prices[j, t + 1, :], cash[j, t + 1, :], quantity[j, t + 1, :] = step(prices[j, t, :], cash[j, t, :],
                                                                                 quantity[j, t, :], actions.squeeze(),
                                                                                 args,
                                                                                 using_torch=False)
        endvals[j, :] = end_value(quantity[j, args.n, :], prices[j, args.n, :], cash[j, args.n, :], args,
                                  using_torch=False)
        gs[j] = np.mean(compute_objective(endvals[j, :], args, using_torch=False))
        print('Objective: ', gs[j])
        print('Costs: ', -np.mean(endvals[j, :]))
        print('Std: ', np.std(endvals[j, :]))

    print('Mean objective: ', np.mean(gs))
    print('Std objective: ', np.std(gs))
    print('# timesteps in training: ', num_timesteps)

    # Saving the results
    data_dict = args.__dict__
    data_dict["endvals"] = endvals
    data_dict["quantity"] = quantity
    data_dict["num_timesteps"] = num_timesteps

    pickle.dump(data_dict,
                open("results/ppo_option_hedging_{}_{}.pkl".format(args.objective, timestamp), "wb"))


if __name__ == '__main__':
    main(parse_args())
