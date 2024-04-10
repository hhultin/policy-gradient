import argparse
import datetime
import pickle

import numpy as np
import torch

from option_hedging import add_parameters, step, end_value

torch.set_default_dtype(torch.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--num_evaluate', type=int, default=int(1e5))
    return parser.parse_args()


def main(args):
    # Setting the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Define problem setup variables
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args = add_parameters(args)

    print('NO HEDGING')
    prices = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    prices[:, 0, :] = args.s0
    cash = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    cash[:, 0, :] = args.cash0
    quantity = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    quantity[:, 0, :] = args.alpha0
    endvals = torch.zeros(args.num_runs, args.num_evaluate)

    for j in range(args.num_runs):
        print(j + 1)
        for t in range(args.n):
            actions = torch.tensor(0.0)
            prices[j, t + 1, :], cash[j, t + 1, :], quantity[j, t + 1, :] = step(
                prices[j, t, :], cash[j, t, :], quantity[j, t, :], actions, args)
        endvals[j, :] = end_value(quantity[j, args.n, :], prices[j, args.n, :], cash[j, args.n, :], args)
        print('Gain: ', torch.mean(endvals[j, :]).item())
        print('Std: ', torch.std(endvals[j, :]).item())

    # Saving the results
    data_dict = args.__dict__
    data_dict["endvals"] = endvals
    pickle.dump(data_dict, open("results/no_hedging_option_hedging_{}.pkl".format(timestamp), "wb"))

    print('DELTA HEDGING')
    prices = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    prices[:, 0, :] = args.s0
    cash = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    cash[:, 0, :] = args.cash0
    quantity = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    quantity[:, 0, :] = args.alpha0
    endvals = torch.zeros(args.num_runs, args.num_evaluate)
    dist = torch.distributions.Normal(loc=0, scale=1)

    for j in range(args.num_runs):
        print(j + 1)
        for t in range(args.n):
            d = (prices[j, t, :] - args.strike) / (args.sigma * np.sqrt(args.end_time - t * args.dt))
            deltas = dist.cdf(d)
            actions = deltas * args.units - quantity[j, t, :]
            prices[j, t + 1, :], cash[j, t + 1, :], quantity[j, t + 1, :] = step(
                prices[j, t, :], cash[j, t, :], quantity[j, t, :], actions, args)
        endvals[j, :] = end_value(quantity[j, args.n, :], prices[j, args.n, :], cash[j, args.n, :], args)
        print('Gain: ', torch.mean(endvals[j, :]).item())
        print('Std: ', torch.std(endvals[j, :]).item())

    # Saving the results
    data_dict = args.__dict__
    data_dict["endvals"] = endvals
    pickle.dump(data_dict, open("results/delta_option_hedging_{}.pkl".format(timestamp), "wb"))


if __name__ == '__main__':
    main(parse_args())
