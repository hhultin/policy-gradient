import argparse
import datetime
import pickle

import numpy as np
import torch

from order_execution import add_parameters, step, end_value


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--num_evaluate', type=int, default=int(1e5))
    parser.add_argument('--cost_type', type=str, default="forsyth", choices=["linear", "forsyth"])
    return parser.parse_args()


def main(args):
    # Setting the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)

    # Define problem setup variables
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args = add_parameters(args)

    print('TWAP')
    twap_action = torch.tensor(-args.alpha0 / (args.n + 1))
    print(twap_action)
    prices = args.s0 * torch.ones(args.num_runs, args.n + 1, args.num_evaluate)
    cash = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    quantity = args.alpha0 * torch.ones(args.num_runs, args.n + 1, args.num_evaluate)
    endvals = torch.zeros(args.num_runs, args.num_evaluate)

    for j in range(args.num_runs):
        print(j + 1)
        for t in range(args.n):
            prices[j, t + 1, :], cash[j, t + 1, :], quantity[j, t + 1, :] = step(twap_action,
                                                                                 prices[j, t, :], cash[j, t, :],
                                                                                 quantity[j, t, :], args,
                                                                                 using_torch=True)
        endvals[j, :] = end_value(prices[j, args.n, :], cash[j, args.n, :], quantity[j, args.n, :], args,
                                  using_torch=True)

        print('Gain: ', torch.mean(endvals[j, :]).item())
        print('Std: ', torch.std(endvals[j, :]).item())

    # Saving the results for the TWAP strategy
    data_dict = args.__dict__
    data_dict["endvals"] = endvals
    pickle.dump(data_dict,
                open("results/twap_order_execution_{}_{}.pkl".format(args.cost_type, timestamp), "wb"))

    print('NO TRADING')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)
    no_action = torch.tensor(0.0)
    prices = args.s0 * torch.ones(args.num_runs, args.n + 1, args.num_evaluate)
    cash = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    quantity = args.alpha0 * torch.ones(args.num_runs, args.n + 1, args.num_evaluate)
    endvals = torch.zeros(args.num_runs, args.num_evaluate)

    for j in range(args.num_runs):
        print(j + 1)
        for t in range(args.n):
            prices[j, t + 1, :], cash[j, t + 1, :], quantity[j, t + 1, :] = step(no_action,
                                                                                 prices[j, t, :], cash[j, t, :],
                                                                                 quantity[j, t, :], args,
                                                                                 using_torch=True)
        endvals[j, :] = end_value(prices[j, args.n, :], cash[j, args.n, :], quantity[j, args.n, :], args,
                                  using_torch=True)

        print('Gain: ', torch.mean(endvals[j, :]).item())
        print('Std: ', torch.std(endvals[j, :]).item())

    # Saving the results for the no trading strategy
    data_dict = args.__dict__
    data_dict["endvals"] = endvals
    pickle.dump(data_dict,
                open("results/no_trading_order_execution_{}_{}.pkl".format(args.cost_type, timestamp), "wb"))

    print('ALL TRADING')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)
    no_action = torch.tensor(0.0)
    all_action = torch.tensor(-args.alpha0)
    prices = args.s0 * torch.ones(args.num_runs, args.n + 1, args.num_evaluate)
    cash = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    quantity = args.alpha0 * torch.ones(args.num_runs, args.n + 1, args.num_evaluate)
    endvals = torch.zeros(args.num_runs, args.num_evaluate)

    for j in range(args.num_runs):
        print(j + 1)
        for t in range(args.n):
            action = no_action if t > 0 else all_action
            prices[j, t + 1, :], cash[j, t + 1, :], quantity[j, t + 1, :] = step(action,
                                                                                 prices[j, t, :], cash[j, t, :],
                                                                                 quantity[j, t, :], args,
                                                                                 using_torch=True)
        endvals[j, :] = end_value(prices[j, args.n, :], cash[j, args.n, :], quantity[j, args.n, :], args,
                                  using_torch=True)

        print('Gain: ', torch.mean(endvals[j, :]).item())
        print('Std: ', torch.std(endvals[j, :]).item())

    # Saving the results for the all trading strategy
    data_dict = args.__dict__
    data_dict["endvals"] = endvals
    pickle.dump(data_dict,
                open("results/all_trading_order_execution_{}_{}.pkl".format(args.cost_type, timestamp), "wb"))


if __name__ == '__main__':
    main(parse_args())
