import argparse
import datetime
import os
import pickle
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--num_iterations', type=int, default=int(1e5))
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_evaluate', type=int, default=int(1e5))
    parser.add_argument('--gradient_type', type=str, default="complete", choices=["complete", "naive"])
    parser.add_argument('--objective', type=str, default="exponential", choices=["cost", "forsyth", "exponential"])
    parser.add_argument('--device', type=str, default="-1", choices=["-1", "0", "1"])
    parser.add_argument('--gamma', type=float, default=0.0)
    parser.add_argument('--exp_param', type=float, default=2e-7)
    return parser.parse_args()


def add_parameters(args):
    """
    Add problem parameters to namespace

    :param args: Namespace to add parameters to
    :return: args: updated Namespace
    """
    args.units = 2e7
    args.s0 = 45
    args.strike = args.s0
    args.end_time = 63
    args.dt = 1
    args.n = args.end_time
    args.market_volume = 4e6
    args.amin = -5 * args.market_volume
    args.amax = 5 * args.market_volume
    args.sigma = 0.6
    args.theta = 0.0
    args.rho = 5
    args.k = 3e-7
    args.eta = 0.1
    args.phi = 0.75
    args.alpha0 = args.units / 2
    args.cash0 = - args.alpha0 * args.s0
    args.normalization = 1e-7
    args.sigmoid_param = np.log(127 / 125)
    return args


def compute_objective(endvals, args, using_torch=True):
    """
    Compute the objective from end values

    :param endvals: array of endvals in torch or numpy
    :param args: Namespace of all parameters
    :param using_torch: boolean of whether to use torch or numpy
    :return: array of values of the objective function
    """
    if args.objective == "forsyth":
        if using_torch:
            gs = torch.square(endvals / args.units - args.gamma / 2)
        else:
            gs = np.square(endvals / args.units - args.gamma / 2)
    elif args.objective == "cost":
        gs = - endvals / args.units
    elif args.objective == "exponential":
        if using_torch:
            gs = torch.exp(-args.exp_param * endvals)
        else:
            gs = np.exp(-args.exp_param * endvals)
    return gs


def execution_cost(rho, args, using_torch):
    if using_torch:
        return args.eta * torch.pow(torch.abs(rho), 1 + args.phi)
    else:
        return args.eta * np.power(np.abs(rho), 1 + args.phi)


def cost_trade(a, s, args, using_torch=True):
    """
    Computes the cost of buying/selling units

    :param a: number of units to buy (if positive) or sell (if negative)
    :param s: current price
    :param args: Namespace of all parameters
    :param using_torch: boolean of whether to use torch or numpy
    :return: total cost
    """
    return a * s + end_costs(a, args, using_torch)


def market_impact(a, args):
    """
    Computes the market impact on the asset price when buying/selling units

    :param a: number of units to buy (if positive) or sell (if negative)
    :param args: Namespace of all parameters
    :return: offset of the price
    """
    return args.k * a


def step(s, x, q, v, args, using_torch=True):
    """
    Updates price, cash and inventory at next timestep when buying/selling units

    :param s: current price
    :param x: current cash
    :param q: current inventory
    :param v: units to buy (if positive) or sell (if negative)
    :param args: Namespace of all parameters
    :param using_torch: boolean of whether to use torch or numpy
    :return: new price, new cash, new inventory
    """
    q1 = q + v
    x1 = x - cost_trade(v, s, args, using_torch)
    if using_torch:
        with torch.no_grad():
            s1 = market_impact(v, args) + s - args.theta * (s - args.s0) + args.sigma * torch.randn(s.shape,
                                                                                                    dtype=s.dtype)
    else:
        s1 = market_impact(v, args) + s - args.theta * (s - args.s0) + args.sigma * np.random.standard_normal(s.shape)
    return s1, x1, q1


def end_costs(q, args, using_torch=True):
    """
    Compute execution costs for discrete trade

    :param q: number of units to trade
    :param args: Namespace of all parameters
    :param using_torch: boolean of whether to use torch or numpy
    :return: execution cost for the trade
    """
    if using_torch:
        q_abs = torch.abs(q)
        return execution_cost(torch.tensor(args.rho), args, using_torch) * q_abs / args.rho + args.k * q_abs * q_abs / 2
    else:
        q_abs = np.abs(q)
        return execution_cost(args.rho, args, using_torch) * q_abs / args.rho + args.k * q_abs * q_abs / 2


def end_value(quantity, prices, cash, args, using_torch=True):
    """
    Computes total cash after exercising of option

    :param quantity: current inventory
    :param prices: current price
    :param cash: current cash
    :param args: Namespace of parameters
    :param using_torch: boolean of whether to use torch or numpy
    :return: cash left after option payoff and making sure zero in terminal inventory
    """
    if using_torch:
        return cash + torch.where(prices >= args.strike,
                                  args.strike * args.units - (args.units - quantity) * prices - end_costs(
                                      quantity - args.units, args, using_torch),
                                  quantity * prices - end_costs(quantity, args, using_torch))
    else:
        return cash + np.where(prices >= args.strike,
                               args.strike * args.units - (args.units - quantity) * prices - end_costs(
                                   quantity - args.units, args, using_torch),
                               quantity * prices - end_costs(quantity, args, using_torch))


def mean_function(x, a, params, args):
    """
    Computes mean for Normal distribution of next price given current price and units bought

    :param x: current price
    :param a: units bought/sold
    :param params: parameter values
    :param args: Namespace of all parameters
    :return: mean value
    """
    return x + params[0] * args.normalization * a


def std_function(params):
    """
    Computes std for Normal distribution of next price

    :param params: parameter values
    :return: std value
    """
    return torch.exp(params[0])


def setup_model(num_features=3, out_features=1, num_hidden=16):
    """
    Setting up neural network to use for policy

    :param num_features: int of number of input features
    :param out_features: int of number of output features
    :param num_hidden: int of number of hidden features
    :return:
    """
    model = torch.nn.Sequential(
        torch.nn.Linear(in_features=num_features, out_features=num_hidden),
        torch.nn.BatchNorm1d(num_features=num_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=num_hidden, out_features=num_hidden),
        torch.nn.BatchNorm1d(num_features=num_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=num_hidden, out_features=out_features)
    )
    return model


def main(args):
    # Setting the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device

    if torch.cuda.is_available() and args.device != "-1":
        print('Using GPU')
        torch.set_default_tensor_type('torch.cuda.DoubleTensor')

    # Define problem setup variables
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args = add_parameters(args)

    print('TRAINING')
    print_freq = 100
    args.grad_offset = 100
    ok = True
    torch_policy_models = []
    best_models = [None] * args.num_runs
    num_iterations = []
    endvals_training = torch.zeros(args.num_runs, args.num_iterations, args.batch_size)
    objective_training = torch.zeros(args.num_runs, args.num_iterations, args.batch_size)

    if args.gradient_type == "complete":
        loss_transitions = torch.zeros(args.num_runs, args.num_iterations)
        prob_num_vars = (1, 1)
        prob_params_sgd = [np.zeros((args.num_runs, args.num_iterations + 1, p)) for p in prob_num_vars]
        for p in range(len(prob_params_sgd)):
            prob_params_sgd[p][:, 0, :] = np.random.uniform(low=-1e-5, high=1e-5,
                                                            size=args.num_runs * prob_num_vars[p]).reshape(
                (args.num_runs, prob_num_vars[p]))

    for i in range(args.num_runs):
        print('run: ', i + 1)
        best_count = 0
        best_gs = torch.inf

        # Setup optimization of policy variables
        policy = setup_model()
        torch_policy_models.append(policy)
        opt_policy = torch.optim.Adam(policy.parameters())  # , lr=0.01)

        # Setup optimization of transition variables
        if args.gradient_type == "complete":
            transition_params = [torch.tensor(p[i, 0, :], requires_grad=True, ) for p in prob_params_sgd]
            opt_transition = torch.optim.Adam(transition_params, lr=1e-2)
        for it in range(args.num_iterations):
            # Print summary
            if not it % print_freq:
                print('iteration: ', it, '/', args.num_iterations)
                if it > 0:
                    print('objectives: ', torch.mean(objective_training[i, it - print_freq:it]).item())
                    print('endvals: ', torch.mean(endvals_training[i, it - print_freq:it]).item())
                    if args.gradient_type == "complete":
                        print('loss_transitions: ', torch.mean(loss_transitions[i, it - print_freq:it]).item())
                        print('transition params: ', transition_params)

            # Simulate episodes
            prices = args.s0 * torch.ones(args.batch_size)
            cash = args.cash0 * torch.ones(args.batch_size)
            quantity = args.alpha0 * torch.ones(args.batch_size)
            if args.gradient_type == "complete":
                logp = torch.zeros(args.batch_size)
            for t in range(args.n):
                actions = policy(
                    torch.stack([(t / args.n) * torch.ones(args.batch_size), (prices - args.s0) / args.s0,
                                 quantity / args.units], axis=-1)).squeeze()
                actions = (args.amax - args.amin) / (1 + torch.exp(-args.sigmoid_param * actions)) + args.amin
                new_prices, cash, quantity = step(prices, cash, quantity, actions, args)
                if args.gradient_type == "complete":
                    mean_params = mean_function(prices, actions, transition_params[0], args)
                    std_params = std_function(transition_params[1])
                    logp = logp + torch.distributions.Normal(loc=mean_params, scale=std_params).log_prob(new_prices)
                prices = new_prices
            endvals = end_value(quantity, prices, cash, args)
            gs = compute_objective(endvals, args)

            # Compute gradient for transition parameters
            if args.gradient_type == "complete":
                loss_transition = -torch.mean(logp)
                opt_transition.zero_grad()
                loss_transition.backward(retain_graph=True)

            # Compute gradient for policy parameters
            if it > args.grad_offset:
                for p in policy.parameters():
                    if p.grad is not None:
                        p.grad.zero_()
                grad_1 = torch.autograd.grad(torch.mean(gs), inputs=policy.parameters(), retain_graph=True)
                if args.gradient_type == "complete":
                    grad_2 = torch.autograd.grad(logp, inputs=policy.parameters(), retain_graph=True,
                                                 grad_outputs=gs.detach() / args.batch_size)
                    zip_args = (policy.parameters(), grad_1, grad_2)
                else:
                    zip_args = (policy.parameters(), grad_1)

                opt_policy.zero_grad()
                for z in zip(*zip_args):
                    if z[0].grad is not None:
                        z[0].grad.zero_()
                    if len(z) == 2:
                        z[0].grad = z[1]
                    else:
                        z[0].grad = z[1] + z[2]
                    if torch.any(torch.isnan(z[0].grad)):
                        print('nans')
                        ok = False
                    else:
                        ok = True
                if ok:
                    torch.nn.utils.clip_grad_value_(policy.parameters(), clip_value=0.1)
                    opt_policy.step()
                else:
                    print('NOT OK')
            if args.gradient_type == "complete":
                opt_transition.step()

            # Store values during training
            if args.gradient_type == "complete":
                loss_transitions[i, it] = loss_transition.detach()
                for p in range(len(prob_params_sgd)):
                    prob_params_sgd[p][i, it + 1, :] = transition_params[p].detach().cpu().numpy()
            endvals_training[i, it, :] = endvals.detach()
            objective_training[i, it, :] = gs.detach()

            if not it % 500:
                print("evaluation")
                print('# evals since current best: ', best_count)
                print("best objective before: ", best_gs)
                # Simulate episodes
                torch.set_default_tensor_type(torch.DoubleTensor)
                prices = args.s0 * torch.ones(args.num_evaluate)
                cash = args.cash0 * torch.ones(args.num_evaluate)
                quantity = args.alpha0 * torch.ones(args.num_evaluate)
                cpu_model = copy.deepcopy(policy).to(torch.device('cpu'))
                for t in range(args.n):
                    actions = cpu_model(
                        torch.stack(
                            [(t / args.n) * torch.ones(args.num_evaluate, device='cpu'), (prices - args.s0) / args.s0,
                             quantity / args.units], axis=-1)).squeeze()
                    actions = (args.amax - args.amin) / (1 + torch.exp(-args.sigmoid_param * actions)) + args.amin
                    prices, cash, quantity = step(prices, cash, quantity, actions, args)
                new_gs = torch.mean(compute_objective(end_value(quantity, prices, cash, args), args))
                if torch.cuda.is_available() and args.device != "-1":
                    torch.set_default_tensor_type('torch.cuda.DoubleTensor')
                print("new objective: ", new_gs)
                if new_gs + 1e-5 < best_gs:
                    best_count = 0
                    best_gs = new_gs
                    best_models[i] = cpu_model
                else:
                    best_count += 1

                if best_count > 5:
                    print("objective not improved for 5 evaluations")
                    num_iterations.append(it)
                    break

    print('EVALUATION')
    torch.set_default_tensor_type(torch.DoubleTensor)
    prices = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    prices[:, 0, :] = args.s0
    cash = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    cash[:, 0, :] = args.cash0
    quantity = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    quantity[:, 0, :] = args.alpha0
    endvals = torch.zeros(args.num_runs, args.num_evaluate)
    gs = torch.zeros(args.num_runs)

    for j in range(args.num_runs):
        print(j + 1)
        cpu_model = best_models[j].to(torch.device('cpu'))
        for t in range(args.n):
            actions = cpu_model(
                torch.stack([(t / args.n) * torch.ones(args.num_evaluate), (prices[j, t, :] - args.s0) / args.s0,
                             quantity[j, t, :] / args.units],
                            axis=-1)).squeeze()
            actions = (args.amax - args.amin) / (1 + torch.exp(-args.sigmoid_param * actions)) + args.amin
            prices[j, t + 1, :], cash[j, t + 1, :], quantity[j, t + 1, :] = step(prices[j, t, :], cash[j, t, :],
                                                                                 quantity[j, t, :], actions, args)
        endvals[j, :] = end_value(quantity[j, args.n, :], prices[j, args.n, :], cash[j, args.n, :], args)
        objectives = compute_objective(endvals[j, :], args)
        gs[j] = torch.mean(objectives).item()
        print('Objective: ', gs[j])
        print('Costs: ', -torch.mean(endvals[j, :]).item())
        print('Std: ', torch.std(endvals[j, :]).item())

    print('Mean objective: ', torch.mean(gs))
    print('Std objective: ', torch.std(gs))
    print('# iterations training: ', num_iterations)

    # Saving the results
    data_dict = args.__dict__
    data_dict["endvals"] = endvals
    data_dict["endvals_training"] = endvals_training.cpu()
    data_dict["objective_training"] = objective_training.cpu()
    data_dict["num_iterations"] = num_iterations

    if args.gradient_type == "complete":
        data_dict["loss_transitions"] = loss_transitions.cpu()
        data_dict["prob_params_sgd"] = prob_params_sgd

    pickle.dump(data_dict,
                open("results/option_hedging_{}_{}_{}.pkl".format(args.gradient_type, args.objective, timestamp), "wb"))

    plt.plot(torch.log(torch.mean(objective_training, axis=-1)).T.cpu().numpy())
    plt.show()


if __name__ == '__main__':
    main(parse_args())
