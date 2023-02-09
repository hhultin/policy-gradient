import argparse
import datetime
import pickle
import copy

import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def parse_args():
    """
    Parse command line arguments
    :return: Namespace object
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=5)
    parser.add_argument('--num_iterations', type=int, default=int(1e5))
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--num_evaluate', type=int, default=int(1e5))
    parser.add_argument('--gradient_type', type=str, default="complete", choices=["complete", "naive"])
    parser.add_argument('--cost_type', type=str, default="forsyth", choices=["linear", "forsyth"])
    parser.add_argument('--objective', type=str, default="forsyth", choices=["cost", "forsyth", "exponential"])
    parser.add_argument('--gamma', type=float, default=200.0)
    parser.add_argument('--exp_param', type=float, default=0.1)

    return parser.parse_args()


def add_parameters(args):
    """
    Add problem parameters to namespace

    :param args: Namespace to add parameters to
    :return: args: updated Namespace
    """
    args.alpha0 = 1
    args.s0 = 100
    args.end_time = 1 / 250
    args.n = 25
    args.dt = args.end_time / args.n
    args.delta = args.dt
    args.sigma = 1.0
    args.eta = 0.0
    args.r = 0.0
    args.kp_s = 0.01 if args.cost_type == 'forsyth' else 0.0
    args.kp_0 = 0 if args.cost_type == 'forsyth' else 0.01 * args.s0
    args.kt = 2e-6 if args.cost_type == 'forsyth' else args.kp_0
    args.ks = 0.0
    args.beta = 1.0
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
            gs = torch.square(endvals - args.gamma / 2)
        else:
            gs = np.square(endvals - args.gamma / 2)
    elif args.objective == "cost":
        gs = args.s0 * args.alpha0 - endvals
    elif args.objective == "exponential":
        if using_torch:
            gs = torch.exp(args.exp_param * (args.s0 * args.alpha0 - endvals))
        else:
            gs = np.exp(args.exp_param * (args.s0 * args.alpha0 - endvals))
    return gs


def f(v, args, using_torch):
    """
    Computes temporary market impact according to Forsyth

    :param v: trading rate
    :param args: Namespace of parameters
    :param using_torch: boolean of whether to use torch or numpy
    :return: multiplier for temporary market impact
    """
    if using_torch:
        x = (1 + args.ks * torch.sign(v)) * torch.exp(args.kt * torch.sign(v) * torch.pow(torch.abs(v), args.beta))
    else:
        x = (1 + args.ks * np.sign(v)) * np.exp(args.kt * np.sign(v) * np.power(np.abs(v), args.beta))
    return x


def step(a, s, b, alpha, args, using_torch):
    """
    Updates price, cash and inventory at next timestep when buying/selling units

    :param a: action
    :param s: current price
    :param b: current balance
    :param alpha: current inventory
    :param args: Namespace of parameters
    :param using_torch: bool of whether to use torch or numpy
    :return: new price, new balance, new inventory
    """
    if args.cost_type == 'forsyth':
        b1 = b * (1 + args.r * args.dt) - a * s * f(a / args.delta, args, using_torch)
    elif args.cost_type == 'linear':
        b1 = b * (1 + args.r * args.dt) - a * (s + args.kt * a)
    alpha1 = alpha + a

    if using_torch:
        with torch.no_grad():
            s1 = (s * (1 + args.kp_s * a) + args.kp_0 * a) * torch.exp(
                (args.eta - args.sigma * args.sigma / 2) * args.dt + args.sigma * np.sqrt(args.dt) * torch.randn(
                    s.shape, dtype=s.dtype))
    else:
        if type(s) == np.ndarray:
            w = np.random.standard_normal(s.shape)
        else:
            w = np.random.standard_normal()
        s1 = (s * (1 + args.kp_s * a) + args.kp_0 * a) * np.exp(
            (args.eta - args.sigma * args.sigma / 2) * args.dt + args.sigma * np.sqrt(
                args.dt) * w)
    return s1, b1, alpha1


def end_value(s, b, alpha, args, using_torch):
    """
    Computes total cash after making sure terminal constraint of number of units met

    :param s: current price
    :param b: current balance
    :param alpha: current inventory
    :param args: Namespace of parameters
    :param using_torch: bool of whether to use torch or numpy
    :return: cash left after making sure zero in inventory
    """
    if args.cost_type == 'forsyth':
        return b + alpha * s * f(-alpha / args.delta, args, using_torch)
    else:
        return b + alpha * (s - args.kt * alpha)


def mean_function(x, a, params, args):
    """
    Computes mean for Normal distribution of next price given current price and units bought

    :param x: current price
    :param a: units bought/sold
    :param params: parameter values
    :param args: Namespace of parameters
    :return: mean value
    """
    if args.cost_type == 'forsyth':
        return torch.log(x * (1 + params[0] * a)) + params[1] * args.dt
    else:
        return torch.log(x + params[0] * a) + params[1] * args.dt


def std_function(params, args):
    """
    Computes std for Normal distribution of next price

    :param params: parameter values
    :param args: Namespace of parameters
    :return: std value
    """
    return torch.exp(params[0]) * np.sqrt(args.dt)


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
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=num_hidden, out_features=num_hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(in_features=num_hidden, out_features=out_features)
    )
    return model


def main(args):
    # Setting the seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)

    # Define problem setup variables
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    args = add_parameters(args)

    print('TRAINING')
    print_freq = 100
    args.grad_offset = 100
    torch_policy_models = []
    best_models = [None] * args.num_runs
    num_iterations = []
    loss_transitions = torch.zeros(args.num_runs, args.num_iterations)
    objective_training = torch.zeros(args.num_runs, args.num_iterations, args.batch_size)

    prob_num_vars = (2, 1)
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
        opt_policy = torch.optim.Adam(policy.parameters())

        # Setup optimization of transition variables
        transition_params = [torch.tensor(p[i, 0, :], requires_grad=True, ) for p in prob_params_sgd]
        opt_transition = torch.optim.Adam(transition_params, lr=1e-2)

        for it in range(args.num_iterations):
            # Print summary
            if not it % print_freq:
                print('iteration: ', it)
                if it > 0:
                    print('vals: ', torch.mean(objective_training[i, it - print_freq:it]).item())
                    print('loss_transitions: ', torch.mean(loss_transitions[i, it - print_freq:it]).item())
                    print('transition params: ', transition_params)

            # Simulate episodes
            prices = args.s0 * torch.ones(args.batch_size)
            cash = torch.zeros(args.batch_size)
            quantity = args.alpha0 * torch.ones(args.batch_size)
            logp = torch.zeros(args.batch_size)
            for t in range(args.n):
                actions = policy(
                    torch.stack([(t / args.n) * torch.ones(args.batch_size), (prices - args.s0) / args.s0, quantity],
                                axis=-1)).squeeze()
                actions = - (1 / (1 + torch.exp(-actions))) * quantity
                new_prices, cash, quantity = step(actions, prices, cash, quantity, args, using_torch=True)
                mean_params = mean_function(prices, actions, transition_params[0], args)
                std_params = std_function(transition_params[1], args)
                logp = logp + torch.distributions.LogNormal(loc=mean_params, scale=std_params).log_prob(new_prices)

                prices = new_prices

            gs = compute_objective(end_value(prices, cash, quantity, args, using_torch=True), args)

            # Compute gradient for transition parameters
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
            opt_transition.step()

            # Store values during training
            loss_transitions[i, it] = loss_transition.detach()
            objective_training[i, it, :] = gs.detach()
            for p in range(len(prob_params_sgd)):
                prob_params_sgd[p][i, it + 1, :] = transition_params[p].detach().numpy()

            if not it % 500:
                print("evaluation")
                print('# evals since current best: ', best_count)
                print("best objective before: ", best_gs)
                # Simulate episodes
                prices = args.s0 * torch.ones(args.num_evaluate)
                cash = torch.zeros(args.num_evaluate)
                quantity = args.alpha0 * torch.ones(args.num_evaluate)
                for t in range(args.n):
                    actions = policy(
                        torch.stack(
                            [(t / args.n) * torch.ones(args.num_evaluate), (prices - args.s0) / args.s0, quantity],
                            axis=-1)).squeeze()
                    actions = - (1 / (1 + torch.exp(-actions))) * quantity
                    prices, cash, quantity = step(actions, prices, cash, quantity, args, using_torch=True)
                new_gs = torch.mean(compute_objective(end_value(prices, cash, quantity, args, using_torch=True), args))
                print("new objective: ", new_gs)
                if new_gs + 1e-5 < best_gs:
                    best_count = 0
                    best_gs = new_gs
                    best_models[i] = copy.deepcopy(policy)
                else:
                    best_count += 1

                if best_count > 5:
                    print("objective not improved for 5 evaluations")
                    num_iterations.append(it)
                    break

        print(transition_params)
    print('EVALUATION')
    prices = args.s0 * torch.ones(args.num_runs, args.n + 1, args.num_evaluate)
    cash = torch.zeros(args.num_runs, args.n + 1, args.num_evaluate)
    quantity = args.alpha0 * torch.ones(args.num_runs, args.n + 1, args.num_evaluate)
    endvals = torch.zeros(args.num_runs, args.num_evaluate)
    gs = torch.zeros(args.num_runs)

    for j in range(args.num_runs):
        print(j + 1)
        cpu_model = best_models[j].to(torch.device('cpu'))
        for t in range(args.n):
            actions = cpu_model(torch.stack(
                [(t / args.n) * torch.ones(args.num_evaluate), (prices[j, t, :] - args.s0) / args.s0,
                 quantity[j, t, :]], axis=-1)).squeeze()
            actions = - (1 / (1 + torch.exp(-actions))) * quantity[j, t, :]
            prices[j, t + 1, :], cash[j, t + 1, :], quantity[j, t + 1, :] = step(actions,
                                                                                 prices[j, t, :], cash[j, t, :],
                                                                                 quantity[j, t, :], args,
                                                                                 using_torch=True)
        endvals[j, :] = end_value(prices[j, args.n, :], cash[j, args.n, :], quantity[j, args.n, :], args,
                                  using_torch=True)
        print('Prices: ', prices[j, args.n, :])
        print('Cash: ', cash[j, args.n, :])
        print('Quantity: ', quantity[j, args.n, :])
        gs[j] = torch.mean(compute_objective(endvals[j, :], args)).item()
        print('Objective: ', gs[j])
        print('Gain: ', torch.mean(endvals[j, :]).item())
        print('Std: ', torch.std(endvals[j, :]).item())

    print('Mean objective: ', torch.mean(gs))
    print('Std objective: ', torch.std(gs))
    print('# iterations training: ', num_iterations)

    # Saving the results
    data_dict = args.__dict__
    data_dict["endvals"] = endvals
    data_dict["quantity"] = quantity
    data_dict["loss_transitions"] = loss_transitions
    data_dict["objective_training"] = objective_training
    data_dict["prob_params_sgd"] = prob_params_sgd
    data_dict["num_iterations"] = num_iterations
    pickle.dump(data_dict,
                open(
                    "results/order_execution_{}_{}_{}_{}.pkl".format(args.gradient_type, args.objective, args.cost_type,
                                                                     timestamp), "wb"))

    plt.plot((torch.mean(objective_training, axis=-1)).T.cpu().numpy())
    plt.figure()
    plt.plot(torch.mean(quantity, axis=(0, -1)).cpu().detach().numpy().squeeze())
    plt.show()


if __name__ == '__main__':
    main(parse_args())
