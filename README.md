# Deterministic policy gradient method

The repo contains scripts for training a deterministic policy gradient method for trading execution ([order_execution.py](https://github.com/hhultin/policy-gradient/blob/main/order_execution.py)) and
option hedging  ([option_hedging.py](https://github.com/hhultin/policy-gradient/blob/main/option_hedging.py)) in the presence of market impact. 

Usage example:
```
python order_execution.py --gradient_type naive --cost_type linear --objective exponential --exp_param 2e-7
```

There are also scripts for training PPO agents ([ppo_order_execution.py](https://github.com/hhultin/policy-gradient/blob/main/ppo_order_execution.py) and [ppo_option_hedging.py](https://github.com/hhultin/policy-gradient/blob/main/ppo_option_hedging.py)) and simulating baseline strategies ([baselines_order_execution.py](https://github.com/hhultin/policy-gradient/blob/main/baselines_order_execution.py) and [baselines_option_hedging.py](https://github.com/hhultin/policy-gradient/blob/main/baselines_option_hedging.py)) for the same problem setups. 

The two notebooks, [Order Execution Evaluation](https://github.com/hhultin/policy-gradient/blob/main/Order%20Execution%20Evaluation.ipynb) and [Option Hedging Evaluation](https://github.com/hhultin/policy-gradient/blob/main/Option%20Hedging%20Evaluation.ipynb), are used for evaluation of the results. 
