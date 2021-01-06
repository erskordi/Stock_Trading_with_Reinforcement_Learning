# Stock Trading with Reinforcement Learning

This repo uses RLlib for training an agent to trade stocks. The environment is based on the one found [here](https://github.com/notadamking/Stock-Trading-Environment). For RL algorithm I used [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)


The environment is slightly changed to work with RLlib, please see `StockTradingEnvironment.py`.


## Necessary packages

Whether you use an already existing Anaconda environment, or if you create a dedicated one, make sure you install `Ray/RLlib`:

```
pip install ray[rllib, debug]
```

You are now ready to run experiments!

The `rllib_trainer.py` uses the `argparse` package to define the number of CPUs/GPUs to use. They are controlled by the following arguments:

```
--num-cpus
--num-gpus
```
# TODOs:

 - Use policy network other than the default fully-connected deep neural network.
  - Perhaps LSTMs could yield better results due to the time-series nature of the problem.
 - Develop a `rllib_eval.py` script where one could evaluate the quality of the trained agent.
