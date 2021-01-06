# Stock Trading with Reinforcement Learning

This repo uses RLlib for training an agent to trade stocks. The environment is based on the one found [here](https://github.com/notadamking/Stock-Trading-Environment). 


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

