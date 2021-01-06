import argparse
import os
import sys

import pandas as pd

import ray
from ray.tune.registry import register_env
from ray.rllib.agents import ppo
from ray.rllib.models import ModelCatalog
from ray import tune

from env.StockTradingEnvironment import StockTradingEnvironment

###############################################
## Command line args
###############################################
parser = argparse.ArgumentParser(description="Script for training RLLIB agents")
parser.add_argument("--num-cpus", type=int, default=1)
parser.add_argument("--num-gpus", type=int, default=0)
parser.add_argument("--tune-log-level", type=str, default="INFO")
parser.add_argument("--env-logging", action="store_true")
parser.add_argument("--redis-password", type=str, default=None)
parser.add_argument("--ip_head", type=str, default=None)
parser.add_argument("--restore", type=str, default=None)
args = parser.parse_args()


################################################
## RLLIB SETUP (should work for most use cases)
################################################
if args.redis_password is None:
    ray.services.get_node_ip_address = lambda: '127.0.0.1'
    ray.init(local_mode=True,temp_dir="/tmp/scratch/ray")#
else:
    assert args.ip_head is not None
    ray.init(redis_password=args.redis_password, address=args.ip_head)

#############################################
## LOAD AND CONFIGURE YOUR PROBLEM INSTANCE 
#############################################
env_config = {
    "df": pd.read_csv("data/AAPL.csv", index_col=[0]),
    "MAX_ACCOUNT_BALANCE": 2147483647,
    "MAX_NUM_SHARES": 2147483647,
    "MAX_SHARE_PRICE": 5000,
    "MAX_OPEN_POSITION": 5,
    "MAX_STEPS": 20000,
    "INITIAL_ACCOUNT_BALANCE": 10000,
}
print("env_config: ", env_config)

env_name = "StockTrading_env"
register_env(env_name, lambda config: StockTradingEnvironment(**env_config))



######################################
## Run TUNE Experiments!
######################################
tune.run(
    "PPO",
    name=env_name,
    checkpoint_freq=3,
    checkpoint_at_end=True,
    checkpoint_score_attr="episode_reward_mean",
    keep_checkpoints_num=50,
    stop={"training_iteration": 10000},
    restore=args.restore,
    config={
        "env": env_name,
        "num_workers": args.num_cpus, 
        "num_gpus": args.num_gpus,
        "log_level": args.tune_log_level,
        "train_batch_size": 4000,
        "ignore_worker_failures": True,
        }
    )
