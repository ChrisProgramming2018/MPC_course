import os
import json
from datetime import datetime
import numpy as np
import argparse
from race_car_env import RaceCarEnv
from tqc_agent import TQC


def train_agent(config):
    """ """

    env = RaceCarEnv()
    state = env.reset()
    action = env.action_space.sample()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    print("obs", state_dim)
    print("act", action_dim)
    print("angel ", env.env_options.max_angle)
    config["target_entropy"] = -np.prod(action_dim)
    policy = TQC(state_dim, action_dim, config)
    # train = True
    train = False
    if train:
        policy.train_agent(env)

    # model_path = os.path.join(os.getcwd(), "models/model-550-rr_v1") # without progression reward
    # model_path = os.path.join(os.getcwd(), "models/model-800") # blocking velocity of other vehicle
    model_path = os.path.join(os.getcwd(), "models/model-900")  # blocking plus progress
    policy.eval_agent(env, model_path)


def main(args):
    """ Starts different tests
    Args:
        param1(args): args
    """
    path = os.path.join(os.path.expanduser("~"), 'experiments_race_car')
    now = datetime.now()
    dt_string = now.strftime("%d_%m_%Y_%H:%M:%S")
    path = os.path.join(path, dt_string)
    # experiment_name = args.experiment_name
    res_path = os.path.join(path, "results")
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    with open(args.param, "r") as f:
        param = json.load(f)
    param["locexp"] = path
    train_agent(param)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--param', default="param.json", type=str)
    parser.add_argument('--locexp', default="experiments/kuka", type=str)
    arg = parser.parse_args()
    main(arg)
