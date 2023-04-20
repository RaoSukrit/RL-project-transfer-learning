import os

# os.environ['MUJOCO_GL'] = 'glfw'
os.environ['MUJOCO_GL'] = "osmesa"
os.environ.get('MUJOCO_GL', 'MUJOCO_GL not set')

import yaml
import argparse

import gym
import dmc2gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DDPG
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import torch
import torch.nn as nn
from gym import spaces
from torchvision.models import resnet18

import models
import training_callbacks as tc


# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

print('*' * 50)
print(f"using device: {device}")
print('*' * 50)


def create_env(env_config):
    '''Create Gym Env'''
    env = dmc2gym.make(**env_config)
    return env


def test_env(env):
    '''Tests pixel observations'''
    try:
        obs = env.reset()
        for i in range(10):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(obs.shape, reward)
        env.reset()
    except Exception as err:
        raise err


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        default='.config.yaml'
                        help='Path to config file specifiying user params')

    args = parser.parse_args()

    with open(args.config, 'r') as fh:
        config = yaml.safe_load(fh)

    # create log dir
    logdir = os.path.abspath(config['output_params']['logdir'])

    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)

    # create env object
    env_config = config['env_params']
    env = create_env(env_config)

    # wrap env with Monitor
    env = Monitor(env, logdir)

    # create custom feature extractor (ResNet-CNN) for agent training
    policy_kwargs = dict(
                            features_extractor_class=models.CustomCNN,
                            features_extractor_kwargs=config['network_params'],
                        )

    # create PPO model
    callback_config = config['callback_params']
    agent_config = config['agent_params']
    model = PPO(agent_config['model_type'],
                env,
                verbose=1,
                policy_kwargs=policy_kwargs,
                device=device)

    callback = tc.SaveOnBestTrainingRewardCallback(check_freq=callback_config['ckpt_freq'],
                                                   log_dir=logdir,
                                                   verbose=1)

    # reset the env
    env.reset()

    print('*' * 50)
    print("\nBeginning Training")

    print('*' * 50)
    print("\nEnv Params")
    for k, v in env_config.items():
        print(f"\tParam:{k}; User Val: {v}")

    print('*' * 50)
    print("\nAgent Params")
    for k, v in agent_config.items():
        print(f"\tParam:{k}; User Val: {v}")

    print('*' * 50)
    print("\nTraining Params")
    for k, v in config['output_params'].items():
        print(f"\tParam:{k}; User Val: {v}")
    print('*' * 50)

    # train the agent
    model.learn(total_timesteps=agent_config['total_training_steps'],
                callback=callback,
                progress_bar=True)

    model_savepath = os.path.join(config['output_params']['savedir'],
                                  config['output_params']['savename'])
    model.save(model_savepath)