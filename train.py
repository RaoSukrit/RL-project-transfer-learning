import os
import yaml
import argparse
import time
import torch

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO, DDPG
from stable_baselines3.common import results_plotter
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from callbacks import *
from utils import * 

# set MUJOCO_GL environment var
os.environ['MUJOCO_GL'] = "osmesa"
os.environ.get('MUJOCO_GL', 'MUJOCO_GL not set')

# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

# print device
print('*' * 50)
print(f"using device: {device}")
print('*' * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str,
                        default='./configs/config.yaml',
                        help='Path to config file specifiying user params')

    args = parser.parse_args()

    with open(args.config, 'r') as fh:
        config = yaml.safe_load(fh)

    # create log dir
    logdir = make_log_dir(config)
    
    # create env object
    env = create_env(config)

    # wrap env with Monitor
    env = Monitor(env, logdir)

    # create custom feature extractor (ResNet-CNN) for agent training
    policy_kwargs = dict(
        features_extractor_class=get_extractor(config['agent_params']),
        features_extractor_kwargs=dict(
            fc_features_dim=config['network_params']['fc_features_dim'],
        ),
    )

    # create agent
    agent_config = config['agent_params']
    model_algo = agent_config['algo']
    if model_algo == "PPO":
        model = PPO(agent_config['model_type'],
                    env,
                    verbose=1,
                    policy_kwargs=policy_kwargs,
                    device=device)

    elif model_algo == "DDPG":
        model = DDPG(agent_config['model_type'],
                     env,
                     verbose=1,
                     batch_size=agent_config['batch_size'],
                     tau=agent_config['tau'],
                     policy_kwargs=policy_kwargs,
                     device=device)
    else:
        raise (ValueError, f"invalid model algo provided {model_algo}. Only PPO and DDPG are accepted")

    callback = save_best_model.SaveOnBestTrainingRewardCallback(
        check_freq=config['callback_params']['ckpt_freq'],
        log_dir=logdir,
        verbose=1
    )

    # reset the env
    env.reset()

    # print training info
    print_training_info()

    # train the agent
    model.learn(
        total_timesteps=agent_config['total_training_steps'],
        callback=callback,
        progress_bar=True
    )

    savedir = config['output_params']['savedir']
    domain_name = config['env_params']['domain_name']
    task_name = config['env_params']['task_name']

    savedir = os.path.join(savedir, domain_name, task_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

    savename = config['output_params']['savename']
    if savename is None:
        savename = f"{model_algo}-{domain_name}-{task_name}-{int(time.time())}"
    else:
        savename = f"{model_algo}-{savename}-{domain_name}-{task_name}-{int(time.time())}"

    model_savepath = os.path.join(savedir, savename)
    model.save(model_savepath)