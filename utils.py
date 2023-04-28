import os
import re

import torch
import dmc2gym
import numpy as np
from collections import OrderedDict

from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.noise import NormalActionNoise

from extractors import *
from callbacks import *


# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')


# create environment
def create_env(cfg):
    '''Create Gym Env'''
    env = dmc2gym.make(**cfg['env_params'])
    return env


def test_env(env, num_steps=10):
    '''Tests pixel observations'''
    try:
        obs = env.reset()
        for i in range(num_steps):
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(obs.shape, reward)
        env.reset()
    except Exception as err:
        raise err


def make_log_dir(config, job_timestamp):
    ''' Creates log directory '''
    model_algo = config['agent_params']['algo']
    cnn_model_type = config['agent_params']['cnn_model_type']
    domain_name = config['env_params']['domain_name']
    task_name = config['env_params']['task_name']

    logdir = os.path.abspath(os.path.join(config['output_params']['logdir'],
                                          model_algo,
                                          cnn_model_type,
                                          domain_name,
                                          task_name,
                                          str(job_timestamp)))
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    return logdir


def get_extractor(agent_cfg, **kwargs):
    cnn_type = agent_cfg['cnn_model_type']
    cnns = {'resnet18': resnet18.CustomCNN,
            'resnet34': resnet34.CustomCNN,
            'dmcnn': dmcnn.CustomCNN,
            'dmcnn2': dmcnn2.CustomCNN}
    return cnns[cnn_type]


def print_training_info(cfg):
    print('*' * 50)
    print("\nBeginning Training")

    for param_type, param_dict in cfg.items():
        if isinstance(param_dict, dict):
            print('*' * 50)
            print(f"{param_type} Params\n")
            print('*' * 50)

            for param_name, param_val in param_dict.items():
                print('*' * 50)
                print(f"\tParam: {param_name} User Val: {param_val}\n")
                print('*' * 50)

    else:
        print('*' * 50)
        print(f"\tParam: {param_type} User Val: {param_dict}\n")
        print('*' * 50)


def define_policy_kwargs(env, config):
    policy_kwargs = {}
    agent_config = config['agent_params']

    # n_actions will be the dim of the output space of the network
    n_actions = env.action_space.shape[-1]
    fc_features_dim = agent_config['fc_features_dim']

    # create custom feature extractor if required
    if agent_config['cnn_model_type']:
        policy_kwargs = dict(
            features_extractor_class=get_extractor(agent_config),
            features_extractor_kwargs=dict(
                fc_features_dim=fc_features_dim,
            ),
            net_arch=[int(fc_features_dim / 2), n_actions],
            n_critics=1,
        )
    print(policy_kwargs)

    return policy_kwargs


def define_model_from_scratch(env, config, logdir):

    # get the configs
    agent_config = config['agent_params']
    training_config = config['training_params']

    model_algo = config['agent_params']['algo']
    domain_name = config['env_params']['domain_name']
    task_name = config['env_params']['task_name']

    # create action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                     sigma=0.1 * np.ones(n_actions))

    train_freq = (training_config['train_freq_num'],
                  training_config['train_freq_type'])

    policy_kwargs = define_policy_kwargs(env, config)

    if model_algo == "PPO":
        # remove n_critics param
        policy_kwargs.pop('n_critics')
        if training_config['use_sde']:
            action_noise = None
            use_sde = training_config['use_sde']

        model = PPO(agent_config['model_type'],
                    env,
                    verbose=1,
                    learning_rate=training_config['learning_rate'],
                    batch_size=training_config['batch_size'],
                    policy_kwargs=policy_kwargs,
                    use_sde=use_sde,
                    device=device,
                    tensorboard_log=logdir)

    elif model_algo == "DDPG":
        model = DDPG(agent_config['model_type'],
                    env,
                    verbose=1,
                    learning_rate=training_config['learning_rate'],
                    batch_size=training_config['batch_size'],
                    tau=agent_config['tau'],
                    policy_kwargs=policy_kwargs,
                    train_freq=train_freq,
                    device=device,
                    action_noise=action_noise,
                    tensorboard_log=logdir)

    elif model_algo == "SAC":
        if training_config['use_sde']:
            action_noise = None
            use_sde = training_config['use_sde']

        model = SAC(agent_config['model_type'],
                    env,
                    verbose=1,
                    learning_rate=training_config['learning_rate'],
                    batch_size=training_config['batch_size'],
                    tau=agent_config['tau'],
                    train_freq=train_freq,
                    action_noise=action_noise,
                    use_sde=use_sde,
                    policy_kwargs=policy_kwargs,
                    device=device,
                    tensorboard_log=logdir)
    else:
        raise (ValueError, f"invalid model algo provided {model_algo}. Only PPO, SAC, TD3 and DDPG are accepted")

    return model


def reload_model_from_ckpt(env, config,
                           load_model_ckpt, logdir):
    # get the configs
    agent_config = config['agent_params']
    training_config = config['training_params']

    model_algo = config['agent_params']['algo']
    domain_name = config['env_params']['domain_name']
    task_name = config['env_params']['task_name']

    # create action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                     sigma=0.1 * np.ones(n_actions))

    train_freq = (training_config['train_freq_num'],
                  training_config['train_freq_type'])

    policy_kwargs = define_policy_kwargs(env, config)

    if model_algo == "PPO":
        policy_kwargs.pop('n_critics')
        if training_config['use_sde']:
            action_noise = None
            use_sde = training_config['use_sde']

        model = PPO.load(load_model_ckpt,
                            env,
                            learning_rate=training_config['learning_rate'],
                            batch_size=training_config['batch_size'],
                            verbose=1,
                            use_sde=use_sde,
                            policy_kwargs=policy_kwargs,
                            device=device,
                            tensorboard_log=logdir)

    elif model_algo == "DDPG":
        model = DDPG.load(load_model_ckpt,
                            env,
                            verbose=1,
                            learning_rate=training_config['learning_rate'],
                            batch_size=training_config['batch_size'],
                            tau=agent_config['tau'],
                            policy_kwargs=policy_kwargs,
                            train_freq=train_freq,
                            device=device,
                            action_noise=action_noise,
                            tensorboard_log=logdir)

    elif model_algo == "SAC":
        if training_config['use_sde']:
            action_noise = None
            use_sde = training_config['use_sde']

        model = SAC.load(load_model_ckpt,
                    env,
                    verbose=1,
                    learning_rate=training_config['learning_rate'],
                    batch_size=training_config['batch_size'],
                    tau=agent_config['tau'],
                    train_freq=train_freq,
                    action_noise=action_noise,
                    use_sde=use_sde,
                    policy_kwargs=policy_kwargs,
                    device=device,
                    tensorboard_log=logdir)
    else:
        raise (ValueError, f"invalid model algo provided {model_algo}. Only PPO, SAC, TD3 and DDPG are accepted")

    return model


def reload_model_from_ckpt_transfer_learn(pretrain_model,
                                          new_env_model,
                                          debug=False):

    keywords = ['features_extractor']
    # keywords = ['features_extractor', 'log_std']

    pretrain_env_state_dict = pretrain_model.get_parameters()['policy']
    new_env_state_dict = new_env_model.get_parameters()['policy']

    if debug:
        print("*" * 50)
        print("New Env Model State Dict")
        # print(type(new_env_model))
        print(new_env_model.get_parameters()['policy'].keys())

    pretrain_feat_ext_state_dict = OrderedDict()
    pretrain_feat_ext_state_dict['policy'] = OrderedDict()
    pretrain_feat_ext_state_dict['policy.optimizer'] = \
        new_env_model.get_parameters()['policy.optimizer']

    for key, weight in pretrain_env_state_dict.items():
        for keyword in keywords:
            if key.startswith(keyword):
                # new_key = key.split('.', 1)[-1]
                new_key = f"{key}"
                # reload_model_state_dict[new_key] = weight
                pretrain_feat_ext_state_dict['policy'][new_key] = weight
            else:
                new_key = f"{key}"
                pretrain_feat_ext_state_dict['policy'][new_key] = \
                    new_env_state_dict[new_key]

    if debug:
        print("*" * 50)
        print("Original State Dict From Checkpoint")
        print(pretrain_env_state_dict.keys())

        print("*" * 50)
        print("Modified State Dict To Reload Feature Extractor Weights")
        print(pretrain_feat_ext_state_dict.keys())
        for k, v in pretrain_feat_ext_state_dict.items():
            print(f"key={k}")
        print("*" * 50)

    print(new_env_model.get_parameters().keys())
    pretrain_model.set_parameters(pretrain_feat_ext_state_dict)

    return pretrain_model
