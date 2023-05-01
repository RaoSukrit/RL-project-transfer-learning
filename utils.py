import os
import time
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

'''
Create gym environment with specified config
@param cfg - specified config
@return gym environment
'''
def create_env(cfg):
    env = dmc2gym.make(**cfg['env_params'])
    return env

'''
Test gym environment for given number of steps
Print out observation shapes and rewards
@param env - gym environment
@param num_steps - number of steps to run
'''
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

'''
Make log directory for specified config
@param cfg - specified config 
@return path to log directory
'''
def make_log_dir(cfg):
    # get timestamp and values from config
    job_timestamp = int(time.time())
    model_algo = cfg['agent_params']['algo']
    cnn_model_type = cfg['agent_params']['cnn_model_type']
    domain_name = cfg['env_params']['domain_name']
    task_name = cfg['env_params']['task_name']

    # create log directory 
    logdir = os.path.abspath(os.path.join(cfg['output_params']['logdir'],
                                          model_algo,
                                          cnn_model_type,
                                          domain_name,
                                          task_name,
                                          str(job_timestamp)))
    # make and return logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    return logdir

'''
Get class of extractor network from specified config
@param agent_cfg - agent config 
@return class of specified extractor
'''
def get_extractor(agent_cfg):
    cnn_type = agent_cfg['cnn_model_type']
    cnns = {'resnet18': resnet18.CustomCNN,
            'resnet34': resnet34.CustomCNN,
            'dmcnn': dmcnn.CustomCNN,
            'dmcnn2': dmcnn2.CustomCNN}
    return cnns[cnn_type]

'''
Creates policy keyword arguments to pass to extractor
from the specified env and config
@param env - gym environment
@param cfg - specified config
@return policy keyword arguments created using config
'''
def define_policy_kwargs(env, cfg):
    policy_kwargs = {}
    agent_cfg = cfg['agent_params']

    # n_actions will be the dim of the output space of the network
    n_actions = env.action_space.shape[-1]
    fc_features_dim = agent_cfg['fc_features_dim']

    # create custom feature extractor if required
    if agent_cfg['cnn_model_type']:
        policy_kwargs = dict(
            features_extractor_class=get_extractor(agent_cfg),
            features_extractor_kwargs=dict(
                fc_features_dim=fc_features_dim,
            ),
            net_arch=[int(fc_features_dim / 2), n_actions],
            n_critics=1,
        )
    print(policy_kwargs)
    return policy_kwargs

'''
Creates a stable baselines 3 model for the specified
env, config, and log directory. If checkpoint file is 
specified, reloads model from there
@param env - gym environment
@param cfg - specified config
@param logdir - given log directory
@param load_model_ckpt - model checkpoint file
@return stable baselines 3 model 
'''
def load_model(env, cfg, logdir, load_model_cpkt=None):
    # get the configs and model algorithm
    agent_cfg = cfg['agent_params']
    training_cfg = cfg['training_params']
    model_algo = cfg['agent_params']['algo']

    # create action noise
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                     sigma=0.1 * np.ones(n_actions))

    # get training frequency
    train_freq = (training_cfg['train_freq_num'],
                  training_cfg['train_freq_type'])

    # create policy kwargs
    policy_kwargs = define_policy_kwargs(env, cfg)

    # load model if specified, otherwise set model type
    model_type = load_model_cpkt if load_model_cpkt else agent_cfg['model_type']

    if model_algo == "PPO":
        # remove n_critics param
        policy_kwargs.pop('n_critics')
        
        # Use SDE action noise
        if training_cfg['use_sde']:
            action_noise = None
            use_sde = training_cfg['use_sde']

        model = PPO(model_type,
                    env,
                    verbose=1,
                    learning_rate=training_cfg['learning_rate'],
                    batch_size=training_cfg['batch_size'],
                    policy_kwargs=policy_kwargs,
                    use_sde=use_sde,
                    device=device,
                    tensorboard_log=logdir)

    elif model_algo == "DDPG":
        model = DDPG(model_type,
                    env,
                    verbose=1,
                    learning_rate=training_cfg['learning_rate'],
                    batch_size=training_cfg['batch_size'],
                    tau=agent_cfg['tau'],
                    policy_kwargs=policy_kwargs,
                    train_freq=train_freq,
                    device=device,
                    action_noise=action_noise,
                    tensorboard_log=logdir)

    elif model_algo == "SAC":
        # Use SDE action noise
        if training_cfg['use_sde']:
            action_noise = None
            use_sde = training_cfg['use_sde']

        model = SAC(model_type,
                    env,
                    verbose=1,
                    learning_rate=training_cfg['learning_rate'],
                    batch_size=training_cfg['batch_size'],
                    tau=agent_cfg['tau'],
                    train_freq=train_freq,
                    action_noise=action_noise,
                    use_sde=use_sde,
                    policy_kwargs=policy_kwargs,
                    device=device,
                    tensorboard_log=logdir)
    else:
        # invalid model
        raise (ValueError, f"invalid model algo provided {model_algo}. Only PPO, SAC, TD3 and DDPG are accepted")

    return model

'''
Given a pretrained model and a new    
'''
def reload_model_from_ckpt_transfer_learn(pretrain_model,
                                          new_env_model,
                                          debug=False):

    keywords = ['features_extractor']
    pretrain_env_state_dict = pretrain_model.get_parameters()['policy']
    new_env_state_dict = new_env_model.get_parameters()['policy']

    if debug:
        print("*" * 50)
        print("New Env Model State Dict")
        print(new_env_model.get_parameters()['policy'].keys())

    pretrain_feat_ext_state_dict = OrderedDict()
    pretrain_feat_ext_state_dict['policy'] = OrderedDict()
    pretrain_feat_ext_state_dict['policy.optimizer'] = \
        new_env_model.get_parameters()['policy.optimizer']

    for key, weight in pretrain_env_state_dict.items():
        for keyword in keywords:
            if key.startswith(keyword):
                new_key = f"{key}"
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

'''
Prints out important info from specified config
@param cfg - specified config
'''
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