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

    if model_algo == "PPO":
        # remove n_critics param
        policy_kwargs.pop('n_critics')
        
        # Use SDE action noise
        use_sde = training_cfg['use_sde']
        if use_sde:
            action_noise = None
            
        if not load_model_cpkt: 
            model = PPO(agent_cfg['model_type'],
                    env,
                    verbose=1,
                    learning_rate=training_cfg['learning_rate'],
                    batch_size=training_cfg['batch_size'],
                    policy_kwargs=policy_kwargs,
                    use_sde=use_sde,
                    device=device,
                    tensorboard_log=logdir)
        else: 
            model = PPO.load(load_model_cpkt,
                    env,
                    verbose=1,
                    learning_rate=training_cfg['learning_rate'],
                    batch_size=training_cfg['batch_size'],
                    policy_kwargs=policy_kwargs,
                    use_sde=use_sde,
                    device=device,
                    tensorboard_log=logdir)

    elif model_algo == "DDPG":
        if not load_model_cpkt: 
            model = DDPG(agent_cfg['model_type'],
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
        else: 
            model = DDPG.load(load_model_cpkt,
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
        use_sde = training_cfg['use_sde']
        if use_sde:
            action_noise = None

        if not load_model_cpkt: 
            model = SAC(agent_cfg['model_type'],
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
            model = SAC.load(load_model_cpkt,
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
Copy weights (in-place) with specified keywords from 
from_model to to_model     
@param from_model - model to copy weights from
@param to_model - model to copy weights to 
@param wt_keys - weights keywords that should be copied  
@return to_model with specified weights copied  
'''
def transfer_weights(from_model, 
                     to_model, 
                     wt_keys = ['features_extractor']):

    # get state dicts 
    from_model_state_dict = from_model.get_parameters()['policy']
    to_model_state_dict = to_model.get_parameters()

    # iterate over from_model state_dict
    for key, weight in from_model_state_dict.items():
        for keyword in wt_keys:
            # copy weight if it is a keyword weight
            if key.startswith(keyword):
                to_model_state_dict['policy'][key] = weight

    # run test to see if weights were updated succesfully
    for k, weight_dict in to_model_state_dict.items():
        if k == 'policy':
            for weight_key, updated_weight in weight_dict.items():
                for keyword in wt_keys:
                    if weight_key.startswith(keyword):
                        assert torch.equal(updated_weight,
                                           from_model_state_dict[weight_key]), f"Weights updated incorrectly. Mismatch for {weight_key}"

    to_model.set_parameters(to_model_state_dict)
    return to_model

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