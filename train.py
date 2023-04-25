import os
import yaml
import argparse
import time
import torch
import numpy as np
from stable_baselines3 import PPO, DDPG, TD3
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.noise import NormalActionNoise

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


    # get configs
    job_timestamp = int(time.time())

    agent_config = config['agent_params']
    training_config = config['training_params']

    model_algo = agent_config['algo']
    domain_name = config['env_params']['domain_name']
    task_name = config['env_params']['task_name']

    # create log dir
    logdir = make_log_dir(config, job_timestamp)
    print(f"Using logdir: {logdir}")

    # create env object
    env = create_env(config)

    # wrap env with Monitor
    env = Monitor(env, logdir)

    # n_actions will be the dim of the output space of the network
    n_actions = env.action_space.shape[-1]
    fc_features_dim = agent_config['fc_features_dim']
    policy_kwargs = {}

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

    # create action noise
    action_noise = NormalActionNoise(mean=np.zeros(n_actions),
                                     sigma=0.1 * np.ones(n_actions))

    train_freq = (training_config['train_freq_num'],
                  training_config['train_freq_type'])



    do_resume_training = training_config['resume_training']
    load_model_ckpt = training_config['load_model_ckpt_path']

    if not do_resume_training:
        print("*" * 50)
        print("Training from scratch!")
        print("*" * 50)
        if model_algo == "PPO":
            # remove n_critics param
            policy_kwargs.pop('n_critics')
            model = PPO(agent_config['model_type'],
                        env,
                        verbose=1,
                        learning_starts=training_config['learning_starts'],                             
                        learning_rate=training_config['learning_rate'],
                        batch_size=training_config['batch_size'],
                        policy_kwargs=policy_kwargs,
                        device=device,
                        tensorboard_log=logdir)

        elif model_algo == "DDPG":
            model = DDPG(agent_config['model_type'],
                        env,
                        verbose=1,
                        buffer_size=training_config['buffer_size'], 
                        learning_starts=training_config['learning_starts'],
                        learning_rate=training_config['learning_rate'],
                        batch_size=training_config['batch_size'],
                        tau=agent_config['tau'],
                        policy_kwargs=policy_kwargs,
                        train_freq=train_freq,
                        device=device,
                        action_noise=action_noise,
                        tensorboard_log=logdir)
        
        elif model_algo == "TD3":
            model = TD3(agent_config['model_type'],
                        env,
                        verbose=1,
                        buffer_size=training_config['buffer_size'], 
                        learning_starts=training_config['learning_starts'],
                        learning_rate=training_config['learning_rate'],
                        batch_size=training_config['batch_size'],
                        tau=agent_config['tau'],
                        policy_kwargs=policy_kwargs,
                        train_freq=train_freq,
                        device=device,
                        action_noise=action_noise,
                        tensorboard_log=logdir)

        else:
            raise (ValueError, f"invalid model algo provided {model_algo}. Only PPO and DDPG are accepted")

    else:
        print("*" * 50)
        print(f"Resuming training from {load_model_ckpt}!")
        print("*" * 50)
        if model_algo == "PPO":
            policy_kwargs.pop('n_critics')
            model = PPO.load(load_model_ckpt,
                             env,
                             learning_starts=training_config['learning_starts'],
                             learning_rate=training_config['learning_rate'],
                             batch_size=training_config['batch_size'],
                             verbose=1,
                             policy_kwargs=policy_kwargs,
                             device=device,
                             tensorboard_log=logdir)
        elif model_algo == "DDPG":
            model = DDPG.load(load_model_ckpt,
                              env,
                              verbose=1,
                              buffer_size=training_config['buffer_size'], 
                              learning_starts=training_config['learning_starts'],
                              learning_rate=training_config['learning_rate'],
                              batch_size=training_config['batch_size'],
                              tau=agent_config['tau'],
                              policy_kwargs=policy_kwargs,
                              train_freq=train_freq,
                              device=device,
                              action_noise=action_noise,
                              tensorboard_log=logdir)
        elif model_algo == "TD3":
            model = TD3.load(load_model_ckpt,
                             env,
                             verbose=1,
                             buffer_size=training_config['buffer_size'], 
                             learning_starts=training_config['learning_starts'],
                             learning_rate=training_config['learning_rate'],
                             batch_size=training_config['batch_size'],
                             tau=agent_config['tau'],
                             policy_kwargs=policy_kwargs,
                             train_freq=train_freq,
                             device=device,
                             action_noise=action_noise,
                             tensorboard_log=logdir)
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
    print_training_info(config)

    savedir = config['output_params']['savedir']
    if not do_resume_training:
        callback.save_path = f"{callback.save_path}-{model_algo}-{domain_name}-{task_name}-{job_timestamp}"
    else:
        callback.save_path = load_model_ckpt

    print("*" * 50)
    print(f"Saving best model with name {callback.save_path}")
    print("*" * 50)

    # train the agent
    model.learn(
        total_timesteps=config['training_params']['total_training_steps'],
        tb_log_name=f"{model_algo}-{domain_name}-{task_name}-{job_timestamp}",
        callback=callback,
        progress_bar=True
    )

    savedir = os.path.join(savedir, domain_name, task_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

    savename = config['output_params']['savename']
    if savename is None:
        savename = f"{model_algo}-{domain_name}-{task_name}-{job_timestamp}"
    else:
        savename = f"{model_algo}-{savename}-{domain_name}-{task_name}-{job_timestamp}"

    model_savepath = os.path.join(savedir, savename)
    model.save(model_savepath)
