import os
import yaml
import argparse
import torch
import time
from stable_baselines3 import PPO, DDPG, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan
from stable_baselines3.common.noise import NormalActionNoise

from callbacks import *
from utils import *

# Set MUJOCO_GL environment var
os.environ['MUJOCO_GL'] = "osmesa"
os.environ.get('MUJOCO_GL', 'MUJOCO_GL not set')

# Set device to GPU if available
device = torch.device('cuda') if torch.cuda.is_available() \
    else torch.device('cpu')

# Print device
print('*' * 50)
print(f"using device: {device}")
print('*' * 50)

if __name__ == "__main__":
    # add config argument  
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        default='./configs/config.yaml',
                        help='Path to config file specifiying user params')

    # get config and read 
    args = parser.parse_args()
    with open(args.config, 'r') as fh:
        cfg = yaml.safe_load(fh)

    # create log dir
    logdir = make_log_dir(cfg)
    print(f"Using logdir: {logdir}")

    # create env object
    env = create_env(cfg)

    # wrap env with Monitor
    env = Monitor(env, logdir)

    # read variables from training config
    training_cfg = cfg['training_params']
    do_resume_training = training_cfg['resume_training']
    load_model_ckpt = training_cfg['load_model_ckpt_path']
    is_transfer_learning = training_cfg['is_transfer_learning']

    # transfer learning with pre-trained model
    if do_resume_training and is_transfer_learning:
        print("*" * 50)
        print("PERFORMING TRANSFER LEARNING!\nCREATING MODEL FROM SCRATCH FOR NEW TASK")
        print("*" * 50)

        # create an env with the original domain and task on which the saved model
        # was trained. For this we need to modify the config and then
        # set the params for the feature extractor alone
        pretrain_params = cfg['transfer_learning']

        pretrain_config = cfg['env_params']
        pretrain_config['domain_name'] = \
            pretrain_params['original_task']['domain_name']
        pretrain_config['task_name'] = \
            pretrain_params['original_task']['task_name']

        print(f"pretrain_config={pretrain_config}")

        # create pretrained env, used to load the pretrained model
        pretrain_env = create_env({"env_params": pretrain_config})

        # create model using checkpoint of pretrained model
        pretrain_model = load_model(pretrain_env, cfg, logdir, load_model_ckpt)

        # create model using new environment. Used as base to reload pretrained params
        new_env_model = load_model(env, cfg, logdir)

        # create model object that loads the pretained params for
        # the feature extractor
        model = transfer_weights(from_model=pretrain_model, to_model=new_env_model)

    # pre-trained model without transfer learning
    elif do_resume_training and not is_transfer_learning:
        print("*" * 50)
        print(f"NOT TRANSFER LEARNING! Resuming training from {load_model_ckpt}!")
        print("*" * 50)
        model = load_model(env, cfg, logdir, load_model_ckpt)

    # training model from scratch 
    else:
        print("*" * 50)
        print("Training from scratch!")
        print("*" * 50)
        model = load_model(env, cfg, logdir)

    # create callback function to save best model
    callback = save_best_model.SaveOnBestTrainingRewardCallback(
        check_freq=cfg['callback_params']['ckpt_freq'],
        log_dir=logdir,
        verbose=1
    )

    # reset the env
    env.reset()

    # print training info
    print_training_info(cfg)

    # more params
    savedir = cfg['output_params']['savedir']
    model_algo = cfg['agent_params']['algo']
    domain_name = cfg['env_params']['domain_name']
    task_name = cfg['env_params']['task_name']
    job_timestamp = int(time.time())

    # set checkpoint path 
    if not do_resume_training:
        callback.save_path = f"{callback.save_path}-{model_algo}-{domain_name}-{task_name}-{job_timestamp}"
    else:
        callback.save_path = load_model_ckpt

    # print save name
    print("*" * 50)
    print(f"Saving best model with name {callback.save_path}")
    print("*" * 50)

    # train the agent
    model.learn(
        total_timesteps=cfg['training_params']['total_training_steps'],
        tb_log_name=f"{model_algo}-{domain_name}-{task_name}-{job_timestamp}",
        callback=callback,
        progress_bar=True
    )

    # create save model directory
    savedir = os.path.join(savedir, domain_name, task_name)
    if not os.path.exists(savedir):
        os.makedirs(savedir, exist_ok=True)

    # set save model name
    savename = cfg['output_params']['savename']
    if not do_resume_training:
        if savename is None:
            savename = f"{model_algo}-{domain_name}-{task_name}-{job_timestamp}"
        else:
            savename = f"{model_algo}-{savename}-{domain_name}-{task_name}-{job_timestamp}"
    else:
        savename = load_model_ckpt

    # save trained model 
    model_savepath = os.path.join(savedir, savename)
    model.save(model_savepath)
