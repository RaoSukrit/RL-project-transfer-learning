import os 
import dmc2gym
from extractors import *

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

def make_log_dir(cfg): 
    ''' Creates log directory '''
    logdir = os.path.abspath(cfg['output_params']['logdir'])
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    return logdir

def get_extractor(agent_cfg): 
    cnn_type = agent_cfg['cnn_model_type']
    cnns = {'resnet18': resnet18.CustomCNN, 
            'resnet34': resnet34.CustomCNN, 
            'dmcnn': dmcnn.CustomCNN}
    return cnns[cnn_type]

def print_training_info(cfg): 
    print('*' * 50)
    print("\nBeginning Training")

    print('*' * 50)
    print("\nEnv Params")
    for k, v in cfg['env_params'].items():
        print(f"\tParam:{k}; User Val: {v}")

    print('*' * 50)
    print("\nAgent Params")
    for k, v in cfg['agent_params'].items():
        print(f"\tParam:{k}; User Val: {v}")

    print('*' * 50)
    print("\nTraining Params")
    for k, v in cfg['output_params'].items():
        print(f"\tParam:{k}; User Val: {v}")
    print('*' * 50)