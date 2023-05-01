import argparse
import yaml
import numpy as np
from stable_baselines3.common.vec_env import VecVideoRecorder

from utils import * 

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

    # create video dir
    video_folder = cfg['video_folder']
    print(f"Using videodir: {video_folder}")

    # create env object
    env = create_env(cfg)

    # load model 
    load_model_ckpt = cfg['load_model_ckpt_path']
    model = load_model(env, cfg, video_folder, load_model_ckpt)

    # Wrap env in VecVideoRecorder
    env = VecVideoRecorder(
        env,
        video_folder,
        record_video_trigger=lambda x: x == 0,
        video_length=cfg['video_length'],
        name_prefix=cfg['name_prefix'],
    )

    # reset environment
    obs = env.reset()

    # run and record model 
    lstm_states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    try:
        for _ in range(cfg['video_length'] + 1):
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_starts,
                deterministic=True,
            )
            obs, _, dones, _ = env.step(action)
            episode_starts = dones
    except KeyboardInterrupt:
        pass

    # close environment
    env.close()