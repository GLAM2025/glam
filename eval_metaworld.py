import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
import copy
import colorama
import random
import json
import shutil
import pickle
import os

import yaml

import train_metaworld 
from utils import seed_np_torch, Logger, load_config
from replay_buffer import ReplayBuffer
import env_wrapper
import agents
from sub_models.functions_losses import symexp



def eval_episodes(num_episode, suite, env_name, seed, 
                  world_model, agent: agents.ActorCriticAgent, 
                  num_envs=1):
    world_model.eval()
    agent.eval()
    vec_env, env_config, _ = train_metaworld.make_env(suite, env_name, seed=seed)

    # print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)
    print("Current env: " + f"{env_name}")
    sum_reward = np.zeros(num_envs)
    current_obs = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    final_rewards = []
    final_dones = []
    while True:
        # sample part >>>
        with torch.no_grad():
            if len(context_action) == 0:
                action = vec_env.action_space.sample()
            else:
                context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                model_context_action = np.stack(list(context_action), axis=0)
                model_context_action = torch.Tensor(model_context_action).cuda().unsqueeze(0)
                prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                action = agent.sample_as_env_action(
                    torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                    greedy=False
                )

            if suite == 'atari':
                context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
            elif suite == 'metaworld':
                context_obs.append(rearrange(torch.Tensor(current_obs.copy()).cuda(), "H W C -> 1 1 C H W")/255)
        context_action.append(action)

        obs, reward, done, truncated, info = vec_env.step(action)

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            sum_reward += reward
            for i in range(num_envs):
                if done_flag:
                    final_rewards.append(sum_reward)
                    final_dones.append(np.float16(done))
                    current_obs = vec_env.reset()
                    sum_reward = np.zeros(num_envs)
                    
                    if len(final_rewards) == num_episode:
                        success = np.mean(final_dones)
                        print("Mean reward: " + f"{np.mean(final_rewards)}" +
                              " | Success rate: " + f"{success}")
                        return np.mean(final_rewards)

        # update current_obs, current_info and sum_reward
        else:
            sum_reward += reward
            current_obs = obs.copy()
            current_info = info
        # <<< sample part

class Config:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


if __name__ == "__main__":
    # ignore warnings
    import warnings
    warnings.filterwarnings('ignore')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # parse arguments
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument("-suite", type=str, default='metaworld', required=False)
    parser.add_argument("-env_name", type=str, default='door-close', required=False)
    parser.add_argument("-seed", type=int, default=1, required=False)
    parser.add_argument("-base_model", type=str, default='Mamba', required=False)
    parser.add_argument("-version", type=str, default='1_2_1', required=False)
    parser.add_argument("-config_path", type=str, default='config_files/Mamba.yaml', required=False)

    parser.add_argument("-cuda_device", type=int, default=0, required=False)
    parser.add_argument("-sample", type=str, default='normal', required=False)

    parser.add_argument("-data_path", type=str, default='/home/hq/LSTW/GLAM/data', required=False)
    args = parser.parse_args()
    
    with open(args.config_path, 'r') as file:
        conf = yaml.safe_load(file)
        conf = Config(conf)
    # print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    print(str(args))

    # set seed
    seed_np_torch(seed=args.seed)

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    # build and load model/agent
    dummy_env, env_conf, action_dim = train_metaworld.make_env(args.suite ,args.env_name, seed=0)

    world_model = train_metaworld.build_world_model(conf, env_conf, args, action_dim, device)
    agent = train_metaworld.build_agent(conf, action_dim, device)
    root_path = f'{args.data_path}/ckpt/{args.env_name}_seed{args.seed}_{args.base_model}_{args.version}'

    import glob 
    pathes = glob.glob(f"{root_path}/world_model_*.pth")
    steps = [int(path.split("_")[-1].split(".")[0]) for path in pathes]
    steps.sort()
    steps = steps[:]
    print(steps)
    results = []
    for step in tqdm(steps):
        world_model.load_state_dict(torch.load(f"{root_path}/world_model_{step}.pth", map_location=torch.device('cuda:0')))
        agent.load_state_dict(torch.load(f"{root_path}/agent_{step}.pth", map_location=torch.device('cuda:0')))
        # # eval
        episode_returns = []
        for i in range(5):
            seed = np.random.randint(0, 100)
            episode_return = eval_episodes(
                num_episode=2,
                suite=args.suite,
                env_name=args.env_name,
                seed=seed,
                world_model=world_model,
                agent=agent,
                num_envs=1
            )
            episode_returns.append(episode_return)

        episode_avg_return = np.mean(episode_returns)
        results.append([step, episode_avg_return])
    os.makedirs(f"eval_result/{args.suite}", exist_ok=True)
    with open(f"eval_result/{args.suite}/{args.env_name}_{args.version}.csv", mode="w") as fout:
        fout.write("step, episode_avg_return\n")
        for step, episode_avg_return in results:
            fout.write(f"{step},{episode_avg_return}\n")

        print(f'save to eval_result/{args.suite}/{args.env_name}_{args.version}.csv')
