import gymnasium
import argparse
from tensorboardX import SummaryWriter
import cv2
import numpy as np
from scipy.special import logit
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
import wrappers

from utils import seed_np_torch, Logger, load_config
import env_wrapper
import agents
from sub_models.functions_losses import symexp
import yaml


def load_env_config(suite):
    """
    Load the environment configuration based on the suite name.
    """
    config_path = "/home/hq/LSTW/GLAM/config_files/envs.yaml"
    with open(config_path, 'r') as file:
        env_configs = yaml.safe_load(file)
    
    if suite not in env_configs:
        raise ValueError(f"Suite '{suite}' not found in {config_path}")
    
    # Return the configuration dictionary for the specified suite
    return Config(env_configs[suite])

def build_single_env(env_name, seed, config):
    env = gymnasium.make(env_name, full_action_space=False, render_mode = config.render_mode, frameskip=config.frameskip)
    env = env_wrapper.SeedEnvWrapper(env, seed=seed)
    env = env_wrapper.MaxLast2FrameSkipWrapper(env, skip=config.action_repeat)
    env = gymnasium.wrappers.ResizeObservation(env, shape=config.image_size)
    env = env_wrapper.LifeLossInfo(env)
    return env


def build_vec_env(env_name, seed, config):
    # lambda pitfall refs to: https://python.plainenglish.io/python-pitfalls-with-variable-capture-dcfc113f39b7
    def lambda_generator(env_name, seed, config):
        return lambda: build_single_env(env_name, seed, config)
    env_fns = []
    env_fns = [lambda_generator(env_name, seed, config) for i in range(config.num_envs)]
    vec_env = gymnasium.vector.AsyncVectorEnv(env_fns=env_fns)   
    return vec_env

def make_env(suite ,env_name, seed):
    config = load_env_config(suite) # TODO
    if suite == 'atari':
        env_name = 'ALE/' + env_name + '-v5'
        env = build_vec_env(env_name, seed, config)
        action_dim = env.single_action_space.n
    elif suite == 'metaworld':
        
        env = wrappers.MetaWorld(
            env_name,
            seed,
            config.action_repeat,
            config.image_size,
            config.camera,
            config.device
        )
        env = wrappers.NormalizeActions(env)
        env = wrappers.TimeLimit(env, config.time_limit)
        env = wrappers.SelectAction(env, key='action')
    else:
        raise NotImplementedError(f"Env not implemented")
    
    return env, config, action_dim
    

def train_world_model_step(replay_buffer, world_model, batch_size, demonstration_batch_size, batch_length, logger):
    obs, action, reward, termination = replay_buffer.sample(batch_size, demonstration_batch_size, batch_length)
    world_model.update(obs, action, reward, termination, logger=logger)


@torch.no_grad()
def world_model_imagine_data(replay_buffer,
                             world_model, agent: agents.ActorCriticAgent,
                             imagine_batch_size, imagine_demonstration_batch_size,
                             imagine_context_length, imagine_batch_length,
                             log_video, logger):
    '''
    Sample context from replay buffer, then imagine data with world model and agent
    '''
    world_model.eval()
    agent.eval()

    sample_obs, sample_action, sample_reward, sample_termination = replay_buffer.sample(
        imagine_batch_size, imagine_demonstration_batch_size, imagine_context_length)
    latent, action, reward_hat, termination_hat = world_model.imagine_data(
        agent, sample_obs, sample_action,
        imagine_batch_size=imagine_batch_size+imagine_demonstration_batch_size,
        imagine_batch_length=imagine_batch_length,
        log_video=log_video,
        logger=logger
    )
    return latent, action, None, None, reward_hat, termination_hat


def joint_train_world_model_agent(suite, env_name, max_steps, 
                                  replay_buffer,
                                  world_model, agent: agents.ActorCriticAgent,
                                  train_dynamics_every_steps, train_agent_every_steps,
                                  batch_size, demonstration_batch_size, batch_length,
                                  imagine_batch_size, imagine_demonstration_batch_size,
                                  imagine_context_length, imagine_batch_length,
                                  save_every_steps, seed, logger, incrementalimagine, 
                                  ckpt_path):
    


    # vec_env = build_vec_env(env_name, image_size, num_envs=num_envs, seed=seed)
    vec_env, env_config, _ = make_env(suite, env_name, seed=seed)
    num_envs = env_config.num_envs
    print("Current env: " + colorama.Fore.YELLOW + f"{env_name}" + colorama.Style.RESET_ALL)

    # reset envs and variables
    sum_reward = np.zeros(num_envs)
    current_obs, current_info = vec_env.reset()
    context_obs = deque(maxlen=16)
    context_action = deque(maxlen=16)

    # sample and train
    for total_steps in tqdm(range(max_steps//num_envs)):
        # sample part >>>
        if replay_buffer.ready():
            world_model.eval()
            agent.eval()
            with torch.no_grad():
                if len(context_action) == 0:
                    action = vec_env.action_space.sample()

                else:
                    context_latent = world_model.encode_obs(torch.cat(list(context_obs), dim=1))
                    model_context_action = np.stack(list(context_action), axis=1)
                    model_context_action = torch.Tensor(model_context_action).cuda()

                    prior_flattened_sample, last_dist_feat = world_model.calc_last_dist_feat(context_latent, model_context_action)
                    action= agent.sample_as_env_action(
                        torch.cat([prior_flattened_sample, last_dist_feat], dim=-1),
                        greedy=False
                    )
            if suite == 'atari':
                context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B H W C -> B 1 C H W")/255)
            elif suite == 'metaworld':
                context_obs.append(rearrange(torch.Tensor(current_obs).cuda(), "B C H W -> B 1 C H W")/255)
            context_action.append(action)

        else:
            action = vec_env.action_space.sample()

        obs, reward, done, truncated, info = vec_env.step(action)
        replay_buffer.append(current_obs, action, reward, np.logical_or(done, info["life_loss"]))

        done_flag = np.logical_or(done, truncated)
        if done_flag.any():
            for i in range(num_envs):
                if done_flag[i]:
                    logger.log(f"sample/{env_name}_reward", sum_reward[i])
                    logger.log(f"sample/{env_name}_episode_steps", current_info["episode_frame_number"][i]//4)  # framskip=4
                    logger.log("replay_buffer/length", len(replay_buffer))
                    sum_reward[i] = 0
                    
                    if total_steps > 60000:
                        print(colorama.Fore.GREEN + f"Episode end!!! Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
                        torch.save(world_model.state_dict(), ckpt_path+f"/world_model_{total_steps}.pth")
                        torch.save(agent.state_dict(), ckpt_path+f"/agent_{total_steps}.pth")     

        # update current_obs, current_info and sum_reward
        sum_reward += reward
        current_obs = obs
        current_info = info
        # <<< sample part

        # train world model part >>>
        if replay_buffer.ready() and total_steps % (train_dynamics_every_steps//num_envs) == 0:
            train_world_model_step(
                replay_buffer=replay_buffer,
                world_model=world_model,
                batch_size=batch_size,
                demonstration_batch_size=demonstration_batch_size,
                batch_length=batch_length,
                logger=logger
            )
        # <<< train world model part

        # train agent part >>>
        if replay_buffer.ready() and total_steps % (train_agent_every_steps//num_envs) == 0 and total_steps*num_envs >= 0:
            if total_steps % (save_every_steps//num_envs) == 0:
                log_video = True
            else:
                log_video = False
           
            if incrementalimagine.Flag: 
                imagine_batch_length = incrementalimagine.InitialSteps + min(int((total_steps/incrementalimagine.IncrementFrequency)*incrementalimagine.IncrementalSteps), incrementalimagine.MaxSteps)
            imagine_latent, agent_action, agent_logprob, agent_value, imagine_reward, imagine_termination = world_model_imagine_data(
                replay_buffer=replay_buffer,
                world_model=world_model,
                agent=agent,
                imagine_batch_size=imagine_batch_size,
                imagine_demonstration_batch_size=imagine_demonstration_batch_size,
                imagine_context_length=imagine_context_length,
                imagine_batch_length=imagine_batch_length,
                log_video=log_video,
                logger=logger
            )

            agent.update(
                latent=imagine_latent,
                action=agent_action,
                old_logprob=agent_logprob,
                old_value=agent_value,
                reward=imagine_reward,
                termination=imagine_termination,
                logger=logger
            )
        # <<< train agent part

        # save model per episode
        if total_steps % (save_every_steps//num_envs) == 0:
            print(colorama.Fore.GREEN + f"Saving model at total steps {total_steps}" + colorama.Style.RESET_ALL)
            torch.save(world_model.state_dict(), ckpt_path+f"/world_model_{total_steps}.pth")
            torch.save(agent.state_dict(), ckpt_path+f"/agent_{total_steps}.pth")


def build_world_model(conf, args,action_dim, device):
    if args.base_model == "Mamba":
        from sub_models.mamba_wm3 import WorldModel # 372
        return WorldModel(
            in_channels=conf.Models.WorldModel.InChannels,
            action_dim=action_dim,
            transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
            transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
            transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
            device = device 
        ).cuda(device)
    elif args.base_model == "Glam":
        from sub_models.mamba_wm12 import WorldModel
        return WorldModel(
            in_channels=conf.Models.WorldModel.InChannels,
            action_dim=action_dim,
            transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
            transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
            transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
            short_layers=conf.Models.WorldModel.ShortNumLayers,
            device = device 
        ).cuda(device)
    elif args.base_model == 'Transformer':
        from sub_models.world_models import WorldModel
        return WorldModel(
            in_channels=conf.Models.WorldModel.InChannels,
            action_dim=action_dim,
            transformer_max_length=conf.Models.WorldModel.TransformerMaxLength,
            transformer_hidden_dim=conf.Models.WorldModel.TransformerHiddenDim,
            transformer_num_layers=conf.Models.WorldModel.TransformerNumLayers,
            transformer_num_heads=conf.Models.WorldModel.TransformerNumHeads,
            device = device
        ).cuda(device)
    else:
        raise NotImplementedError(f"Model {args.base_model} not implemented")


def build_agent(conf, action_dim, device):
    return agents.ActorCriticAgent(
        feat_dim=32*32+conf.Models.WorldModel.TransformerHiddenDim,
        num_layers=conf.Models.Agent.NumLayers,
        hidden_dim=conf.Models.Agent.HiddenDim,
        action_dim=action_dim,
        gamma=conf.Models.Agent.Gamma,
        lambd=conf.Models.Agent.Lambda,
        entropy_coef=conf.Models.Agent.EntropyCoef,
        device = device
    ).cuda(device)

def build_replay_buffer(conf):
    if args.sample == 'normal':
        print('args.sample == normal')
        from replay_buffer import ReplayBuffer
        replay_buffer = ReplayBuffer(
            obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
            action_dim=action_dim,
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_length=conf.JointTrainAgent.BufferMaxLength,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU,
            device=device
        )
    elif args.sample == 'time_balance':
        from replay_buffer import TB_ReplayBuffer
        replay_buffer = TB_ReplayBuffer(
            obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
            action_dim=action_dim,
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_length=conf.JointTrainAgent.BufferMaxLength,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU,
            device=device
        )    
    elif args.sample == 'tb_agent_sample':
        from replay_buffer import TB_Agent_ReplayBuffer
        replay_buffer = TB_Agent_ReplayBuffer(
            obs_shape=(conf.BasicSettings.ImageSize, conf.BasicSettings.ImageSize, 3),
            action_dim=action_dim,
            num_envs=conf.JointTrainAgent.NumEnvs,
            max_length=conf.JointTrainAgent.BufferMaxLength,
            warmup_length=conf.JointTrainAgent.BufferWarmUp,
            store_on_gpu=conf.BasicSettings.ReplayBufferOnGPU,
            device=device
        )  
    else:
        raise NotImplementedError(f"Sample {conf.sample} not implemented")   
        
    return replay_buffer

def save_args(args, path):
    with open(path, 'w') as f:
        json.dump(args.__dict__, f, indent=4)

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
    parser.add_argument("-suite", type=str, default='atari', required=False)
    parser.add_argument("-env_name", type=str, default='RoadRunner', required=False)
    parser.add_argument("-seed", type=int, default=1, required=False)
    parser.add_argument("-base_model", type=str, default='Glam', required=False)
    parser.add_argument("-version", type=str, default='1_1', required=False)
    parser.add_argument("-config_path", type=str, default='config_files/Glam.yaml', required=False)

    parser.add_argument("-cuda_device", type=int, default=0, required=False)
    parser.add_argument("-sample", type=str, default='normal', required=False)

    args = parser.parse_args()

    with open(args.config_path, 'r') as file:
        conf = yaml.safe_load(file)
        conf = Config(conf)

    print(colorama.Fore.RED + str(args) + colorama.Style.RESET_ALL)
    
    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")

    logger_path = f"data/runs/{args.env_name}_seed{args.seed}_{args.base_model}_{args.version}"
    os.makedirs(logger_path, exist_ok=True)
    # set seed
    seed_np_torch(seed=args.seed)
    # tensorboard writer
    logger = Logger(path=logger_path)
    # copy config file
    shutil.copy(args.config_path, logger_path+f"/config.yaml")

    # distinguish between tasks, other debugging options are removed for simplicity
    if conf.Task == "JointTrainAgent":
        # getting action_dim with dummy env
        dummy_env,_, action_dim = make_env(args.suite ,args.env_name, seed=0)

        # build world model and agent
        world_model = build_world_model(conf, args, action_dim, device)
        agent = build_agent(conf, action_dim, device)

        # DEBUG
        # world_model.load_state_dict(torch.load(f"/home/hq/LSTW/MSTORM_base/data/ckpt/Breakout_seed1_Mamba11_772632/world_model_101719.pth", map_location=torch.device("cuda:0")))
        # agent.load_state_dict(torch.load(f"/home/hq/LSTW/MSTORM_base/data/ckpt/Breakout_seed1_Mamba11_772632/agent_101719.pth", map_location=torch.device("cuda:0")))

        import tools as t
        t.model_structure(world_model)
        t.model_structure(agent)

        # build replay buffer
        replay_buffer =  build_replay_buffer(conf)

        # create ckpt dir
        ckpt_path = f"data/ckpt/{args.env_name}_seed{args.seed}_{args.base_model}_{args.version}"
        os.makedirs(ckpt_path, exist_ok=True)
        shutil.copy(args.config_path, ckpt_path+f"/config.yaml")
        save_args(args, ckpt_path+f"/args.json")

        # train
        joint_train_world_model_agent(
            suite=args.suite,
            env_name=args.env_name,
            max_steps=conf.JointTrainAgent.SampleMaxSteps,
            replay_buffer=replay_buffer,
            world_model=world_model,
            agent=agent,
            train_dynamics_every_steps=conf.JointTrainAgent.TrainDynamicsEverySteps,
            train_agent_every_steps=conf.JointTrainAgent.TrainAgentEverySteps,
            batch_size=conf.JointTrainAgent.BatchSize,
            demonstration_batch_size=conf.JointTrainAgent.DemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            batch_length=conf.JointTrainAgent.BatchLength,
            imagine_batch_size=conf.JointTrainAgent.ImagineBatchSize,
            imagine_demonstration_batch_size=conf.JointTrainAgent.ImagineDemonstrationBatchSize if conf.JointTrainAgent.UseDemonstration else 0,
            imagine_context_length=conf.JointTrainAgent.ImagineContextLength,
            imagine_batch_length=conf.JointTrainAgent.ImagineBatchLength,
            save_every_steps=conf.JointTrainAgent.SaveEverySteps,
            seed=args.seed,
            logger=logger,
            incrementalimagine=conf.IncrementalImagine,
            ckpt_path=ckpt_path
        )
    else:
        raise NotImplementedError(f"Task {conf.Task} not implemented")