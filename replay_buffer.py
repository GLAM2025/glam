import numpy as np
import random
import unittest
import torch
from einops import rearrange
import copy
import pickle
import torch.nn.functional as F
import os

class ReplayBuffer():
    def __init__(self, obs_shape, action_dim, num_envs, max_length=int(1E6), warmup_length=50000, store_on='gpu', device=None) -> None:
        self.store_on = store_on
        self.device = device
        if store_on == 'disk':

            self.directory = os.path.join(os.getcwd(), "replay_buffer")
            os.makedirs(self.directory, exist_ok=True)

            self.obs_file = os.path.join(self.directory, "obs_buffer.npy")
            self.action_file = os.path.join(self.directory, "action_buffer.npy")
            self.reward_file = os.path.join(self.directory, "reward_buffer.npy")
            self.termination_file = os.path.join(self.directory, "termination_buffer.npy")
            self.metadata_file = os.path.join(self.directory, "metadata.pkl")

            for file_path in [self.obs_file, self.action_file, self.reward_file, self.termination_file, self.metadata_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)

            np.save(self.obs_file, np.empty((max_length, num_envs, *obs_shape), dtype=np.uint8))
            np.save(self.action_file, np.empty((max_length, num_envs, action_dim), dtype=np.float32))
            np.save(self.reward_file, np.empty((max_length, num_envs), dtype=np.float32))
            np.save(self.termination_file, np.empty((max_length, num_envs), dtype=np.float32))
            self._save_metadata({"length": 0, "last_pointer": -1})

        elif store_on == 'gpu':
            self.obs_buffer = torch.empty((max_length//num_envs, num_envs, *obs_shape), dtype=torch.uint8, device=self.device, requires_grad=False)
            self.action_buffer = torch.empty((max_length//num_envs, num_envs, action_dim), dtype=torch.float32, device=self.device, requires_grad=False)
            self.reward_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=self.device, requires_grad=False)
            self.termination_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=self.device, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length//num_envs, num_envs, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_length//num_envs, num_envs, action_dim), dtype=np.float32)
            self.reward_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.termination_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)

        self.length = 0
        self.num_envs = num_envs
        self.last_pointer = -1
        self.max_length = max_length
        self.warmup_length = warmup_length
        self.external_buffer_length = None

    def load_trajectory(self, path):
        buffer = pickle.load(open(path, "rb"))
        if self.store_on_gpu:
            self.external_buffer = {name: torch.from_numpy(buffer[name]).to("cuda") for name in buffer}
        else:
            self.external_buffer = buffer
        self.external_buffer_length = self.external_buffer["obs"].shape[0]

    def sample_external(self, batch_size, batch_length):
        indexes = np.random.randint(0, self.external_buffer_length+1-batch_length, size=batch_size)
        if self.store_on_gpu:
            obs = torch.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = torch.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = torch.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = torch.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        else:
            obs = np.stack([self.external_buffer["obs"][idx:idx+batch_length] for idx in indexes])
            action = np.stack([self.external_buffer["action"][idx:idx+batch_length] for idx in indexes])
            reward = np.stack([self.external_buffer["reward"][idx:idx+batch_length] for idx in indexes])
            termination = np.stack([self.external_buffer["done"][idx:idx+batch_length] for idx in indexes])
        return obs, action, reward, termination

    def ready(self):
        return self.length * self.num_envs > self.warmup_length
    
    def _save_metadata(self, metadata):
        with open(self.metadata_file, "wb") as f:
            pickle.dump(metadata, f)

    def _load_metadata(self):
        with open(self.metadata_file, "rb") as f:
            return pickle.load(f)
        
    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length):
        if self.store_on == 'disk':
            metadata = self._load_metadata()
            length = metadata["length"]

            if length < batch_length:
                raise ValueError("Not enough data to sample.")

            # 随机采样起始索引
            indexes = np.random.randint(0, length + 1 - batch_length, size=batch_size)

            # 加载现有数据
            obs_buffer = np.load(self.obs_file, mmap_mode="r")
            action_buffer = np.load(self.action_file, mmap_mode="r")
            reward_buffer = np.load(self.reward_file, mmap_mode="r")
            termination_buffer = np.load(self.termination_file, mmap_mode="r")

            # 采样数据
            obs = np.stack([obs_buffer[idx:idx + batch_length] for idx in indexes])
            action = np.stack([action_buffer[idx:idx + batch_length] for idx in indexes])
            reward = np.stack([reward_buffer[idx:idx + batch_length] for idx in indexes])
            termination = np.stack([termination_buffer[idx:idx + batch_length] for idx in indexes])

            obs = torch.from_numpy(obs).float().cuda().squeeze(2)/ 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.from_numpy(action).cuda().squeeze(2)
            reward = torch.from_numpy(reward).cuda().squeeze(2)
            termination = torch.from_numpy(termination).cuda().squeeze(2)

            return obs, action, reward, termination
        elif self.store_on == 'gpu':
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    # indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
                    indexes = np.random.randint(0, self.length-batch_length-5, size=batch_size//self.num_envs-5)
                    indexes = np.append(indexes, [self.length-batch_length-1, 
                                                  self.length-batch_length-2, 
                                                  self.length-batch_length-3, 
                                                  self.length-batch_length-4, 
                                                  self.length-batch_length-5])
                    obs.append(torch.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(torch.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(torch.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(torch.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_action_logit, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)

            obs = torch.cat(obs, dim=0).float() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.cat(action, dim=0)
            reward = torch.cat(reward, dim=0)
            termination = torch.cat(termination, dim=0)

            return obs, action, reward, termination
        else:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = np.random.randint(0, self.length+1-batch_length, size=batch_size//self.num_envs)
                    obs.append(np.stack([self.obs_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    action.append(np.stack([self.action_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    reward.append(np.stack([self.reward_buffer[idx:idx+batch_length, i] for idx in indexes]))
                    termination.append(np.stack([self.termination_buffer[idx:idx+batch_length, i] for idx in indexes]))

            if self.external_buffer_length is not None and external_batch_size > 0:
                external_obs, external_action, external_action_logit, external_reward, external_termination = self.sample_external(
                    external_batch_size, batch_length)
                obs.append(external_obs)
                action.append(external_action)
                reward.append(external_reward)
                termination.append(external_termination)

            obs = torch.from_numpy(np.concatenate(obs, axis=0)).float().cuda() / 255
            obs = rearrange(obs, "B T H W C -> B T C H W")
            action = torch.from_numpy(np.concatenate(action, axis=0)).cuda()
            reward = torch.from_numpy(np.concatenate(reward, axis=0)).cuda()
            termination = torch.from_numpy(np.concatenate(termination, axis=0)).cuda()

            return obs, action, reward, termination

    def append(self, obs, action, reward, termination):
        # obs/nex_obs: torch Tensor
        # action/reward/termination: int or float or bool
        self.last_pointer = (self.last_pointer + 1) % (self.max_length//self.num_envs)
        if self.store_on == 'disk':
            metadata = self._load_metadata()
            length = metadata["length"]
            last_pointer = (metadata["last_pointer"] + 1) % self.max_length

            # 加载现有数据
            obs_buffer = np.load(self.obs_file, mmap_mode="r+")
            action_buffer = np.load(self.action_file, mmap_mode="r+")
            reward_buffer = np.load(self.reward_file, mmap_mode="r+")
            termination_buffer = np.load(self.termination_file, mmap_mode="r+")

            # 写入新数据
            obs_buffer[last_pointer] = obs
            action_buffer[last_pointer] = action
            reward_buffer[last_pointer] = reward
            termination_buffer[last_pointer] = termination

            # 更新元数据
            metadata["last_pointer"] = last_pointer
            metadata["length"] = min(length + 1, self.max_length)
            self._save_metadata(metadata)    
        elif self.store_on == 'gpu':
            self.obs_buffer[self.last_pointer] = torch.from_numpy(obs.copy())
            self.action_buffer[self.last_pointer] = torch.from_numpy(action)
            self.reward_buffer[self.last_pointer] = torch.from_numpy(reward)
            self.termination_buffer[self.last_pointer] = torch.from_numpy(termination)
        else:
            self.obs_buffer[self.last_pointer] = obs
            self.action_buffer[self.last_pointer] = action
            self.reward_buffer[self.last_pointer] = reward
            self.termination_buffer[self.last_pointer] = termination

        if len(self) < self.max_length:
            self.length += 1

    def __len__(self):
        return self.length * self.num_envs
