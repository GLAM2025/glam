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
        # 自动设置存储目录
            self.directory = os.path.join(os.getcwd(), "replay_buffer")
            os.makedirs(self.directory, exist_ok=True)

            # 初始化文件路径
            self.obs_file = os.path.join(self.directory, "obs_buffer.npy")
            self.action_file = os.path.join(self.directory, "action_buffer.npy")
            self.reward_file = os.path.join(self.directory, "reward_buffer.npy")
            self.termination_file = os.path.join(self.directory, "termination_buffer.npy")
            self.metadata_file = os.path.join(self.directory, "metadata.pkl")

            # 如果文件存在，删除旧文件
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
    
class TB_ReplayBuffer():
    def __init__(self, obs_shape, action_dim, num_envs, max_length=int(1E6), warmup_length=50000, store_on_gpu=False, device=None) -> None:
        self.store_on_gpu = store_on_gpu
        self.device = device
        if store_on_gpu:
            self.obs_buffer = torch.empty((max_length//num_envs, num_envs, *obs_shape), dtype=torch.uint8, device=self.device, requires_grad=False)
            self.action_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=self.device, requires_grad=False)
            self.reward_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=self.device, requires_grad=False)
            self.termination_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=self.device, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length//num_envs, num_envs, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.reward_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.termination_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)

        self.sample_visits = torch.zeros(max_length//num_envs, dtype=torch.long, device='cpu')  # just sample indices on cpu

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

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length):
        if self.store_on_gpu:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = self.sample_indices(batch_size, batch_length)
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
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = torch.from_numpy(obs)
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

    def _compute_visit_probs(self, n, sample2agent):
        temperature = 20
        if temperature == 'inf':
            visits = self.sample_visits[:n].float()
            visit_sum = visits.sum()
            if visit_sum == 0:
                probs = torch.full_like(visits, 1 / n)
            else:
                probs = 1 - visits / visit_sum
        else:
            a = -temperature*((-1)**sample2agent)
            logits = self.sample_visits[:n].float() / -temperature*((-1)**sample2agent)
            probs = F.softmax(logits, dim=0)
        assert probs.device.type == 'cpu'
        return probs

    def sample_indices(self, batch_size, length):   # 默认单个环境
        n = self.length - length + 1
        sample2agent = batch_size==1024     # 现在默认 agent 训练的 bs==1024， 以后可以分开写函数
        # if batch_size * length > n:   
        #     raise ValueError('Not enough data in buffer')

        probs = self._compute_visit_probs(n, sample2agent)
        start_idx = torch.multinomial(probs, batch_size, replacement=False)

        all_idx = start_idx.unsqueeze(-1) + torch.arange(length, device=start_idx.device)   # (batch_size, length)

        # stay on cpu
        flat_idx = all_idx.reshape(-1)
        flat_idx, counts = torch.unique(flat_idx, return_counts=True)
        self.sample_visits[flat_idx] += counts

        start_idx = start_idx.numpy()
        # idx = start_idx.unsqueeze(-1) + torch.arange(length, device=torch.device('cuda'))
        return start_idx
    
class TB_Agent_ReplayBuffer():
    def __init__(self, obs_shape, action_dim, num_envs, max_length=int(1E6), warmup_length=50000, store_on_gpu=False, device=None) -> None:
        self.store_on_gpu = store_on_gpu
        self.device = device
        if store_on_gpu:
            self.obs_buffer = torch.empty((max_length//num_envs, num_envs, *obs_shape), dtype=torch.uint8, device=self.device, requires_grad=False)
            self.action_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=self.device, requires_grad=False)
            self.reward_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=self.device, requires_grad=False)
            self.termination_buffer = torch.empty((max_length//num_envs, num_envs), dtype=torch.float32, device=self.device, requires_grad=False)
        else:
            self.obs_buffer = np.empty((max_length//num_envs, num_envs, *obs_shape), dtype=np.uint8)
            self.action_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.reward_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)
            self.termination_buffer = np.empty((max_length//num_envs, num_envs), dtype=np.float32)

        self.sample_wm_visits = torch.zeros(max_length//num_envs, dtype=torch.long, device='cpu')  # just sample indices on cpu
        self.sample_agent_visits = torch.zeros(max_length//num_envs, dtype=torch.long, device='cpu')  # just sample indices on cpu
        self.sample2wm_times = 0

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

    @torch.no_grad()
    def sample(self, batch_size, external_batch_size, batch_length):
        if self.store_on_gpu:
            obs, action, reward, termination = [], [], [], []
            if batch_size > 0:
                for i in range(self.num_envs):
                    indexes = self.sample_indices(batch_size, batch_length)
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
        if self.store_on_gpu:
            self.obs_buffer[self.last_pointer] = torch.from_numpy(obs)
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

    def _compute_visit_probs(self, n):
        wm_temperature = 10 # wm_t 越大，wm 采样过的样本占比越小
        agent_temperature = 80   # agent_t 越大，agent 采样过的样本占比越大
            
        wm_logits = self.sample_wm_visits[:n].float() / wm_temperature
        agent_logits = self.sample_agent_visits[:n].float() / -agent_temperature 

        wm_probs = F.softmax(wm_logits, dim=0)
        agent_probs = F.softmax(agent_logits, dim=0)

        probs = F.softmax(wm_probs * agent_probs, dim=0)
        assert probs.device.type == 'cpu'
        return probs

    def sample_indices(self, batch_size, length):   # 默认单个环境
        n = self.length - length + 1
        sample2agent = batch_size==1024     # 现在默认 agent 训练的 bs==1024， 以后可以分开写函数

        if sample2agent:
            probs = self._compute_visit_probs(n)
            start_idx = torch.multinomial(probs, batch_size, replacement=False)

            flat_idx = start_idx.reshape(-1)
            flat_idx, counts = torch.unique(flat_idx, return_counts=True)

            self.sample_agent_visits[flat_idx] += counts

            start_idx = start_idx.numpy()
        else:
            start_idx = np.random.randint(0, n, size=batch_size)
            # all_idx = start_idx.unsqueeze(-1) + torch.arange(length, device=start_idx.device)   # (batch_size, length)
            all_idx = start_idx[:, None] + np.arange(length)

            flat_idx = all_idx.reshape(-1)
            flat_idx, counts = np.unique(flat_idx, return_counts=True)
            self.sample_wm_visits[flat_idx] += counts

        return start_idx