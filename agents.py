import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import copy
from torch.cuda.amp import autocast
from torch import distributions as torchd

from sub_models.functions_losses import SymLogTwoHotLoss
from utils import EMAScalar


def percentile(x, percentage):
    flat_x = torch.flatten(x)
    kth = int(percentage*len(flat_x))
    per = torch.kthvalue(flat_x, kth).values
    return per


def calc_lambda_return(rewards, values, termination, gamma, lam, dtype=torch.float32, device=None):
    # Invert termination to have 0 if the episode ended and 1 otherwise
    inv_termination = (termination * -1) + 1

    batch_size, batch_length = rewards.shape[:2]
    # gae_step = torch.zeros((batch_size, ), dtype=dtype, device="cuda")
    gamma_return = torch.zeros((batch_size, batch_length+1), dtype=dtype, device=device)
    gamma_return[:, -1] = values[:, -1]
    for t in reversed(range(batch_length)):  # with last bootstrap
        gamma_return[:, t] = \
            rewards[:, t] + \
            gamma * inv_termination[:, t] * (1-lam) * values[:, t] + \
            gamma * inv_termination[:, t] * lam * gamma_return[:, t+1]
    return gamma_return[:, :-1]

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.dim = dim
        self.eps = eps

    def forward(self, x):

        rms = torch.sqrt(torch.mean(x**2, dim=self.dim, keepdim=True) + self.eps)

        normalized_x = x / rms
        return normalized_x


class ActorCriticAgent(nn.Module):
    def __init__(self, feat_dim, num_layers, hidden_dim, action_dim, gamma, lambd, entropy_coef, device) -> None:
        super().__init__()
        self.gamma = gamma
        self.lambd = lambd
        self.entropy_coef = entropy_coef
        self.use_amp = True
        self.tensor_dtype = torch.bfloat16 if self.use_amp else torch.float32
        self._size = action_dim
        self.device = device

        self.symlog_twohot_loss = SymLogTwoHotLoss(255, -20, 20)

        actor = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            actor.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
        self.actor = nn.Sequential(
            *actor,
            nn.Linear(hidden_dim, 2*action_dim)
        )
        
        critic = [
            nn.Linear(feat_dim, hidden_dim, bias=False),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            critic.extend([
                nn.Linear(hidden_dim, hidden_dim, bias=False),
                nn.LayerNorm(hidden_dim),
                nn.ReLU()
            ])
        
        self.critic = nn.Sequential(
            *critic,
            nn.Linear(hidden_dim, 255)
        )
        self.slow_critic = copy.deepcopy(self.critic)

        self.lowerbound_ema = EMAScalar(decay=0.99)
        self.upperbound_ema = EMAScalar(decay=0.99)

        lr=3e-5
        print("agent lr ",lr)
        self.optimizer = torch.optim.Adam(self.parameters(), lr, eps=1e-5)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    @torch.no_grad()
    def update_slow_critic(self, decay=0.98):
        for slow_param, param in zip(self.slow_critic.parameters(), self.critic.parameters()):
            slow_param.data.copy_(slow_param.data * decay + param.data * (1 - decay))

    def policy(self, x):
        x = self.actor(x)
        mean, std = torch.split(x, [self._size] * 2, -1)
        mean = torch.tanh(mean)
        return mean, std 

    def value(self, x):
        value = self.critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    @torch.no_grad()
    def slow_value(self, x):
        value = self.slow_critic(x)
        value = self.symlog_twohot_loss.decode(value)
        return value

    def get_logits_raw_value(self, x):
        logits = self.actor(x)
        raw_value = self.critic(x)
        return logits, raw_value

    @torch.no_grad()
    def sample(self, latent, greedy=False):
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            mean, std  = self.policy(latent)
            
            if greedy:
                action = mean
            else:
                std = F.softplus(std) + 0.0001
                dist = torchd.normal.Normal(mean, std)
                action = dist.sample()
                action = torch.clamp(action, min=-1.0, max=1.0)  
        return action

    def sample_as_env_action(self, latent, greedy=False):
        action = self.sample(latent, greedy)
        return action.detach().cpu().float().squeeze(0).squeeze(0).numpy()   # actioin: aray([1])
    
    def sample_td_action(self, latents, actions, wm, horizon=4):
        self.eval()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp) and torch.no_grad():
            # list action space
            prior_flattened_sample, last_dist_feat = wm.calc_last_dist_feat(latents, actions)
            logits = self.policy(torch.cat([prior_flattened_sample, last_dist_feat], dim=-1))
            dist = distributions.Categorical(logits=logits)
            # assert dist.probs.shape[0] == 1, f'{dist.probs.shape}'
            _, idx = dist.probs.sort(descending=True, dim=-1)

            # sample pi trajectories
            num_pi_trajs = 3
            traj_reward = torch.empty(
                dist.probs.shape[0], num_pi_trajs, horizon, device=self.device
            )
            for i in range(num_pi_trajs):
                action = idx[:,:,i]
                context_latent = latents.clone()
                context_action = actions.clone() 
                for t in range(horizon):
                    if context_latent.shape[1] > 64:
                        last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = wm.predict_next(
                            context_latent[:, -24:], context_action[:, -24:])
                    else:
                        last_obs_hat, last_reward_hat, last_termination_hat, last_latent, last_dist_feat = wm.predict_next(
                            context_latent, context_action)
                    context_latent = torch.cat([context_latent, last_latent], dim=1)
                    action = self.sample(torch.cat([last_latent, last_dist_feat], dim=-1), greedy=True)
                    context_action = torch.cat([context_action, action], dim=1)
                    traj_reward[:, i, t] = last_reward_hat.squeeze(-1)

            # Output
            test = traj_reward.mean(dim=-1)
            best_traj = traj_reward.mean(dim=-1).argmax(dim=-1)
            return idx[torch.arange(dist.probs.shape[0]), :, best_traj].detach().cpu().squeeze(-1).numpy()

    def update(self, latent, action, old_logprob, old_value, reward, termination, logger=None):
        '''
        Update policy and value model
        '''

        # sum_t = torch.sum(termination, dim=1)
        # print(torch.sum(sum_t))

        self.train()
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.use_amp):
            logits, raw_value = self.get_logits_raw_value(latent)
            mean, std = torch.split(logits, [self._size] * 2, -1)
            mean = torch.tanh(mean)
            std = F.softplus(std) + 0.0001
            dist = torchd.normal.Normal(mean[:, :-1], std[:, :-1])
            # dist = distributions.Categorical(logits=logits[:, :-1])
            log_prob = dist.log_prob(action)
            log_prob = log_prob.sum(dim=-1, keepdim=False)
            entropy = dist.entropy()

            # decode value, calc lambda return
            slow_value = self.slow_value(latent)
            slow_lambda_return = calc_lambda_return(reward, slow_value, termination, self.gamma, self.lambd, device=self.device)
            value = self.symlog_twohot_loss.decode(raw_value)
            lambda_return = calc_lambda_return(reward, value, termination, self.gamma, self.lambd, device=self.device)

            # update value function with slow critic regularization
            value_loss = self.symlog_twohot_loss(raw_value[:, :-1], lambda_return.detach())
            slow_value_regularization_loss = self.symlog_twohot_loss(raw_value[:, :-1], slow_lambda_return.detach())

            lower_bound = self.lowerbound_ema(percentile(lambda_return, 0.05))
            upper_bound = self.upperbound_ema(percentile(lambda_return, 0.95))
            S = upper_bound-lower_bound
            norm_ratio = torch.max(torch.ones(1).cuda(), S)  # max(1, S) in the paper
            norm_advantage = (lambda_return-value[:, :-1]) / norm_ratio
            policy_loss = -(log_prob * norm_advantage.detach()).mean()

            entropy_loss = entropy.mean()

            loss = policy_loss + value_loss + slow_value_regularization_loss - self.entropy_coef * entropy_loss

        # gradient descent
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)  # for clip grad
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=100.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)

        self.update_slow_critic()

        if logger is not None:
            logger.log('ActorCritic/policy_loss', policy_loss.item())
            logger.log('ActorCritic/value_loss', value_loss.item())
            logger.log('ActorCritic/entropy_loss', entropy_loss.item())
            logger.log('ActorCritic/S', S.item())
            logger.log('ActorCritic/norm_ratio', norm_ratio.item())
            logger.log('ActorCritic/total_loss', loss.item())
