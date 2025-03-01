

import gym
import torch as th
import torch.nn as nn
import torch.distributions as td
from stable_baselines3.common.policies import ActorCriticPolicy

class CustomDiagGaussianDistribution:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.dist = td.Normal(mean, std)
        
    def get_actions(self, deterministic=False):
        if deterministic:
            return self.mean
        else:
            return self.dist.rsample()  # 使用 rsample 保证可微分
    
    def log_prob(self, actions):
        return self.dist.log_prob(actions).sum(dim=-1)
    
    def entropy(self):
        return self.dist.entropy().sum(dim=-1)

class CustomNetwork(nn.Module):
    def __init__(self, feature_dim, last_layer_dim, min_thrust, max_thrust):
        super().__init__()
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust
        # self.angular_scale = angular_scale
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.Tanh(),
            # nn.Linear(128, 128),
            # nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, last_layer_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        raw_output = self.net(x)
        thrust = self.min_thrust + (raw_output[:, 0] + 1.0) * 0.5 * (self.max_thrust - self.min_thrust)
        thrust = th.clamp(thrust, self.min_thrust, self.max_thrust)
        # rates = raw_output[:, 1:] * self.angular_scale
        return th.cat([th.unsqueeze(thrust, 1)], dim=1)

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, 
                 observation_space: gym.spaces.Space, 
                 action_space: gym.spaces.Space, 
                 lr_schedule, 
                 min_thrust: float = 0, 
                 max_thrust: float = 28.1 ,
                #  angular_scale: float = 3.0,
                 **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        
        self.min_thrust = min_thrust
        self.max_thrust = max_thrust
        # self.angular_scale = angular_scale
        
        feature_dim = self.features_extractor.features_dim
        self.actor_net = CustomNetwork(feature_dim, action_space.shape[0], min_thrust, max_thrust)
        self.log_std = nn.Parameter(th.ones(action_space.shape[0]) * -0.5)  # 初始 std ≈ 0.6

    def _get_action_dist_from_latent(self, latent_pi):
        mean = self.actor_net(latent_pi)
        std = th.exp(self.log_std)
        return self._make_dist_from_mean_std(mean, std)

    def _make_dist_from_mean_std(self, mean, std):
        return CustomDiagGaussianDistribution(mean, std)