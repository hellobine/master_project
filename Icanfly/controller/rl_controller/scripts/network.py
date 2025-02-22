

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from environment import QuadrotorEnv  # 假设已定义环境

# network.py 内容
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, lower_bound, upper_bound, angular_scale=1.0):
        """
        :param state_dim: 状态维度
        :param action_dim: 动作维度（这里应为4）
        :param lower_bound: 推力下限（例如 7.02 N）
        :param upper_bound: 推力上限（例如 21.06 N）
        :param angular_scale: 角速率缩放因子
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        self.fc2 = nn.Linear(64, 64)
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        self.mu = nn.Linear(64, action_dim)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)

        self.lower_bound = lower_bound  # 推力下限
        self.upper_bound = upper_bound  # 推力上限
        self.angular_scale = angular_scale  # 角速率缩放

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        raw = torch.tanh(self.mu(x))  # raw 的各元素范围在 [-1, 1]
        # 对第一个分量做非对称映射：将 [-1, 1] 映射到 [lower_bound, upper_bound]
        thrust = self.lower_bound + (raw[:, 0:1] + 1) / 2 * (self.upper_bound - self.lower_bound)
        # 对其余三个通道采用线性缩放
        rates = raw[:, 1:] * self.angular_scale
        return torch.cat([thrust, rates], dim=1)
    
class Critic(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        self.fc2 = nn.Linear(64, 64)
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        self.value = nn.Linear(64, 1)
        nn.init.orthogonal_(self.value.weight, gain=1.0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.value(x)
    

