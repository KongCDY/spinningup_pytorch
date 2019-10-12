import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

def count_vars(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class MLP(nn.Module):
    def __init__(self, in_dim, h_sizes, out_dim, activation = nn.Tanh, output_activation = None):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(in_dim, h_sizes[0]))
        layers.append(activation())
        for i in range(1, len(h_sizes)):
            layers.append(nn.Linear(h_sizes[i - 1], h_sizes[i]))
            layers.append(activation())
        layers.append(nn.Linear(h_sizes[-1], out_dim))
        if output_activation is not None:
            layers.append(output_activation())
        self.mlp = nn.Sequential(*layers)

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.mlp(x)

class Actor_Critic(nn.Module):
    def __init__(self, in_dim, action_space, hidden_sizes = (400, 300), activation = nn.ReLU, output_activation = nn.Tanh):
        super(Actor_Critic, self).__init__()
        out_dim = action_space.shape[0]
        self.act_limit = action_space.high[0]
        self.actor = MLP(in_dim, hidden_sizes, out_dim, activation, output_activation)
        self.critic = MLP(in_dim + out_dim, hidden_sizes, 1, activation)

    def forward(self, x):
        pi = self.act_limit * self.actor(x)
        q_pi = self.apply_critic(x, pi)
        return q_pi

    def apply_critic(self, x, a):
        q = self.critic(torch.cat([x, a], dim = 1))
        return q

    def get_action(self, x, noise_scale = 0):
        x = torch.from_numpy(x.astype(np.float32))
        act = self.act_limit * self.actor(x)
        if noise_scale > 0:
            act += noise_scale * torch.randn_like(act)
            act.clamp_(-self.act_limit, self.act_limit)
        return act.detach().numpy()[0]

