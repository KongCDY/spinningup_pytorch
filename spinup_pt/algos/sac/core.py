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

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(torch.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return torch.sum(pre_sum, dim=1, keepdim = True)

def clip_but_pass_gradient(x, l=-1., u=1.):
    clip_up = x > u
    clip_low = x < l
    return x + ((u - x)*clip_up.float() + (l - x)*clip_low.float()).detach()

class MLP(nn.Module):
    def __init__(self, in_dim, h_sizes, activation = nn.Tanh, output_activation = None):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(in_dim, h_sizes[0]))
        layers.append(activation())
        for i in range(1, len(h_sizes) - 1):
            layers.append(nn.Linear(h_sizes[i - 1], h_sizes[i]))
            layers.append(activation())
        # last layer
        layers.append(nn.Linear(h_sizes[-2], h_sizes[-1]))
        if output_activation is not None:
            layers.append(output_activation())
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

"""
Policies
"""

LOG_STD_MAX = 2
LOG_STD_MIN = -20

class GaussianPolicy(nn.Module):
    def __init__(self, in_dim, hidden_sizes, out_dim, activation = nn.ReLU):
        super(GaussianPolicy, self).__init__()
        self.mlp = MLP(in_dim, hidden_sizes, activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], out_dim)
        self.logstd_layer = nn.Sequential(
                nn.Linear(hidden_sizes[-1], out_dim),
                nn.Tanh(),)
        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight.data)
                init.constant_(m.bias.data, 0.0)

    def apply_squashing_func(self, mu, pi, logp_pi):
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)
        # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
        logp_pi -= torch.sum(torch.log(clip_but_pass_gradient(1 - pi**2, l=0, u=1) + 1e-6), dim=1, keepdim = True)
        return mu, pi, logp_pi
    
    def forward(self, x):
        hid = self.mlp(x)
        mu = self.mu_layer(hid)

        """
        Because algorithm maximizes trade-off of reward and entropy,
        entropy must be unique to state---and therefore log_stds need
        to be a neural network output instead of a shared-across-states
        learnable parameter vector. But for deep Relu and other nets,
        simply sticking an activationless dense layer at the end would
        be quite bad---at the beginning of training, a randomly initialized
        net could produce extremely large values for the log_stds, which
        would result in some actions being either entirely deterministic
        or too random to come back to earth. Either of these introduces
        numerical instability which could break the algorithm. To 
        protect against that, we'll constrain the output range of the 
        log_stds, to lie within [LOG_STD_MIN, LOG_STD_MAX]. This is 
        slightly different from the trick used by the original authors of
        SAC---they used tf.clip_by_value instead of squashing and rescaling.
        I prefer this approach because it allows gradient propagation
        through log_std where clipping wouldn't, but I don't know if
        it makes much of a difference.
        """
        log_std = self.logstd_layer(hid)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        std = torch.exp(log_std)
        pi = mu + torch.randn_like(mu) * std
        logp_pi = gaussian_likelihood(pi, mu, log_std)

        mu, pi, logp_pi = self.apply_squashing_func(mu, pi, logp_pi)
        return mu, pi, logp_pi

class Actor_Critic(nn.Module):
    def __init__(self, in_dim, action_space, hidden_sizes = (400, 300), activation = nn.ReLU, output_activation = None):
        super(Actor_Critic, self).__init__()
        out_dim = action_space.shape[0]
        self.action_scale = action_space.high[0]
        self.actor = GaussianPolicy(in_dim, hidden_sizes, out_dim, activation)
        self.qf1 = MLP(in_dim + out_dim, hidden_sizes + [1], activation)
        self.qf2 = MLP(in_dim + out_dim, hidden_sizes + [1], activation)
        self.vf = MLP(in_dim, hidden_sizes + [1], activation)
        self.vf_targ = MLP(in_dim, hidden_sizes + [1], activation)
        self.vf_targ.eval()

    def apply_policy(self, x):
        mu, pi, logp_pi = self.actor(x)

        # make sure actions are in correct range
        mu = mu * self.action_scale
        pi = pi * self.action_scale
        return mu, pi, logp_pi

    def apply_qf(self, x, a):
        q1 = self.qf1(torch.cat([x, a], dim = 1))
        q2 = self.qf2(torch.cat([x, a], dim = 1))
        return q1, q2

    def apply_vf(self, x):
        return self.vf(x)

    def apply_vf_targ(self, x):
        return self.vf_targ(x)

    def get_action(self, x, deterministic=False):
        x = torch.from_numpy(x.astype(np.float32))
        with torch.no_grad():
            mu, pi, _ = self.actor(x)
        if deterministic:
            act = mu * self.action_scale
        else:
            act = pi * self.action_scale
        return act.detach().numpy()[0]

