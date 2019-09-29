import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

# def gaussian_likelihood(x, mu, log_std):
    # pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    # return tf.reduce_sum(pre_sum, axis=1)

class MLP(nn.Module):
    def __init__(self, in_dim, h_sizes, out_dim, activation = nn.Tanh):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(in_dim, h_sizes[0]))
        layers.append(activation())
        for i in range(1, len(h_sizes)):
            layers.append(nn.Linear(h_sizes[i - 1], h_sizes[i]))
            layers.append(activation())
        layers.append(nn.Linear(h_sizes[-1], out_dim))
        self.mlp = nn.Sequential(*layers)

        # init
        # for m in self.modules():
            # if isinstance(m, nn.Linear):
                # init.xavier_normal_(m.weight.data, gain=0.02)
                # init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.mlp(x)

class Actor_Critic(nn.Module):
    def __init__(self, in_dim, action_space, hidden_sizes = (64, 64), activation = nn.Tanh):
        super(Actor_Critic, self).__init__()
        if isinstance(action_space, Discrete):
            self.action_type = 'D'
            out_dim = action_space.n
        elif isinstance(action_space, Box):
            self.action_type = 'C'
            out_dim = action_space.shape[0]
            self.log_std = -0.5 * torch.ones(1, out_dim)
        self.actor = MLP(in_dim, hidden_sizes, out_dim, activation)
        self.critic = MLP(in_dim, hidden_sizes, 1, activation)

    def gaussian_likelihood(self, x, mu, log_std):
        pre_sum = -0.5 * (((x - mu)/(torch.exp(log_std) + EPS))**2 + 2*log_std + float(np.log(2*np.pi)))
        return pre_sum.sum(dim = 1)

    def choose_action(self, logits):
        if self.action_type == 'D':
            probs = F.softmax(logits, dim=1)
            return torch.multinomial(probs.data, 1).squeeze(dim=1) # to (batch,)
        else:
            mu = logits
            act = mu + torch.randn(mu.size())*torch.exp(self.log_std)
            return act

    def apply_actor(self, x, a = None):
        logits = self.actor(x)
        pi = self.choose_action(logits)
        logp = None
        if self.action_type == 'D':
            logp_all = F.log_softmax(logits, dim = 1)
            logp_pi = torch.gather(logp_all, 1, pi.unsqueeze(1)).squeeze()
            if a is not None:
                logp = torch.gather(logp_all, 1, a.long().unsqueeze(1)).squeeze()
        else:
            logp_pi = self.gaussian_likelihood(pi, logits, self.log_std)
            if a is not None:
                logp = self.gaussian_likelihood(a, logits, self.log_std)

        return pi, logp, logp_pi

    def apply_critic(self, x):
        v = self.critic(x)
        return v.squeeze()

    def get_action(self, x):
        x = torch.from_numpy(x.astype(np.float32))
        logits = self.actor(x)
        action = self.choose_action(logits)
        return action.detach().numpy()[0]

