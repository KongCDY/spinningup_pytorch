import numpy as np
import gym
from gym.spaces import Discrete, Box
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import torch.optim as optim
import matplotlib.pyplot as plt
import ipdb


class MLP(nn.Module):
    def __init__(self, in_dim, h_sizes, out_dim):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(in_dim, h_sizes[0]))
        layers.append(nn.Tanh())
        for i in range(1, len(h_sizes)):
            layers.append(nn.Linear(h_sizes[i - 1], h_sizes[i]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(h_sizes[-1], out_dim))
        self.mlp = nn.Sequential(*layers)

        # init
        # for m in self.modules():
            # if isinstance(m, nn.Linear):
                # init.xavier_normal_(m.weight.data, gain=0.02)
                # init.constant_(m.bias.data, 0.0)

    def forward(self, x):
        return self.mlp(x)

    def choose_action(self, x):
        logits = F.softmax(self.mlp(x), dim=1)
        return torch.multinomial(logits.data, 1)

class WeightCELoss(nn.Module):
    def __init__(self):
        super(WeightCELoss, self).__init__()
    def forward(self, inputs, actions, weights):
        logits = F.log_softmax(inputs, dim = 1)
        logits = torch.gather(logits, 1, actions.unsqueeze(1))
        return torch.mean(-logits.squeeze() * weights)

def reward_to_go(rews):
    n = len(rews)
    rtgs = np.zeros_like(rews)
    for i in reversed(range(n)):
        rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
    return rtgs

def train(env_name='CartPole-v0', hidden_sizes=[32], lr=1e-2, 
        epochs=50, batch_size=5000, render=False, use_rtg=False):
    # ipdb.set_trace()
    # make environment, check spaces, get obs / act dims
    all_rewards = []
    env = gym.make(env_name)
    assert isinstance(env.observation_space, Box), \
        "This example only works for envs with continuous state spaces."
    assert isinstance(env.action_space, Discrete), \
        "This example only works for envs with discrete action spaces."

    obs_dim = env.observation_space.shape[0]
    n_acts = env.action_space.n

    # make core of policy network
    net = MLP(obs_dim, hidden_sizes, n_acts)
    print(net)

    # loss function
    criterion = WeightCELoss()

    # optim
    optimizer = optim.Adam(net.parameters(), lr = lr)

    # for training policy
    def train_one_epoch():
        # make some empty lists for logging.
        batch_obs = []          # for observations
        batch_acts = []         # for actions
        batch_weights = []      # for R(tau) weighting in policy gradient
        batch_rets = []         # for measuring episode returns
        batch_lens = []         # for measuring episode lengths

        # reset episode-specific variables
        obs = env.reset()       # first obs comes from starting distribution
        done = False            # signal from environment that episode is over
        ep_rews = []            # list for rewards accrued throughout ep

        # render first episode of each epoch
        finished_rendering_this_epoch = False

        # collect experience by acting in the environment with current policy
        while True:

            # rendering
            if (not finished_rendering_this_epoch) and render:
                env.render()

            # save obs
            batch_obs.append(obs.copy())

            # act in the environment
            with torch.no_grad():
                net.eval()
                act = net.choose_action(torch.from_numpy(obs[np.newaxis].astype(np.float32)))
                act = act.item()
            obs, rew, done, _ = env.step(act)

            # save action, reward
            batch_acts.append(act)
            ep_rews.append(rew)

            if done:
                # if episode is over, record info about episode
                ep_ret, ep_len = sum(ep_rews), len(ep_rews)
                batch_rets.append(ep_ret)
                batch_lens.append(ep_len)

                # the weight for each logprob(a|s) is R(tau)
                if use_rtg:
                    batch_weights += list(reward_to_go(ep_rews))
                else:
                    batch_weights += [ep_ret] * ep_len

                # reset episode-specific variables
                obs, done, ep_rews = env.reset(), False, []

                # won't render again this epoch
                finished_rendering_this_epoch = True

                # end experience loop if we have enough of it
                if len(batch_obs) > batch_size:
                    break

        # take a single policy gradient update step
        obs_ph = torch.from_numpy(np.array(batch_obs, dtype = np.float32))
        act_ph = torch.from_numpy(np.array(batch_acts))
        weights_ph = torch.from_numpy(np.array(batch_weights, dtype = np.float32))
        
        # forward and update param
        net.train()
        logits_ph = net(obs_ph)
        loss = criterion(logits_ph, act_ph, weights_ph)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        return batch_loss, batch_rets, batch_lens

    # training loop
    for i in range(epochs):
        batch_loss, batch_rets, batch_lens = train_one_epoch()
        all_rewards.append(np.mean(batch_rets))
        print('epoch: %3d \t loss: %.3f \t return: %.3f \t ep_len: %.3f'%
                (i, batch_loss, np.mean(batch_rets), np.mean(batch_lens)))
    # plot reward
    plt.figure()
    plt.plot(np.array(all_rewards))
    plt.xlabel('epoch')
    plt.ylabel('avg R')
    plt.title("Reward")
    plt.show()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--use_rtg', action='store_true')
    parser.add_argument('--lr', type=float, default=1e-2)
    args = parser.parse_args()
    print('\nUsing simplest formulation of policy gradient.\n')
    train(env_name=args.env_name, render=args.render, lr=args.lr, use_rtg=args.use_rtg)
