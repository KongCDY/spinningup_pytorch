import numpy as np
import gym
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import torch.optim as optim
import spinup_pt.algos.ppo.core as core
from spinup_pt.utils.logx import EpochLogger
from spinup_pt.utils.mpi_tools import mpi_fork, proc_id, mpi_avg, mpi_statistics_scalar, num_procs
from spinup_pt.utils.mpi_torch import sync_all_params, average_gradients

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)    # observations
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)    # actions
        self.adv_buf = np.zeros(size, dtype=np.float32)                                  # advantage value
        self.rew_buf = np.zeros(size, dtype=np.float32)                                  # rewards
        self.ret_buf = np.zeros(size, dtype=np.float32)                                  # reward-to-go target value of value function
        self.val_buf = np.zeros(size, dtype=np.float32)                                  # critic value
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)
        
        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam)
        
        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]
        
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, 
                self.ret_buf, self.logp_buf]


"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""
def ppo(env_fn, actor_critic=core.Actor_Critic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:


            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure 
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to PPO.
            
        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while 
            still profiting (improving the objective function)? The new policy 
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        pi_lr (float): Learning rate for policy optimizer.

        vf_lr (float): Learning rate for value function optimizer.

        train_pi_iters (int): Maximum number of gradient descent steps to take 
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used 
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    seed += 10000 * proc_id()
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    
    # Share information about action space with policy architecture
    ac_kwargs['action_space'] = env.action_space

    # Experience buffer
    local_steps_per_epoch = int(steps_per_epoch / num_procs())
    buf = PPOBuffer(obs_dim, act_dim, local_steps_per_epoch, gamma, lam)

    # make core of policy network
    net = actor_critic(obs_dim[0], **ac_kwargs)
    print(net)

    # loss function
    criterion_mse = nn.MSELoss()

    # optim
    optimizer_actor = optim.Adam(net.actor.parameters(), lr = pi_lr)
    optimizer_critic = optim.Adam(net.critic.parameters(), lr = vf_lr)

    # Sync params across processes
    sync_all_params(net.parameters())

    # Count variables
    # var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    def update():
        obs_buf, act_buf, adv_buf, ret_buf, logp_buf = buf.get()
        x_ph = torch.from_numpy(obs_buf)
        a_ph = torch.from_numpy(act_buf)
        adv_ph = torch.from_numpy(adv_buf)
        ret_ph = torch.from_numpy(ret_buf)
        logp_old_ph = torch.from_numpy(logp_buf)

        net.train()
        # update actor
        for pi_iter in range(train_pi_iters):
            _, logp, _ = net.apply_actor(x_ph, a_ph)
            ratio = torch.exp(logp - logp_old_ph)
            min_adv = torch.where(adv_ph > 0, (1 + clip_ratio)*adv_ph, (1 - clip_ratio)*adv_ph)
            pi_loss = -torch.mean(torch.min(ratio * adv_ph, min_adv))

            if pi_iter == 0:
                pi_l_old = pi_loss.item()
                approx_ent = torch.mean(-logp).item()                  # a sample estimate for entropy, also easy to compute

            optimizer_actor.zero_grad()
            pi_loss.backward()
            average_gradients(optimizer_actor.param_groups)
            optimizer_actor.step()

            # compute kl
            kl = torch.mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
            kl = mpi_avg(kl.detach().numpy())
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.'% pi_iter)
                break
        logger.store(StopIter=pi_iter)
        clipped = (ratio > 1 + clip_ratio) | (ratio < 1 - clip_ratio)
        clipfrac = torch.mean(clipped.float()).item()

        # Value function learning
        for v_iter in range(train_v_iters):
            v_ph = net.apply_critic(x_ph)
            v_loss = criterion_mse(v_ph, ret_ph)

            if v_iter == 0:
                v_l_old = v_loss.item()

            optimizer_critic.zero_grad()
            v_loss.backward()
            average_gradients(optimizer_critic.param_groups)
            optimizer_critic.step()

        # Log changes from update
        pi_l_new = pi_loss.item()
        v_l_new = v_loss.item()
        logger.store(LossPi=pi_l_old, LossV=v_l_old, 
                     KL=kl, Entropy=approx_ent, ClipFrac=clipfrac,
                     DeltaLossPi=pi_l_new - pi_l_old,
                     DeltaLossV=v_l_new - v_l_old)

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            x_ph = torch.from_numpy(o[np.newaxis].astype(np.float32))
            with torch.no_grad():
                net.eval()
                a, _, logp_t = net.apply_actor(x_ph)
                v_t = net.apply_critic(x_ph)

            # save and log
            a = a.numpy()[0]
            v_t = v_t.data.numpy()
            logp_t = logp_t.data.numpy()
            ot = o.copy()
            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a)
            # buf.store(ot, a, r, v_t, logp_t)
            ep_ret += r
            ep_len += 1

            terminal = d or (ep_len == max_ep_len)
            if terminal or (t==local_steps_per_epoch-1):
                if not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
                # if trajectory didn't reach terminal state, bootstrap value target
                last_val = r if d else v_t
                # if d:
                    # last_val = 0
                # else:
                    # with torch.no_grad():
                        # net.eval()
                        # x_ph = torch.from_numpy(o[np.newaxis].astype(np.float32))
                        # v_t = net.apply_critic(x_ph)
                        # last_val = v_t.detach().numpy()

                buf.finish_path(last_val)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, net, None)

        # Perform VPG update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--env', type=str, default='CartPole-v0')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--pi_lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr', type=float, default=1e-3)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='ppo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup_pt.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    ppo(lambda : gym.make(args.env), actor_critic=core.Actor_Critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, pi_lr=args.pi_lr, vf_lr=args.vf_lr,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
