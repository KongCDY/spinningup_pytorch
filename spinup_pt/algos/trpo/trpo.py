import sys

import numpy as np
import gym
import time
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import init
import torch.optim as optim
from torch.nn.utils import parameters_to_vector, vector_to_parameters
import spinup_pt.algos.trpo.core as core
from spinup_pt.utils.logx import EpochLogger
from spinup_pt.utils.mpi_tools import mpi_fork, proc_id, mpi_avg,  mpi_statistics_scalar, num_procs
from spinup_pt.utils.mpi_torch import sync_all_params, average_gradients
from gym.spaces import Box, Discrete

EPS = 1e-8

class GAEBuffer:
    """
    A buffer for storing trajectories experienced by a TRPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, info_shapes, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.info_bufs = {k: np.zeros([size] + list(v), dtype=np.float32) for k,v in info_shapes.items()}
        self.sorted_info_keys = core.keys_as_sorted_list(self.info_bufs)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp, info):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        for i, k in enumerate(self.sorted_info_keys):
            self.info_bufs[k][self.ptr] = info[k]
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
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, 
                self.logp_buf] + core.values_as_sorted_list(self.info_bufs)


"""

Trust Region Policy Optimization 

(with support for Natural Policy Gradient)

"""
def trpo(env_fn, actor_critic=core.Actor_Critic, ac_kwargs=dict(), seed=0, 
         steps_per_epoch=4000, epochs=50, gamma=0.99, delta=0.01, vf_lr=1e-3,
         train_v_iters=80, damping_coeff=0.1, cg_iters=10, backtrack_iters=10, 
         backtrack_coeff=0.8, lam=0.97, max_ep_len=1000, logger_kwargs=dict(), 
         save_freq=10, algo='trpo'):
    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols 
            for state, ``x_ph``, and action, ``a_ph``, and returns the main 
            outputs from the agent's Tensorflow computation graph:

            ============  ================  ========================================
            Symbol        Shape             Description
            ============  ================  ========================================
            ``pi``        (batch, act_dim)  | Samples actions from policy given 
                                            | states.
            ``logp``      (batch,)          | Gives log probability, according to
                                            | the policy, of taking actions ``a_ph``
                                            | in states ``x_ph``.
            ``logp_pi``   (batch,)          | Gives log probability, according to
                                            | the policy, of the action sampled by
                                            | ``pi``.
            ``info``      N/A               | A dict of any intermediate quantities
                                            | (from calculating the policy or log 
                                            | probabilities) which are needed for
                                            | analytically computing KL divergence.
                                            | (eg sufficient statistics of the
                                            | distributions)
            ``info_phs``  N/A               | A dict of placeholders for old values
                                            | of the entries in ``info``.
            ``d_kl``      ()                | A symbol for computing the mean KL
                                            | divergence between the current policy
                                            | (``pi``) and the old policy (as 
                                            | specified by the inputs to 
                                            | ``info_phs``) over the batch of 
                                            | states given in ``x_ph``.
            ``v``         (batch,)          | Gives the value estimate for states
                                            | in ``x_ph``. (Critical: make sure 
                                            | to flatten this!)
            ============  ================  ========================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic 
            function you provided to TRPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        delta (float): KL-divergence limit for TRPO / NPG update. 
            (Should be small for stability. Values like 0.01, 0.05.)

        vf_lr (float): Learning rate for value function optimizer.

        train_v_iters (int): Number of gradient descent steps to take on 
            value function per epoch.

        damping_coeff (float): Artifact for numerical stability, should be 
            smallish. Adjusts Hessian-vector product calculation:
            
            .. math:: Hv \\rightarrow (\\alpha I + H)v

            where :math:`\\alpha` is the damping coefficient. 
            Probably don't play with this hyperparameter.

        cg_iters (int): Number of iterations of conjugate gradient to perform. 
            Increasing this will lead to a more accurate approximation
            to :math:`H^{-1} g`, and possibly slightly-improved performance,
            but at the cost of slowing things down. 

            Also probably don't play with this hyperparameter.

        backtrack_iters (int): Maximum number of steps allowed in the 
            backtracking line search. Since the line search usually doesn't 
            backtrack, and usually only steps back once when it does, this
            hyperparameter doesn't often matter.

        backtrack_coeff (float): How far back to step during backtracking line
            search. (Always between 0 and 1, usually above 0.5.)

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        algo: Either 'trpo' or 'npg': this code supports both, since they are 
            almost the same.

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
    if isinstance(env.action_space, Discrete):
        info_shapes = {'logp_all': [env.action_space.n]}
    else:
        info_shapes = {'mu': [env.action_space.shape[0]]}
    buf = GAEBuffer(obs_dim, act_dim, local_steps_per_epoch, info_shapes, gamma, lam)

    # make core of policy network
    net = actor_critic(obs_dim[0], **ac_kwargs)
    print(net)

    # loss function
    criterion_mse = nn.MSELoss()

    # optim
    optimizer_critic = optim.Adam(net.critic.parameters(), lr = vf_lr)

    # Sync params across processes
    sync_all_params(net.parameters())

    # Count variables
    # var_counts = tuple(core.count_vars(scope) for scope in ['pi', 'v'])
    # logger.log('\nNumber of parameters: \t pi: %d, \t v: %d\n'%var_counts)

    def cg(Ax, b):
        """
        Conjugate gradient algorithm
        (see https://en.wikipedia.org/wiki/Conjugate_gradient_method)
        """
        x = np.zeros_like(b)
        r = b.copy() # Note: should be 'b - Ax(x)', but for x=0, Ax(x)=0. Change if doing warm start.
        p = r.copy()
        r_dot_old = np.dot(r,r)
        for _ in range(cg_iters):
            z = Ax(p)
            alpha = r_dot_old / (np.dot(p, z) + EPS)
            x += alpha * p
            r -= alpha * z
            r_dot_new = np.dot(r,r)
            p = r + (r_dot_new / r_dot_old) * p
            r_dot_old = r_dot_new
        return x

    def update():
        net.train()
        inputs = [torch.from_numpy(x) for x in buf.get()]

        # Main outputs from computation graph, plus placeholders for old pdist (for KL)
        x_ph, a_ph, adv_ph, ret_ph, logp_old_ph = inputs[:5]
        _, logp, _, _, d_kl = net.apply_actor(x_ph, a_ph, old_logp_or_mu = inputs[-1])
        v = net.apply_critic(x_ph)

        # TRPO losses
        ratio = torch.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
        pi_l_old = -torch.mean(ratio * adv_ph)
        v_l_old = criterion_mse(v, ret_ph)

        # Prepare hessian func, gradient eval
        g = core.flat_grad(pi_l_old, net.actor.parameters(), retain_graph=True)
        g = mpi_avg(g.numpy())
        pi_l_old = mpi_avg(pi_l_old.item())

        def Hx(x):
            x = torch.from_numpy(x)
            hvp = core.hessian_vector_product(d_kl, net.actor, x)
            if damping_coeff > 0:
                hvp += damping_coeff * x
            return mpi_avg(hvp.detach().numpy())

        # Core calculations for TRPO or NPG
        x = cg(Hx, g)
        alpha = np.sqrt(2*delta/(np.dot(x, Hx(x))+EPS))
        old_params = parameters_to_vector(net.actor.parameters())

        x = torch.from_numpy(x)
        def set_and_eval(step):
            vector_to_parameters(old_params - alpha * x * step, net.actor.parameters())
            _, logp, _, _, d_kl = net.apply_actor(x_ph, a_ph, old_logp_or_mu = inputs[-1])
            ratio = torch.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
            pi_loss = -torch.mean(ratio * adv_ph)
            return mpi_avg(d_kl.item()), mpi_avg(pi_loss.item())

        if algo=='npg':
            # npg has no backtracking or hard kl constraint enforcement
            kl, pi_l_new = set_and_eval(step=1.)

        elif algo=='trpo':
            # trpo augments npg with backtracking line search, hard kl
            for j in range(backtrack_iters):
                kl, pi_l_new = set_and_eval(step=backtrack_coeff**j)
                if kl <= delta and pi_l_new <= pi_l_old:
                    logger.log('Accepting new params at step %d of line search.'%j)
                    logger.store(BacktrackIters=j)
                    break

                if j==backtrack_iters-1:
                    logger.log('Line search failed! Keeping old params.')
                    logger.store(BacktrackIters=j)
                    kl, pi_l_new = set_and_eval(step=0.)

	# Value function learning
        for _ in range(train_v_iters):
            v = net.apply_critic(x_ph)
            v_loss = criterion_mse(v, ret_ph)

            optimizer_critic.zero_grad()
            v_loss.backward()
            average_gradients(optimizer_critic.param_groups)
            optimizer_critic.step()

        # Log changes from update
        with torch.no_grad():
            net.eval()
            v = net.apply_critic(x_ph)
        v_l_new = criterion_mse(v, ret_ph)

        # Log changes from update
        logger.store(LossPi=pi_l_old, LossV=v_l_old.item(), KL=kl,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old).item())

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            x_ph = torch.from_numpy(o[np.newaxis].astype(np.float32))
            with torch.no_grad():
                net.eval()
                a, _, logp_t, info_t, _ = net.apply_actor(x_ph)
                v_t = net.apply_critic(x_ph)

            # save and log
            a = a.numpy()[0]
            v_t = v_t.data.numpy()
            logp_t = logp_t.data.numpy()
            ot = o.copy()
            buf.store(o, a, r, v_t, logp_t, info_t)
            logger.store(VVals=v_t)

            o, r, d, _ = env.step(a)
            # buf.store(ot, a, r, v_t, logp_t, info_t)
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

        # Perform TRPO or NPG update!
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
        logger.log_tabular('KL', average_only=True)
        if algo=='trpo':
            logger.log_tabular('BacktrackIters', average_only=True)
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
    parser.add_argument('--vf_lr', type=float, default=0.001)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='trpo')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup_pt.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    trpo(lambda : gym.make(args.env), actor_critic=core.Actor_Critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma, vf_lr=args.vf_lr,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
