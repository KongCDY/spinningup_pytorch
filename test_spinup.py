from spinup import vpg
import tensorflow as tf
import gym
import ipdb

env_fn = lambda : gym.make('CartPole-v0')
ac_kwargs = dict(hidden_sizes=[32,32], activation=tf.nn.relu)
logger_kwargs = dict(output_dir='./output_dir', exp_name='experiment_name')

ipdb.set_trace()
vpg(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=4000, epochs=50, logger_kwargs=logger_kwargs)
