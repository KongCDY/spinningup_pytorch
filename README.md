# Welcome to Spinning Up in Deep RL! 

This is an educational resource produced by OpenAI that makes it easier to learn about deep reinforcement learning (deep RL).

For the unfamiliar: [reinforcement learning](https://en.wikipedia.org/wiki/Reinforcement_learning) (RL) is a machine learning approach for teaching agents how to solve tasks by trial and error. Deep RL refers to the combination of RL with [deep learning](http://ufldl.stanford.edu/tutorial/).

This module contains a variety of helpful resources, including:

- a short [introduction](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) to RL terminology, kinds of algorithms, and basic theory,
- an [essay](https://spinningup.openai.com/en/latest/spinningup/spinningup.html) about how to grow into an RL research role,
- a [curated list](https://spinningup.openai.com/en/latest/spinningup/keypapers.html) of important papers organized by topic,
- a well-documented [code repo](https://github.com/openai/spinningup) of short, standalone implementations of key algorithms,
- and a few [exercises](https://spinningup.openai.com/en/latest/spinningup/exercises.html) to serve as warm-ups.

Get started at [spinningup.openai.com](https://spinningup.openai.com)!

# Installation

## Installing OpenMPI

### Ubuntu

```bash
sudo apt-get update && sudo apt-get install libopenmpi-dev
```

### Mac OS X

Installation of system packages on Mac requires [Homebrew](https://brew.sh/). With Homebrew installed, run the follwing:

```bash
brew install openmpi
```

### Installing Spinning Up Pytorch

```bash
git clone https://github.com/KongCDY/spinningup_pytorch.git
cd spinningup_pytorch
pip install -e .
```

### Check Your Install

To see if you’ve successfully installed Spinning Up, try running PPO in the LunarLander-v2 environment with

```bash
python -m spinup_pt.run ppo --hid "[32,32]" --env LunarLander-v2 --exp_name installtest --gamma 0.999
```

This might run for around 10 minutes, and you can leave it going in the background while you continue reading through documentation. This won’t train the agent to completion, but will run it for long enough that you can see *some* learning progress when the results come in.

After it finishes training, watch a video of the trained policy with

```bash
python -m spinup_pt.run test_policy data/installtest/installtest_s0
```

And plot the results with

```bahs
python -m spinup_pt.run plot data/installtest/installtest_s0
```

Other options see official [webpage](<https://spinningup.openai.com/en/latest/user/installation.html>).
