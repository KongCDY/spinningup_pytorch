import gym
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v0')
args = parser.parse_args()

if __name__ == '__main__':
    env = gym.make(args.env)
    print('action_space is {}'.format(env.action_space))
    print('observation space is {}'.format(env.observation_space))
    for i_episode in range(20):
        observation = env.reset()
        for t in range(500):
            env.render()
            # print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
