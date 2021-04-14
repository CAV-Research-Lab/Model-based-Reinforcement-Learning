import numpy as np
import argparse
import logging
import matplotlib.pyplot as plt
import gym
# from a2c_agent import A2CAgent
# from a2c_agent import Model
# from gym import spaces, envs
from ddpg_agent.ddpg import DDPG
import spacecraftRobot

parser = argparse.ArgumentParser()
parser.add_argument('-b', '--batch_size', type=int, default=64)
parser.add_argument('-n', '--num_updates', type=int, default=250)
parser.add_argument('-lr', '--learning_rate', type=float, default=7e-3)
parser.add_argument('-r', '--render_test', action='store_true', default=True)
parser.add_argument('-p', '--plot_results', action='store_true', default=True)

args = parser.parse_args()
env = gym.make('SpaceRobot-v0')

# model = Model(num_actions=7)
# agent = A2CAgent(model, args.learning_rate)

s_dim, a_dim = env.observation_space.shape[0], env.action_space.shape[0]
agent = DDPG(env=env, s_dim=s_dim, a_dim=a_dim)

logging.getLogger().setLevel(logging.INFO)

agent.train(render=1)
# rewards_history = agent.train(env, args.batch_size, args.num_updates)
# print("Finished training. Testing...")
# print("Total Episode Reward: %d out of 200" % agent.test(env, args.render_test))

# if args.plot_results:
#     plt.style.use('seaborn')
#     plt.plot(np.arange(0, len(rewards_history), 10), rewards_history[::10])
#     plt.xlabel('Episode')
#     plt.ylabel('Total Reward')
#     plt.show()


