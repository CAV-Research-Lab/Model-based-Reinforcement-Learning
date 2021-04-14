# import gym
# import spacecraftRobot
# import time
from spacerobot_env import SpaceRobotEnv
import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt

# actions = np.load('actions_server.npy', allow_pickle=True)
# actions = np.load('actions_trueDyn.npy', allow_pickle=True)
# actions = np.load('actions_floatbase_try_improve2.npy', allow_pickle=True)
actions = np.load('actions_floatbase_lstm.npy', allow_pickle=True)
"""
# The actions below are found out using MPC & env.step()
# dynamics_true = actions_trueDyn_batch.npy but very slow
# dynamics = None (fast)
actions = np.load('actions_trueDyn.npy', allow_pickle=True)
# actions = np.load('actions_trueDyn_batch.npy', allow_pickle=True)
"""
env = SpaceRobotEnv()
env.reset()
r = np.zeros(actions.shape[0])
for i, a in enumerate(actions):
    s, r[i], d, _ = env.step(a)
    print(env.sim.get_state())
    env.render()
    # time.sleep(0.1)

plt.plot(r, 'r')
plt.show()
print('done')

# R2 metric

