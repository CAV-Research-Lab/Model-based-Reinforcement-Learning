# import gym
# import spacecraftRobot
# import time
from spacerobot_env import SpaceRobotEnv
import numpy as np
np.set_printoptions(precision=3)
import matplotlib.pyplot as plt

# actions_true = np.load('opt_actions_true_dyn.npy', allow_pickle=True)
# actions = np.load('opt_actions.npy', allow_pickle=True)
# actions_lab = np.load('opt_actions_frm_labsys.npy', allow_pickle=True)
# actions = np.load('actions.npy', allow_pickle=True)
actions = np.load('results/2_forTarget_-2_-0.5_5/actions_floatbase_target2.npy', allow_pickle=True)
# actions = np.load('actions1.npy', allow_pickle=True)
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

