#!/usr/bin/python3
"""
Author: Ash Babu [shyamashi@gmail.com ; a.rajendrababu@surrey.ac.uk]
"""
import torch
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
# from mujoco_py import MjViewer, functions

""" 
    There are three sites defined as of now. 1. 'debrisSite': at the COM of debris, 2. 'baseSite': at the COM of base.
    3. 'end_effector': at the middle of the two fingers. 
    BELOW WORKS ONLY AFTER CALLING STEP() AT LEAST ONCE
    sim.data.site_xpos : gives the cartesian position of all the sites
    sim.data.site_xmat : gives the rotation matrix of all the sites
                            OR
    sim.data.get_site_xpos(name)

"""

# void mj_jac(const mjModel* m, const mjData* d,
#                       mjtNum* jacp, mjtNum* jacr, const mjtNum point[3], int body)
# while t<tlim:
#     sim.data.qpos[:] = pos[t,:]
#     sim.data.qvel[:] = vel[t,:]
#     sim.data.qacc[:] = acc[t,:]
#     functions.mj_inverse(sim.model, sim.data)
#     torque_est[t,:] = sim.data.qfrc_inverse
#     t += 1
#     env.sim.step()

"""
Base environment class for reinforcement learning. How to create such a class is explained in
'https://github.com/ashBabu/Utilities/wiki/Custom-MuJoCo-Environment-in-openai-gym'. 

Conservation of both linear and angular momentum is assumed. This makes env.reset() bring back the 
spacecraft-robotArm to exactly the same state at the start of the simulation

Here the action space is continuous or Box(9) which has 7 manipulator joint torques and 2 finger torques
"""


class SpaceRobotEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.seeding = False
        self.real_step = True
        self.env_timestep = 0

        self.hand_sid = 2
        self.target_sid = 0

        # fullpath = os.path.join(os.path.dirname(__file__), "assets", "spaceRobot.xml")
        fullpath = '/home/ash/Ash/repo/spaceRobot_RL/MBRL/spacerobot/spaceRobot4mbrl.xml'
        mujoco_env.MujocoEnv.__init__(self, fullpath, 2)
        utils.EzPickle.__init__(self)
        """
        MujocoEnv has the most important functions viz
            self.model = mujoco_py.load_model_from_path(fullpath)
            self.sim = mujoco_py.MjSim(self.model)
            self.data = self.sim.data
        """
        """
        H = np.zeros(self.sim.model.nv * self.sim.model.nv)
        L = functions.mj_fullM(self.sim.model, H, self.sim.data.qM)  # L = full Joint-space inertia matrix
        """
        self.on_goal = 0  # If the robot eef stays at the target for sometime, on_goal=1. Need to implement
        self.init_state = self.sim.get_state()

        self.target_sid = self.model.site_name2id("debrisSite")
        self.hand_sid = self.model.site_name2id("end_effector")
        """
        observation:
        free_base: (x,y,z,qx,qy,qz,qw | vx, vy, vz, wx, wy, wz) = 13
        7 rotary joints: (7 angles + 7 ang vel) = 14
        end_eff_pos(hand_sid): 3
        relative_dist between end_eff and target: 3
        """
        self.observation_dim = 39
        self.action_dim = 7
        print('Environment Successfully Created')

    def reset_model(self, seed=None):
        if seed is not None:
            self.seeding = True
            self.seed(seed)
        # assert self.init_state.qpos == qpos and self.init_state.qvel == qvel
        self.set_state(self.init_qpos, self.init_qvel)
        # self.target_reset()
        self.env_timestep = 0
        return self._get_obs()

    def _get_obs(self):
        """
        As per the spaceRobot.xml file, the first body is the debris. This has 3 position co-ordinates and 4
        quaternions included in the sim.data.qpos as the first 7 elements. Similarly sim.data.qvel contains
        3 linear velocity and 3 angular velocity components of the debris. This information is not required for
        learning algorithm since this is already provided as target_loc = data.get_site_xpos('debrisSite').

        Now the observation space consists of
        Positions:
        1. base (3 pos and 4 quaternions) for the free joint
        2. links (7 joint positions)
        3. endEff (2 linear joint positions)
        Total: 16
        Velocities:
        1. base (3 linear and 3 angular velocities) for the free joint
        2. links (7 joint velocities)
        3. endEff (2 linear joint velocities)
        Total: 15
        Grant Total: 31

        site_xvelp: position_jacobian * joint_vel
        site_xvelr: rotational_jacobian * joint_vel
        """
        return np.concatenate([self.sim.data.qpos.ravel(),
                               self.sim.data.qvel.ravel(),
                               # self.model.site_pos[self.hand_sid],
                               # self.data.site_xpos[self.hand_sid],
                               # self.data.site_xpos[self.hand_sid] - self.data.site_xpos[self.target_sid],
                               # self.data.site_xvelp[self.hand_sid] - self.data.site_xvelp[self.target_sid],
                               # self.data.site_xvelr[self.hand_sid] - self.data.site_xvelr[self.target_sid],
                               ])

    def reward(self, act=None):
        lam_a, lam_b = 0.001, 0
        target_loc = self.data.get_site_xpos('debrisSite')
        endEff_loc = self.data.get_site_xpos('end_effector')
        base_linVel = self.data.get_site_xvelp('baseSite')
        base_angVel = self.data.get_site_xvelr('baseSite')
        act, base_linVel, base_angVel = np.squeeze(act), np.squeeze(base_linVel), np.squeeze(base_angVel)
        rw_vel = np.dot(base_linVel, base_linVel) # + np.dot(base_angVel, base_angVel)
        # reward = -np.linalg.norm((target_loc - endEff_loc))
        reward = -np.linalg.norm((target_loc - endEff_loc)) - lam_a * np.dot(act, act) - rw_vel
        return reward

    def done(self, reward):
        if np.abs(reward) < 1e-03:
            return True
        else:
            return False

    def step(self, act):
        self.do_simulation(act, self.frame_skip)
        # self.data.qpos
        reward = self.reward(act=act)
        done = self.done(reward)
        obs = self._get_obs()
        self.env_timestep += 1
        return obs, reward, done, {}

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def get_env_state1(self):
        # target_pos1 = self.data.get_site_xpos('debrisSite').copy()
        target_pos = self.model.site_pos[self.target_sid].copy()
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    qa=self.data.qacc.copy(),
                    target_pos=target_pos, timestep=self.env_timestep)

    def get_env_state(self):
        return self.data.qpos.copy(), self.data.qvel.copy()

    def set_env_state(self, qp, qv):
        self.sim.reset()
        # qp = state_dict['qp'].copy()
        # qv = state_dict['qv'].copy()
        # qa = state_dict['qa'].copy()
        # target_pos = state_dict['target_pos']
        # self.env_timestep = state_dict['timestep']
        # self.model.site_pos[self.target_sid] = target_pos
        # self.data.site_xpos[self.target_sid] = target_pos
        # self.sim.forward()
        self.data.qpos[:] = qp
        self.data.qvel[:] = qv
        # self.data.qacc[:] = qa
        self.sim.forward()

    def target_reset(self):
        # target_pos = np.array([-1, -1, 5])
        # target_pos = self.data.get_site_xpos('debrisSite').copy()
        target_pos = self.model.site_pos[self.target_sid]
        a = 0.9
        target_pos[0] -= self.np_random.uniform(low=-a, high=a)
        target_pos[1] -= self.np_random.uniform(low=-a, high=a)
        target_pos[2] -= self.np_random.uniform(low=-.1, high=.1)
        self.model.site_pos[self.target_sid] = target_pos
        self.sim.forward()

    # def set_cam_position(self, viewer, cam_pos):
    #     for i in range(3):
    #         viewer.cam.lookat[i] = cam_pos[i]
    #     viewer.cam.distance = cam_pos[3]
    #     viewer.cam.elevation = cam_pos[4]
    #     viewer.cam.azimuth = cam_pos[5]
    #     viewer.cam.trackbodyid = -1

    # def close_gripper(self, left_gap=0, right_gap=0):
    #     pass

    # def get_idx_by_name(self, body_name):
    #     return self.model.body_names.index(six.b(body_name))


class SpaceRobotContinuousEnv(SpaceRobotEnv):

    def trigger_timed_events(self):
        if self.env_timestep % 50 == 0 and self.env_timestep > 0 and self.real_step is True:
            self.target_reset()


if __name__ == "__main__":
    import gym
    import spacecraftRobot

    env = gym.make('SpaceRobot-v0')
    # env = gym.make('SpaceRobotContinuous-v0')
    """
    #####################################
    env.model.body_id2name(2)
        Out[23]: 'spacecraft_base'
    env.model.body_id2name(0)
        Out[24]: 'world'
    env.model.body_id2name(1)
        Out[25]: 'sdebris'
    #######################################
    env.model.body_name2id('debrisSite')
        Out[31]: 0
    env.model.body_names  # for ur5
        Out[32]:  ('world', 'sdebris', 'spacecraft_base', 'spacecraft_link1', 'spacecraft_link2', 'spacecraft_link3',
        'spacecraft_link4', 'spacecraft_link5', 'spacecraft_link6', 'spacecraft_link7', 'leftfinger', 'rightfinger')

    env.model.body_pos
        Out[33]: array([[ 0.    ,  0.    ,  0.    ], [-2.    ,  3.    ,  5.5   ], [ 0.    ,  0.    ,  5.    ], .....])  # in local frame
    """

    obs1 = env.reset()
    print(env.data.site_xpos)  # site_xpos gives the position of site in world frame. works only after calling step()
    action = np.ones(7)
    obs, rew, done, _ = env.step(action)
    print('hi')
