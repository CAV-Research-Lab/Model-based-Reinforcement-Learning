#!/usr/bin/python3
"""
Author: Mehran Raisi [mehranraisi74@gmail.com]
"""
import numpy as np
from gym import utils
from mjrl.envs import mujoco_env
from mujoco_py import MjViewer
import os
from math import atan2
import gym

class SpaceRobotEnvV0(mujoco_env.MujocoEnv, utils.EzPickle):
    sol_endEff_loc = []
    def __init__(self):
        self.seeding = False
        self.real_step = True
        self.env_timestep = 0

        self.hand_sid = 2
        self.target_sid = 0
        path1 = '/spaceRobot4mbrl.xml'
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        mujoco_env.MujocoEnv.__init__(self, curr_dir+path1, 5)
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
        self.action_dim = 7+6


    def __call__(self):
        pass

    def reset_model(self):
        qp = self.init_qpos.copy()
        qv = self.init_qvel.copy()
        # assert self.init_state.qpos == qpos and self.init_state.qvel == qvel
        self.set_state(qp, qv)
        # self.target_reset()
        self.env_timestep = 0

        self.sim.forward()
        return self.get_obs()

    def get_obs(self):
        return np.concatenate([self.sim.data.qpos.ravel(),
                               self.sim.data.qvel.ravel(),
                               ])

    def reward(self, act=None):
        lam_a, lam_b = 0.001, 0
        target_loc = self.data.get_site_xpos('debrisSite')
        endEff_loc = self.data.get_site_xpos('end_effector')
        endEff_ang = self.data.get_site_xmat('end_effector')
        endEff_linVel = self.data.get_site_xvelp('end_effector')
        base_linVel = self.data.get_site_xvelp('baseSite')
        base_angVel = self.data.get_site_xvelr('baseSite')
        base_ang = self.data.get_site_xmat('baseSite')
        base_loc = self.data.get_site_xpos('baseSite')
        act, base_linVel, base_angVel = np.squeeze(act), np.squeeze(base_linVel), np.squeeze(base_angVel)
        rw_vel = np.dot(base_linVel, base_linVel) + np.dot(base_angVel, base_angVel)
        rw_angl = np.linalg.norm(base_ang-np.eye((3)))
        reward = 1-4*np.linalg.norm((target_loc - endEff_loc))
        x = abs(base_loc[0])
        z = abs(base_loc[2])
        r = -5*(x+z)**2
        reward += r
        self.sol_endEff_loc.append(endEff_loc.copy())
        #print(endEff_loc)
        if np.linalg.norm((target_loc - endEff_loc)) < 5 :
            reward += 2 - 2*np.linalg.norm(self.Pitch_Roll_Yaw(endEff_ang) - np.array([0,0,np.pi]))
        return reward

    def done(self, reward):
        if np.abs(reward) < 1e-03:
            return True
        else:
            return False

    def step(self, act):
        self.do_simulation(act, self.frame_skip)

        reward = self.reward(act=act)
        done = self.done(reward)
        obs = self.get_obs()
        self.env_timestep += 1
        return obs, reward, done, {}

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def get_env_state1(self):
        target_pos = self.model.site_pos[self.target_sid].copy()
        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy(),
                    qa=self.data.qacc.copy(),
                    target_pos=target_pos, timestep=self.env_timestep)

    def get_env_state(self):
        qp = self.data.qpos.ravel().copy()
        qv = self.data.qvel.ravel().copy()

        return dict(qp=self.data.qpos.copy(), qv=self.data.qvel.copy())

    def set_env_state(self, state_dict):
        self.sim.reset()
        qp = state_dict['qp'].copy()
        qv = state_dict['qv'].copy()
        self.set_state(qp, qv)
        self.sim.forward()

    def set_env_state1(self, qp, qv):
        self.sim.reset()
        qp = qp.copy()
        qv = qv.copy()
        self.data.qpos[:] = qp
        self.data.qvel[:] = qv
        self.sim.forward()

    def target_reset(self):
        target_pos = self.model.site_pos[self.target_sid]
        a = 0.9
        target_pos[0] -= self.np_random.uniform(low=-a, high=a)
        target_pos[1] -= self.np_random.uniform(low=-a, high=a)
        target_pos[2] -= self.np_random.uniform(low=-.1, high=.1)
        self.model.site_pos[self.target_sid] = target_pos
        self.sim.forward()

    def mj_viewer_setup(self):
        self.viewer = MjViewer(self.sim)
        self.viewer.cam.azimuth = -45
        self.sim.forward()
        self.viewer.cam.distance = 1.0
    
    def Pitch_Roll_Yaw(self, Rotation_matrix) :
        Rotation_matrix = np.array(Rotation_matrix)
        if Rotation_matrix.shape[0] == 3:
            m = Rotation_matrix
            r31, r11, r21 = m[2][0], m[0][0], m[1][0]
            r32 , r33 = m[2][1] , m[2][2]
            beta = atan2(-r31, (r11**2 + r21**2)**0.5)
            if abs(np.cos(beta)) > 0.001 :
                gamma = atan2(r21/np.cos(beta), r11/np.cos(beta))
                alpha = atan2(r32/np.cos(beta), r33/np.cos(beta))
            else :
                beta = np.pi/2
                gamma = 0
                alpha = atan2(r12, r22)
        teta = np.array([alpha, beta, gamma]) + np.array([np.pi/2, np.pi/4, 1*np.pi/2])
        return teta