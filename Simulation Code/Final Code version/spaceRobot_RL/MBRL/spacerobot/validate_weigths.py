"""
This implements a shooting trajectory optimization algorithm.
The closest known algorithm is perhaps MPPI and hence we stick to that terminology.
Uses a filtered action sequence to generate smooth motions.
"""

import numpy as np
import tensorflow as tf
import tensorflow.keras.optimizers as opt
import copy
import time
import mppi_polo_vecAsh
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from spacerobot_env import SpaceRobotEnv
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from pathlib import Path
import cProfile, pstats
import matplotlib.pyplot as plt
import types
from numba import jit, vectorize, cuda
from numba.experimental import jitclass
import os
from tensorflow.keras import regularizers

np.set_printoptions(precision=3, suppress=True)
print('This will work only with Tensorflow 2.x')
gpus = tf.config.experimental.list_physical_devices('GPU')

qpos=np.array([ 0.04 , -0.196,  4.877,  0.995, -0.026, -0.045,  0.084,  0.103,
        0.559,  0.668,  0.048, -0.17 , -0.14 ,  0.201])
qvel=np.array([0.00   ,  0.00   , 0.00   , 0.00   , 0.00,  0.00   , 0.00, 0.00,
        0.00,  0.00, 0.00, 0.00,  0.00])

# @jit
class MBRL:
    # dynamics=None and reward=None uses the env.step() to calculate the next_state and reward
    def __init__(self, dynamics=1, reward=1, horizon=500,
                 rollouts=500, epochs=150, model='DNN'):
        # self.env = gym.make(env_name)
        self.env = SpaceRobotEnv()
        self.env.reset()
        self.model = model
        self.env_cpy = copy.deepcopy(self.env)
        self.env_cpy.reset()
        self.target_loc = self.env.data.get_site_xpos('debrisSite')
        # target_loc = self.env.data.site_xpos[self.env.target_sid]  # another way to find self.target_loc
        self.dt = self.env.dt
        self.a_dim = self.env.action_space.shape[0]
        self.s_dim = self.env.observation_space.shape[0]
        self.a_low, self.a_high = self.env.action_space.low, self.env.action_space.high
        self.lr = 0.001
        self.horizon = horizon  # T
        self.rollouts = rollouts  # K
        self.storeData = None  # np.random.randn(self.bootstrapIter, self.s_dim + self.a_dim)
        self.dyn_opt = opt.Adam(learning_rate=self.lr)
        self.dyn = self.dyn_model(model=self.model)
        if self.model == 'DNN':
            self.dyn.load_weights('save_weights/trainedWeights500_floatbase_solved1')
            scalarX = Path("save_scalars/scalarXU_float_base_solved1.gz")
            self.load_scalars(scalarX, modelName='float_base_solved1.gz')
        else:
            self.dyn.load_weights('save_weights/trainedWeights500_floatbase_lstm2')
            scalarX = Path("save_scalars/scalarX_float_base_lstm.gz")
            self.load_scalars(scalarX, modelName='float_base_lstm.gz')

        self.dynamics = self.dynamics_batch
        self.reward = self.reward_batch

        self.mppi_gym = mppi_polo_vecAsh.MPPI(self.env, dynamics=self.dynamics, reward=self.reward,
                                              H=self.horizon,
                                              rollouts=self.rollouts,
                                              num_cpu=1,
                                              kappa=5,
                                              gamma=1,
                                              mean=np.zeros(self.env.action_space.shape[0]),
                                              filter_coefs=[np.ones(self.env.action_space.shape[0]), 0.25, 0.8, 0.0],
                                              default_act='mean',
                                              seed=2145
                                              )
    def load_scalars(self, scalarX, modelName='a'):
        if scalarX.is_file():
            self.scalarXU = joblib.load('save_scalars/scalarXU_'+ modelName)
            self.scalardX = joblib.load('save_scalars/scalardX_'+ modelName)
            self.fit = False
        else:
            self.scalarX = StandardScaler()  # StandardScaler()  RobustScaler(), MinMaxScaler(feature_range=(-1, 1))
            self.scalarU = StandardScaler()
            self.scalardX = StandardScaler()
            self.fit = True

    def run_mbrl(self, iter=200, train=False, render=False, retrain_after_iter=50):
        rewards, dataset, actions = mppi_polo_vecAsh.run_mppi(self.mppi_gym, self.env,
                                                            iter=iter, retrain_after_iter=retrain_after_iter,
                                                              render=render)
        self.animate(np.array(actions))

    def animate(self, actions):
        r = np.zeros(actions.shape[0])
        self.env.reset()
        for i, a in enumerate(actions):
            s, r[i], d, _ = self.env.step(a)
            self.env.render()
        plt.plot(r, 'r')
        plt.show()

    def reward_batch(self, x0, act):
        lam_a, lam_b = 0.001, 0
        if x0.ndim == 1:
            ss = 0
        else:
            ss = x0.shape[0]
        reward = np.zeros(ss)
        s0 = x0.copy()
        if ss:
            for i in range(ss):
                # self.env_cpy.reset()
                qp, qv = s0[i, :14], s0[i, 14:]
                self.env_cpy.set_env_state(qp, qv)
                endEff_loc = self.env_cpy.data.get_site_xpos('end_effector')
                # endEff_vel = self.env_cpy.data.get_site_xvel('end_effector')
                # base_linVel = self.env_cpy.data.get_site_xvelp('baseSite')
                # base_angVel = self.env_cpy.data.get_site_xvelr('baseSite')
                # act, base_linVel, base_angVel = np.squeeze(act), np.squeeze(base_linVel), np.squeeze(base_angVel)
                # rw_vel = np.dot(base_angVel, base_angVel) + np.dot(base_linVel, base_linVel)
                rel_vel = self.env_cpy.data.site_xvelp[self.env_cpy.hand_sid] - self.env_cpy.data.site_xvelp[
                    self.env_cpy.target_sid]  # relative velocity between end-effec & target
                reward[i] = -np.linalg.norm((self.target_loc - endEff_loc)) - lam_a * np.dot(act[i], act[i]) \
                            - np.dot(rel_vel, rel_vel)  #- lam_a * rw_vel
            # reward[i] += 1
        else:
            qp, qv = s0[:14], s0[14:]
            self.env_cpy.set_env_state(qp, qv)
            endEff_loc = self.env_cpy.data.get_site_xpos('end_effector')
            base_linVel = self.env_cpy.data.get_site_xvelp('baseSite')
            base_angVel = self.env_cpy.data.get_site_xvelr('baseSite')
            # act, base_linVel, base_angVel = np.squeeze(act), np.squeeze(base_linVel), np.squeeze(base_angVel)
            rw_vel = np.dot(base_angVel, base_angVel) + np.dot(base_linVel, base_linVel)
            reward = -np.linalg.norm((self.target_loc - endEff_loc)) - lam_a * np.dot(act, act) - lam_a * rw_vel
        return reward

    def dynamics_batch(self, state, perturbed_action):
        dt = 1
        u = np.clip(perturbed_action, self.a_low, self.a_high)
        next_state = state.copy()  # np.zeros_like(state)
        s1 = copy.deepcopy(state)
        stateAction = np.hstack((s1, u))
        stateAction = self.scalarXU.transform(stateAction)
        pred_dx = self.dyn(stateAction).numpy()
        state_residual = dt * self.scalardX.inverse_transform(pred_dx)
        next_state += state_residual
        return next_state

    def dyn_model(self, model='DNN'):
        ##############################################################
        """
        Layer Initializers
        https://keras.io/api/layers/initializers/
        Xavier or Glorot initializer solves vanishing gradient problem
        """
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=5 * 1000,
            decay_rate=1,
            staircase=False)
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
        initializer = tf.keras.initializers.GlorotNormal(seed=None)
        if model == 'DNN':  # just fully connected feedforward neural networks
            model = tf.keras.Sequential([
                # total_reward, dataset, actions = mppi_polo.run_mppi(self.mppi_gym, self.env, retrain_dynamics=None,
                #                                                     iter=iter, retrain_after_iter=100, render=True)
                tf.keras.Input(shape=(self.s_dim + self.a_dim,)),
                # tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(512, activation='relu', kernel_initializer=initializer,
                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                      bias_regularizer=regularizers.l2(1e-4),
                                      activity_regularizer=regularizers.l2(1e-5)
                                      ),
                tf.keras.layers.Dropout(0.02),
                tf.keras.layers.Dense(512, activation='relu', kernel_initializer=initializer,
                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                      bias_regularizer=regularizers.l2(1e-4),
                                      activity_regularizer=regularizers.l2(1e-5)
                                      ),
                # tf.keras.layers.Dropout(0.2),
                # tf.keras.layers.Dense(256, activation='relu'),
                # tf.keras.layers.Dropout(0.2),
                # tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dense(self.s_dim),
            ])
        else:  # LSTM
            ts_inputs = tf.keras.Input(shape=(self.bootstrapIter - 1, self.s_dim + self.a_dim))
            x = tf.keras.layers.LSTM(units=50)(ts_inputs)
            x = tf.keras.layers.Dropout(0.05)(x)
            x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=initializer)(x)
            x = tf.keras.layers.Dense(512, activation='relu', kernel_initializer=initializer)(x)
            outputs = tf.keras.layers.Dense(self.s_dim, activation='linear')(x)
            model = tf.keras.Model(inputs=ts_inputs, outputs=outputs)

        # model.compile(optimizer='adam', loss='kl_divergence', metrics=['accuracy'])
        # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        # """
        model.compile(optimizer=self.dyn_opt, loss='mse')
        # model.compile(optimizer=optimizer, loss='mse')
        model.summary()
        return model

    def resh(self, innp):
        return innp.reshape(*innp.shape, 1)


if __name__ == '__main__':

    dyn = 0
    render = 0
    retrain_after_iter = 100
    model = 'DNN'
    # model = 'LSTM'
    mbrl = MBRL(horizon=40, model=model, rollouts=80)
    mbrl.run_mbrl(train=1, iter=200, render=render)

