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
import mppi_polo
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from spacerobot_env import SpaceRobotEnv
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from pathlib import Path
from tensorflow.keras import regularizers

import cProfile, pstats
import matplotlib.pyplot as plt
import types
from numba import jit, vectorize, cuda
from numba.experimental import jitclass
import os
np.set_printoptions(precision=3, suppress=True)
print('This will work only with Tensorflow 2.x')
gpus = tf.config.experimental.list_physical_devices('GPU')

################################################################
# target1 = [-2, 0, 5.5]
# qpos=np.array([ 0.04 , -0.196,  4.877,  0.995, -0.026, -0.045,  0.084,  0.103,
#         0.559,  0.668,  0.048, -0.17 , -0.14 ,  0.201])
# qvel=np.array([0.00   ,  0.00   , 0.00   , 0.00   , 0.00,  0.00   , 0.00, 0.00,
#         0.00,  0.00, 0.00, 0.00,  0.00])
######################################################################
# target2 = [-2, -0.5, 5.]
# qpos=np.array([0.019, -0.09 ,  4.933,  0.998, -0.02 , -0.035,  0.042, -0.139,
#         0.237,  0.853, -0.217, -0.524,  0.232,  0.358])
# qvel=np.array([-3.643e-04,  2.810e-04, -1.476e-04, -1.504e-04, -3.055e-04,
#        -8.791e-05, -1.600e-03, -7.377e-04,  3.989e-03, -2.144e-03,
#        -1.266e-03,  8.000e-04,  4.289e-04])
#########################################################################
target3 = [0, 2.5, 5.]
qpos=np.array([-0.218, -0.389,  4.861,  0.91 , -0.013, -0.126,  0.394,  0.621,
        2.98 ,  1.61 ,  0.327, -2.106, -2.004,  0.806])
qvel=np.array([-6.487e-06,  1.672e-03,  7.721e-04,  9.462e-05, -9.215e-04,
        1.720e-03,  3.114e-03,  8.306e-03,  4.512e-03,  2.483e-03,
       -6.172e-03, -4.362e-04,  1.948e-03])
##########################################################################

# @jit
class MBRL:
    # dynamics=None and reward=None uses the env.step() to calculate the next_state and reward
    def __init__(self, dynamics=1, reward=1, env_name='SpaceRobot-v0', lr=0.001, horizon=500,
                 rollouts=500, epochs=150, bootstrap=False, bootstrapIter=300, bootstrap_rollouts=300, model='DNN',
                 startConfigIter=50, val_rollout=200, val_iter_per_rollout=40):
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
        self.lr = lr
        self.early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

        self.horizon = horizon  # T
        self.rollouts = rollouts  # K

        self.epochs = epochs

        scalarXU = Path("save_scalars/scalarXU_float_base_solved1.gz")
        self.load_scalars(scalarXU, modelName='float_base_solved1.gz')

        if dynamics is None:
            self.dynamics = dynamics
        else:
            self.dynamics = self.dynamics_batch

        if reward is None:
            self.reward = reward
        else:
            self.reward = self.reward_batch

        if bootstrap:
            self.dyn_opt = opt.Adam(learning_rate=self.lr)
            self.dyn = self.dyn_model(model=self.model)
            # self.dyn.load_weights('save_weights/trainedWeights500_floatbase_testDnn')
            self.dyn.load_weights('save_weights/trainedWeights500_floatbase_target2')
            self.startConfigIter = startConfigIter
            self.bootstrap_rollouts = bootstrap_rollouts
            self.bootstrapIter = bootstrapIter
            self.val_rollout = val_rollout
            self.val_iter_per_rollout = val_iter_per_rollout
            print('collecting data')
            self.randData = self.collectBootstrapData(self.bootstrap_rollouts, self.bootstrapIter)
            self.ValData = self.collectValData(self.val_rollout, self.val_iter_per_rollout)
            self.augData = self.bootstrapAugment(self.bootstrap_rollouts, self.bootstrapIter)
            print('starting preprocessing')
            self.X_rand, self.Y_rand = self.preprocess(self.randData, fit=self.fit)
            self.X_val, self.Y_val = self.preprocess(self.ValData)
            self.X_aug, self.Y_aug = self.preprocess(self.augData)
            XtrainBootstrap, YtrainBootstrap = np.vstack((self.X_rand, self.X_aug)), np.vstack((self.Y_rand, self.Y_aug))
            self.bootstrapDataTrain(XtrainBootstrap, YtrainBootstrap)
            print('Finished bootsrapping and training the bootsrapped dataset')

        self.mppi_gym = mppi_polo.MPPI(self.env, dynamics=self.dynamics, reward=self.reward,
                                              H=self.horizon, rollouts=self.rollouts, num_cpu=1, kappa=5,
                                              gamma=1, mean=np.zeros(self.env.action_space.shape[0]),
                                              filter_coefs=[np.ones(self.env.action_space.shape[0]), 0.25, 0.8, 0.0],
                                              default_act='mean',
                                              seed=2145
                                              )

    def load_scalars(self, scalarX, modelName='a'):
        if scalarX.is_file():
            self.scalarXU = joblib.load('save_scalars/scalarXU_'+ modelName)
            # self.scalarU = joblib.load('save_scalars/scalarU_'+ modelName)
            self.scalardX = joblib.load('save_scalars/scalardX_'+ modelName)
            self.fit = False
        else:
            self.scalarXU = StandardScaler()  # StandardScaler()  RobustScaler(), MinMaxScaler(feature_range=(-1, 1))
            # self.scalarU = StandardScaler()  # StandardScaler()  RobustScaler(), MinMaxScaler(feature_range=(-1, 1))
            self.scalardX = StandardScaler()  # StandardScaler()  RobustScaler(), MinMaxScaler(feature_range=(-1, 1))
            self.fit = True

    def run_mbrl(self, iter=200, train=False, render=False, retrain_after_iter=50):
        if train:
            rewards, dataset, actions = mppi_polo.run_mppi(self.mppi_gym, self.env, retrain_dynamics=self.train,
                                                                iter=iter, retrain_after_iter=retrain_after_iter, render=render)

            if self.model == 'DNN':
                np.save('actions_floatbase_target3.npy', np.array(actions), allow_pickle=True)
                self.save_weights(self.dyn, 'trainedWeights500_floatbase_target3')
            else:
                np.save('actions_floatbase_lstm2_1.npy', np.array(actions), allow_pickle=True)
                self.save_weights(self.dyn, 'trainedWeights500_floatbase_lstm2_1')
        else:
            rewards, dataset, actions = mppi_polo.run_mppi(self.mppi_gym, self.env, iter=iter)
            np.save('actions_trueDyn.npy', np.array(actions), allow_pickle=True)
            plt.plot(rewards, 'r')
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

    def dynamics_true(self, state, perturbed_action):
        dt = 1
        u = np.clip(perturbed_action, self.a_low, self.a_high)
        ss = u.shape[0]
        next_state = np.zeros_like(state)
        # if state.ndim == 1:
        #     s1 = np.hstack((state[7:14], state[20:27]))
        #     s1_tr, act_tr = self.scalarX.transform(s1.reshape(1, -1)), self.scalarU.transform(u.reshape(1, -1))
        #     s2 = np.hstack((s1_tr, act_tr))
        #     pred_dx = np.squeeze(self.dyn(s2[None, :]).numpy())
        #     state_residual = dt * self.scalardX.inverse_transform(pred_dx.reshape(1, -1)).squeeze()
        # else:
        self.env_cpy.reset()
        for i in range(ss):
            qp, qv = state[i, :14], state[i, 14:]
            self.env_cpy.set_env_state(qp, qv)
            next_state[i], _, _, _ = self.env_cpy.step(u[i])
            # self.env_cpy.reset()
        return next_state

    def preprocess(self, data, fit=False):
        if data.ndim == 3:
            # numSamples, numTimeSteps, numFeatures = data.shape
            X = data[:, :, :self.s_dim]
            U = data[:, :, self.s_dim:]
            dX = np.diff(X, axis=1)
            X = X[:, :-1, :].reshape(-1, self.s_dim)  # to make dimension of dX same as X
            U = U[:, :-1, :].reshape(-1, U.shape[2])
            stateAction = np.hstack((X, U))
            dX = dX.reshape(-1, dX.shape[2])
        else:
            X = data[:, :self.s_dim]
            U = data[:, self.s_dim:]
            dX = np.diff(X, axis=0)
            stateAction = np.hstack((X, U))[:-1]
        if fit:
            inputs = self.scalarXU.fit_transform(stateAction)
            outputs = self.scalardX.fit_transform(dX)
            joblib.dump(self.scalarXU, 'save_scalars/scalarXU_float_base_lstm.gz')
            joblib.dump(self.scalardX, 'save_scalars/scalardX_float_base_lstm.gz')
        else:
            inputs = self.scalarXU.transform(stateAction)  # fitting with scalar
            outputs = self.scalardX.transform(dX)  # fitting with scalar
        return inputs, outputs

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
             tf.keras.Input(shape=(self.s_dim+self.a_dim, )),
                # tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(512, activation='relu', kernel_initializer=initializer,
                                      kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
                                      bias_regularizer=regularizers.l2(1e-4),
                                      activity_regularizer=regularizers.l2(1e-5)
                                      ),
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
            ts_inputs = tf.keras.Input(shape=(self.bootstrapIter-1, self.s_dim+self.a_dim))
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

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def train(self, dataset):
        inputs, outputs = self.preprocess(dataset)
        self.fitModel(inputs, outputs)

    def fitModel(self, inputs, outputs):
        tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()) + 'float_base')
        self.dyn.fit(
                    inputs,
                    outputs,
                    batch_size=128,
                    epochs=self.epochs,
                    # We pass some validation for
                    # monitoring validation loss and metrics
                    # at the end of each epoch
                    validation_data=(self.X_val, self.Y_val),
                    callbacks=[self.early_stop, tensorboard],
                    # callbacks=[self.early_stop, self.tensorboard, self.reduce_lr],
                    )
        self.losses = pd.DataFrame(self.dyn.history.history)
        print('hi')

    def bootstrapAugment(self, n_rollouts, n_iter_per_rollout):
        new_data = np.zeros((n_rollouts, n_iter_per_rollout, self.s_dim + self.a_dim))
        self.env_cpy.reset()
        qp, qv = qpos, qvel
        for k in range(n_rollouts):
            self.env_cpy.set_env_state(qp, qv)
            for i in range(n_iter_per_rollout):
                pre_action_state = self.env_cpy.state_vector()  # [num_base_states:]
                action = np.random.uniform(low=self.a_low, high=self.a_high) + np.random.normal(loc=0, scale=0.01, size=7)
                self.env_cpy.step(action)
                new_data[k, i, :self.s_dim] = pre_action_state
                new_data[k, i, self.s_dim:] = action
        new_data = np.flip(new_data, axis=1)
        self.env_cpy.reset()
        return new_data

    def bootstrap(self, n_rollouts, n_iter_per_rollout):
        new_data = np.zeros((n_rollouts*self.startConfigIter, n_iter_per_rollout, self.s_dim + self.a_dim))
        self.env_cpy.reset()
        nr = 0
        for j in range(self.startConfigIter):
            qp, qv = self.env_cpy.get_env_state()
            for k in range(n_rollouts):
                self.env_cpy.set_env_state(qp, qv)
                az = nr*n_rollouts + k
                for i in range(n_iter_per_rollout):
                    pre_action_state = self.env_cpy.state_vector()  # [num_base_states:]
                    action = np.random.uniform(low=self.a_low, high=self.a_high) + np.random.normal(loc=0, scale=0.01, size=7)
                    self.env_cpy.step(action)
                    new_data[az, i, :self.s_dim] = pre_action_state
                    new_data[az, i, self.s_dim:] = action
            nr += 1
        self.env_cpy.reset()
        return new_data

    def bootstrapDataTrain(self, X, Y):
        self.fitModel(X, Y)

    def collectBootstrapData(self, val_num, iter):
        new_data = self.bootstrap(val_num, iter)
        print('Finished collecting Bootstrap dataset')
        return new_data

    def collectValData(self, val_num, iter):
        new_data = self.bootstrap(val_num, iter)
        print('Finished collecting validation dataset')
        return new_data

    def policy(self, observation):  # random policy
        return self.env.action_space.sample()
        # return np.random.uniform(env.action_space.low, env.action_space.high, env.action_space.shape)

    def save_weights(self, nn_network, save_name='final_weights'):
        nn_network.save_weights("save_weights/%s.h5" % save_name)
        # to save in other format
        nn_network.save_weights('save_weights/%s' % save_name, save_format='tf')
        print('Training completed and network weights saved')

    def load_weights(self, nn_network, name='pend_fwd_dyn_model'):
        network = nn_network(self.s_dim, self.a_dim).model()
        network.load_weights('save_weights/%s' % name)
        return network


if __name__ == '__main__':

    dyn = 1
    render = 0
    retrain_after_iter = 20
    model = 'DNN'
    # model = 'LSTM'
    if dyn:
        bootstrap = 0
        train = 0
    else:
        bootstrap = 1
        train = 1
    if dyn:
        mbrl = MBRL(env_name='SpaceRobot-v0', lr=0.001, dynamics=None, reward=None,
                    horizon=20, rollouts=30, epochs=150, bootstrapIter=3,
                    bootstrap_rollouts=3)  # to run using env.step()
    else:

        mbrl = MBRL(env_name='SpaceRobot-v0', lr=0.001, horizon=20, model=model,
                    rollouts=200, epochs=50, bootstrapIter=20, bootstrap_rollouts=500,
                    bootstrap=bootstrap, startConfigIter=10, val_rollout=300, val_iter_per_rollout=20)  # to run using dyn and rew

    mbrl.run_mbrl(train=train, iter=80, render=render, retrain_after_iter=retrain_after_iter)
