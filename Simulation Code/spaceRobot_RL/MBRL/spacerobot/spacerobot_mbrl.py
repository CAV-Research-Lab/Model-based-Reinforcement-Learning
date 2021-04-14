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
# import mppi_polo_vecAsh
import mppi_polo
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from spacerobot_env import SpaceRobotEnv
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
from pathlib import Path
import cProfile, pstats
import types
from numba import jit
from numba.experimental import jitclass
np.set_printoptions(precision=3, suppress=True)
print('This will work only with Tensorflow 2.x')

# @jitclass
class MBRL:
    # dynamics=None and reward=None uses the env.step() to calculate the next_state and reward
    def __init__(self, dynamics=1, reward=1, env_name='SpaceRobot-v0', lr=0.001, horizon=500,
                 rollouts=500, epochs=150, bootstrap=False, bootstrapIter=300, bootstrap_rollouts=300):
        # self.env = gym.make(env_name)
        self.env = SpaceRobotEnv()
        self.env.reset()
        self.env_cpy = copy.deepcopy(self.env)
        self.env_cpy.reset()
        self.target_loc = self.env.data.get_site_xpos('debrisSite')

        self.dt = self.env.dt
        self.a_dim = self.env.action_space.shape[0]
        self.s_dim = self.env.observation_space.shape[0]
        self.a_low, self.a_high = self.env.action_space.low, self.env.action_space.high
        self.lr = lr
        self.early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
        self.reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

        if dynamics is None:
            self.dynamics = dynamics
        else:
            # self.dynamics = self.dynamics_true
            self.dynamics = self.dynamics_batch
            # self.dynamics = types.MethodType(dynamics_batch, self)
        if reward is None:
            self.reward = reward
        else:
            self.reward = self.reward_batch
            # self.reward = types.MethodType(reward_batch, self)

        self.horizon = horizon  # T
        self.rollouts = rollouts  # K
        self.epochs = epochs
        self.storeData = None  # np.random.randn(self.bootstrapIter, self.s_dim + self.a_dim)
        self.dyn = self.dyn_model(21, 14)
        # self.dyn = self.dyn_model(self.s_dim + self.a_dim, self.s_dim)
        self.dyn_opt = opt.Adam(learning_rate=self.lr)
        # self.dyn.load_weights('save_weights/trainedWeights500_1')
        # self.tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
        self.val_rollout = 50
        self.val_iter_per_rollout = 500
        self.storeValData = self.collectValdata(self.val_rollout, self.val_iter_per_rollout)

        scalarX = Path("save_scalars/scalarX.gz")
        # scalarU = Path("save_scalars/scalarU.gz")
        # scalardX = Path("save_scalars/scalardX.gz")
        if scalarX.is_file():
            self.scalarX = joblib.load('save_scalars/scalarX.gz')
            self.scalarU = joblib.load('save_scalars/scalarU.gz')
            self.scalardX = joblib.load('save_scalars/scalardX.gz')
            self.fit = False
        else:
            self.scalarX = MinMaxScaler(feature_range=(-1, 1))  # StandardScaler()  RobustScaler()
            self.scalarU = MinMaxScaler(feature_range=(-1, 1))
            self.scalardX = MinMaxScaler(feature_range=(-1, 1))
            self.fit = True

        if bootstrap:
            self.bootstrap_rollouts = bootstrap_rollouts
            self.bootstrapIter = bootstrapIter
            self.bootstrap(self.bootstrap_rollouts, self.bootstrapIter, storeData=True, train=True)
            print('Finished bootsrapping and training the bootsrapped dataset')
        self.X_val, self.Y_val = self.preprocess(self.storeValData, fit=self.fit)

        self.mppi_gym = mppi_polo.MPPI(self.env, dynamics=self.dynamics, reward=self.reward,
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

    def run_mbrl(self, iter=200, train=False, render=False, retrain_after_iter=50):
        if train:
            total_reward, dataset, actions = mppi_polo.run_mppi(self.mppi_gym, self.env, retrain_dynamics=self.train,
                                                                iter=iter, retrain_after_iter=retrain_after_iter, render=render)
            # total_reward, dataset, actions = mppi_polo.run_mppi(self.mppi_gym, self.env, retrain_dynamics=None,
            #                                                     iter=iter, retrain_after_iter=100, render=True)

            np.save('actions_cem.npy', np.array(actions), allow_pickle=True)
            self.save_weights(self.dyn, 'trainedWeights500_cem')
        else:
            total_reward, dataset, actions = mppi_polo.run_mppi(self.mppi_gym, self.env, iter=iter)
            np.save('actions_trueDyn.npy', np.array(actions), allow_pickle=True)

    # @jit(nopython=True)
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
                # base_linVel = self.env_cpy.data.get_site_xvelp('baseSite')
                # base_angVel = self.env_cpy.data.get_site_xvelr('baseSite')
                # act, base_linVel, base_angVel = np.squeeze(act), np.squeeze(base_linVel), np.squeeze(base_angVel)
                # rw_vel = np.dot(base_angVel, base_angVel) + np.dot(base_linVel, base_linVel)
                reward[i] = -np.linalg.norm((self.target_loc - endEff_loc)) - lam_a * np.dot(act[i], act[i]) #- lam_a * rw_vel
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

    # @jit(nopython=True)
    def dynamics_batch(self, state, perturbed_action):
        dt = 1
        u = np.clip(perturbed_action, self.a_low, self.a_high)
        next_state = state.copy()  # np.zeros_like(state)
        # ang_manip, vel_manip = state[:, 7:14], state[:, 20:27]
        # ang_manip = self.angle_normalize(ang_manip)
        # s1 = np.hstack((ang_manip, vel_manip))
        if state.ndim == 1:
            s1 = np.hstack((state[7:14], state[20:27]))
            s1_tr, act_tr = self.scalarX.transform(s1.reshape(1, -1)), self.scalarU.transform(u.reshape(1, -1))
            s2 = np.hstack((s1_tr, act_tr))
            pred_dx = np.squeeze(self.dyn(s2[None, :]).numpy())
            state_residual = dt * self.scalardX.inverse_transform(pred_dx.reshape(1, -1)).squeeze()
        else:
            s1 = np.hstack((state[:, 7:14], state[:, 20:27]))
            s1_tr, act_tr = self.scalarX.transform(s1), self.scalarU.transform(u)
            s2 = np.hstack((s1_tr, act_tr))
            # pred_dx = np.squeeze(self.dyn(s2.reshape(*s2.shape, 1)).numpy())  ## if model is lstm
            pred_dx = np.squeeze(self.dyn(s2[None, :]).numpy())
            state_residual = dt * self.scalardX.inverse_transform(pred_dx)

        # xx = np.hstack((ang_manip, vel_manip, u))
        # state_residual = self.dyn.predict(xx)
        if state.ndim == 1:
            next_state[7:14] += state_residual[:7]
            next_state[20:27] += state_residual[7:]
        else:
            next_state[:, 7:14] += state_residual[:, :7]
            next_state[:, 20:27] += state_residual[:, 7:]
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
        X = np.hstack((data[1:, 7:14], data[1:, 20:27]))
        U = data[1:, 27:]
        dX = np.diff(X, axis=0)  # state residual. makes the dimension less by one

        if fit:
            self.scalarX.fit(X)
            self.scalarU.fit(U)
            self.scalardX.fit(dX)
            joblib.dump(self.scalarX, 'save_scalars/scalarX.gz')
            joblib.dump(self.scalarU, 'save_scalars/scalarU.gz')
            joblib.dump(self.scalardX, 'save_scalars/scalardX.gz')
        normX = self.scalarX.transform(X)
        normU = self.scalarU.transform(U)
        normdX = self.scalardX.transform(dX)

        inputs = np.hstack((normX, normU))
        inputs = inputs[:-1]  # to make same dimension as dX
        outputs = normdX
        return inputs, outputs

    def dyn_model(self, in_dim, out_dim):
        ##############################################################
        lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
            0.001,
            decay_steps=5 * 1000,
            decay_rate=1,
            staircase=False)
        optimizer = tf.keras.optimizers.Adam(lr_schedule)
        initializer = tf.keras.initializers.GlorotNormal(seed=None)
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(in_dim, )),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer=initializer),
            # tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(512, activation='relu', kernel_initializer=initializer),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Dense(256, activation='relu'),
            # tf.keras.layers.Dropout(0.2),
            # tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(out_dim),
        ])

        #"""
        # Sequential Model
        #################
        # model.compile(optimizer='adam', loss='kl_divergence', metrics=['accuracy'])
        # model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        # """

        """
        # LSTM Model
        ###########
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=(in_dim, 1)))
        model.add(tf.keras.layers.Dropout(0.2))
        # Adding a second LSTM layer and Dropout layer
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        # Adding a third LSTM layer and Dropout layer
        model.add(tf.keras.layers.LSTM(units=50, return_sequences=True))
        model.add(tf.keras.layers.Dropout(0.2))
        # Adding a fourth LSTM layer and and Dropout layer
        model.add(tf.keras.layers.LSTM(units=50))
        model.add(tf.keras.layers.Dropout(0.2))
        # Adding the output layer
        # For Full connection layer we use dense
        # As the output is 1D so we use unit=1
        model.add(tf.keras.layers.Dense(units=out_dim))
        # model.compile(optimizer='adam', loss='mae')
        """
        model.compile(optimizer='Adam', loss='mse')
        # model.compile(optimizer=optimizer, loss='mse')

        return model

    def angle_normalize(self, x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def train(self, dataset, fit=False, model='STM', preprocessVal=False):
        """
        Trying to find the increment in states, f_(theta), from the equation
        s_{t+1} = s_t + dt * f_(theta)(s_t, a_t)

        states for a spacerobot: there is a free-floating base (passive joints) and a 7-DoF arm (active joints).
        base: (x, y, z, qx, qy, qz, qw) ; arm: (q1, q2, q3, q4, q5, q6, q7)
        and the corresponding velocities
        """

        # manip_joint_angles, manip_joint_vel, actions = dataset[:, 7:14], dataset[:, 20:27], dataset[:, 27:]
        # manip_joint_angles = self.angle_normalize(manip_joint_angles)
        # actions = np.clip(actions, self.a_low, self.a_high)
        # xu = np.hstack((manip_joint_angles, manip_joint_vel, actions))
        # dtheta_manip = manip_joint_angles[1:, :] - manip_joint_angles[:-1, :]
        # # dtheta_manip = angular_diff_batch(manip_joint_angles[1:, :], manip_joint_angles[:-1, :])
        # dtheta_dt_manip = manip_joint_vel[1:, :] - manip_joint_vel[:-1, :]
        # Y = np.hstack((dtheta_manip, dtheta_dt_manip))  # x' - x residual
        # xu = xu[:-1]  # make same size as Y
        tensorboard = TensorBoard(log_dir="logs/{}".format(time.time()))
        if self.storeData is not None:
            n = self.storeData.shape[0]
            newData = self.storeData[np.random.choice(n, n//3, replace=False), :]
            Data = np.vstack((newData, dataset))
        else:
            Data = dataset
        inputs, outputs = self.preprocess(Data, fit=fit)
        if preprocessVal:
            self.X_val, self.Y_val = self.preprocess(self.storeValData, fit=False)
        if model == 'LSTM':
            inputs = inputs.reshape(*inputs.shape, 1)
            outputs = outputs.reshape(*outputs.shape, 1)
        self.dyn.fit(
                    inputs,
                    outputs,
                    batch_size=400,
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
        # create scaler
        # fit and transform in one step
        # normalized = scaler.fit_transform(dataset)
        # nor_states, nor_actions = normalized[:, :s_dim], normalized[:, s_dim:]
        # Y = nor_states[1:, :] - nor_states[:-1, :]  # true_state_residual
        # xu = normalized[:-1]
        # fwd_dyn_nn.fit(xu, Y, epochs=100)

        # inverse transform
        # inverse = scaler.inverse_transform(normalized)

        # dataset[:, 7:14] = angle_normalize(dataset[:, 7:14])  # 7:14 are manip joint angles
        # dtheta_manip = angular_diff_batch(dataset[1:, 7:14], dataset[:-1, 7:14])
        # dtheta_dt_manip = dataset[1:, 20:27] - dataset[:-1, 20:27]
        # dtheta_base = dataset[1:, :7] - dataset[:-1, :7]
        # dtheta_dt_base = dataset[1:, 14:20] - dataset[:-1, 14:20]
        # Y = np.hstack((dtheta_base,  dtheta_manip, dtheta_dt_base, dtheta_dt_manip))  # x' - x residual
        # xu = dataset[:-1]  # make same size as Y
        # fwd_dyn_nn.fit(xu, Y, epochs=500)

    def bootstrap(self, n_rollouts, n_iter_per_rollout, storeData=False, train=False):
        # logger.info("bootstrapping with random action for %d actions", self.bootstrapIter)
        new_data = np.zeros((n_iter_per_rollout, self.s_dim + self.a_dim))
        # new_data = np.zeros((bootstrapIter, num_arm_states+a_dim))
        dataset = list()
        for k in range(n_rollouts):
            self.env_cpy.reset()
            for i in range(n_iter_per_rollout):
                # pre_action_state = self.env.state  # [num_base_states:]
                pre_action_state = self.env_cpy.state_vector()  # [num_base_states:]
                # pre_action_state = env_cpy.state_vector()[num_base_states:]
                action = np.random.uniform(low=self.a_low, high=self.a_high) * 0.6
                self.env_cpy.step(action)
                # env_cpy.render()
                new_data[i, :self.s_dim] = pre_action_state
                new_data[i, self.s_dim:] = action
            dataset.append(new_data)
        data = np.concatenate(dataset, axis=0)
        # np.save('data.npy', new_data, allow_pickle=True)
        if train:
            self.train(data, fit=self.fit, preprocessVal=True)
        if storeData:
            self.storeData = data
        # logger.info("bootstrapping finished")
        self.env_cpy.reset()
        return data

    def loss(self, data):
        """
        :param data: a batch of (state, action) concatenated
        :return: loss = (1/N)*(Y - f(X_i))**2 for a batch
        """
        N = data.shape[0]
        inputs, outputs = self.preprocess(data)  # inputs = normalized (state, action), outputs = normalized (s'-s)
        Ypred = self.dyn(inputs)
        loss_ = tf.math.reduce_mean(tf.math.square(Ypred - outputs))
        return loss_

    def collectValdata(self, val_num, iter):
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

    def check_model(self, dyn_model=None):
        s0 = self.env.reset()
        s1 = np.hstack((s0[7:14], s0[20:27]))
        print('starting_state:', s1)
        if dyn_model:
            dyn = dyn_model
        else:
            dyn = self.dyn
        for _ in range(100):
            act = self.env.action_space.sample()
            s2 = np.hstack((s1, act))
            obs_actual, _, _, _ = self.env.step(act)
            obs_pred = s1 + self.dt * np.squeeze(dyn(s2[None, :]).numpy())
            s1 = obs_pred

        obs_actual = np.hstack((obs_actual[7:14], obs_actual[20:27]))

        # obs = np.array([-1., 0., 0.03])  # [cos(theta), sin(theta), theta_dot] corresponding to theta = pi
        # self.env.state = [np.pi, 0.03]
        # print('action taken:', act)
        print('current_state:', obs_actual)
        print('Predicted state:', obs_pred)
        print('differece:', obs_actual - obs_pred)


if __name__ == '__main__':

    dyn = 0
    render = False
    retrain_after_iter = 50
    if dyn:
        bootstrap = 0
        train = 0
    else:
        bootstrap = 0
        train = 1
    if dyn:
        mbrl = MBRL(env_name='SpaceRobot-v0', lr=0.001, dynamics=None, reward=None,
                    horizon=20,
                    rollouts=30, epochs=150, bootstrapIter=3, bootstrap_rollouts=3
                    )  # to run using env.step()
    else:

        mbrl = MBRL(env_name='SpaceRobot-v0', lr=0.001, horizon=20,
                    rollouts=50, epochs=50, bootstrapIter=10, bootstrap_rollouts=10,
                    bootstrap=bootstrap)  # to run using dyn and rew
    # statement = "mbrl.run_mbrl(train=train, iter=50)"
    # cProfile.run(statement, filename="cpro.txt", sort=-1)
    profiler = cProfile.Profile()
    profiler.enable()
    start = time.time()
    mbrl.run_mbrl(train=train, iter=25, render=render, retrain_after_iter=retrain_after_iter)
    # print(time.time() - start)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats = pstats.Stats(profiler).sort_stats('tottime')
    stats.strip_dirs()
    # stats.print_stats()
    # stats.dump_stats('cpro_cython.prof')
    # stats.dump_stats('cpro_python.prof')
    # import snakeviz
    # snakeviz "cpro.prof"
    # mbrl.losses[['loss', 'val_loss']].plot()
