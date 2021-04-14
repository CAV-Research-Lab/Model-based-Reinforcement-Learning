"""
This implements a shooting trajectory optimization algorithm.
The closest known algorithm is perhaps MPPI and hence we stick to that terminology.
Uses a filtered action sequence to generate smooth motions.
"""

import gym
import multiprocessing as mp
import numpy as np
import numpy.matlib as nm
import time
import copy
import tensorflow as tf
import json
import argparse
import pickle
import spacecraftRobot
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
np.set_printoptions(precision=3, suppress=True)


class MPPI:
    def __init__(self, env, H=16, paths_per_cpu=1, dynamics=None, reward=None,
                 num_cpu=1,
                 kappa=1.0,
                 gamma=1.0,
                 mean=None,
                 filter_coefs=None,
                 default_act='repeat',
                 warmstart=True,
                 seed=123,
                 ):
        self.env, self.seed = env, seed
        self.env_cpy = copy.deepcopy(env)
        self.env_cpy.reset()
        if not dynamics:
            self.step = True
        else:
            self.step = False
            self.fwd_dyn = dynamics
        if not reward:
            self.reward_fn = self.env.reward
        else:
            self.reward_fn = reward
        self.nx, self.nu = env.observation_space.shape[0], env.action_dim
        self.a_low, self.a_high = self.env.action_space.low, self.env.action_space.high
        self.H, self.paths_per_cpu, self.num_cpu = H, paths_per_cpu, num_cpu
        self.warmstart = warmstart

        self.mean, self.filter_coefs, self.kappa, self.gamma = mean, filter_coefs, kappa, gamma
        if mean is None:
            self.mean = np.zeros(self.nu)
        if filter_coefs is None:
            self.filter_coefs = [np.ones(self.nu), 1.0, 0.0, 0.0]
        self.default_act = default_act

        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []

        self.env.reset()
        # self.env.set_seed(seed)
        # self.env.reset(seed=seed)
        self.sol_state.append(self.env.get_env_state().copy())
        # self.sol_obs.append(self.env.env._get_obs())
        # self.sol_obs.append(self.env.get_obs())
        self.act_sequence = np.ones((self.H, self.nu)) * self.mean
        self.init_act_sequence = self.act_sequence.copy()

    def update(self, paths):
        num_traj = len(paths)
        act = np.array([paths[i]["actions"] for i in range(num_traj)])
        R = self.score_trajectory(paths)
        S = np.exp(self.kappa*(R-np.max(R)))

        # blend the action sequence
        weighted_seq = S*act.T
        act_sequence = np.sum(weighted_seq.T, axis=0)/(np.sum(S) + 1e-6)
        self.act_sequence = act_sequence
        print('action updated')

    def advance_time(self, rew, act_sequence=None):
        act_sequence = self.act_sequence if act_sequence is None else act_sequence
        # accept first action and step
        action = act_sequence[0].copy()
        # self.env.real_env_step(True)
        # _, r, _, _ = self.env.step(action)
        self.sol_act.append(action)
        self.sol_state.append(self.env.get_env_state().copy())
        # self.sol_obs.append(self.env.env._get_obs())
        # self.sol_obs.append(self.env.get_obs())
        self.sol_reward.append(rew)

        # get updated action sequence
        if self.warmstart:
            self.act_sequence[:-1] = act_sequence[1:]
            if self.default_act == 'repeat':
                self.act_sequence[-1] = self.act_sequence[-2]
            else:
                self.act_sequence[-1] = self.mean.copy()
        else:
            self.act_sequence = self.init_act_sequence.copy()

    def score_trajectory(self, paths):
        scores = np.zeros(len(paths))
        for i in range(len(paths)):
            scores[i] = 0.0
            for t in range(paths[i]["rewards"].shape[0]):
                scores[i] += (self.gamma**t)*paths[i]["rewards"][t]
        return scores

    def control(self, state):
        paths = self.gather_paths_parallel(state,
                                           self.act_sequence,
                                           self.filter_coefs,
                                           base_seed=2134,
                                           paths_per_cpu=self.paths_per_cpu,
                                           num_cpu=self.num_cpu,
                                           )
        self.update(paths)

    def do_env_rollout1(self, start_state, act_list):
        """
            1) Construct env with env_id and set it to start_state.
            2) Generate rollouts using act_list.
               act_list is a list with each element having size (H,m).
               Length of act_list is the number of desired rollouts.
        """
        paths = []
        H = act_list[0].shape[0]
        N = len(act_list)
        next_states = np.zeros((N, H, self.nx))  # N x H x nx
        act = np.array(act_list)  # dim of N x H x nu
        s0 = np.hstack((start_state['qp'], start_state['qv']))
        s0_batch = nm.repmat(s0, N, 1)
        next_states[:, 0, :] = s0_batch
        rewards = np.zeros((N, H))
        for t in range(H):
            reward = self.reward_fn(s0_batch, act[:, t, :])
            rewards[:, t] = reward
            next_state = self.fwd_dyn(s0_batch, act[:, t, :])
            next_states[:, t, :] = next_state
            s0_batch = next_state

        for j in range(N):
            path = dict(actions=act[j, :, :],
                        rewards=rewards[j, :],
                        )
            paths.append(path)
        return paths

    def do_env_rollout2(self, start_state, act_list):
        """
            1) Construct env with env_id and set it to start_state.
            2) Generate rollouts using act_list.
               act_list is a list with each element having size (H,m).
               Length of act_list is the number of desired rollouts.
        """

        # fwd_paths = self.do_env_rollout1(start_state, act_list)
        paths = []
        H = act_list[0].shape[0]
        N = len(act_list)
        s0 = np.hstack((start_state['qp'], start_state['qv']))
        self.env_cpy.reset_model()
        for i in range(N):
            self.env_cpy.set_env_state(start_state)
            act = []
            rewards = []
            for k in range(H):
                act.append(act_list[i][k])
                reward = self.reward_fn(s0, act[-1])
                rewards.append(reward)
                next_state = self.fwd_dyn(s0, act[-1])
                s0 = next_state

            path = dict(actions=np.array(act),
                        rewards=np.array(rewards),
                        )
            paths.append(path)

        # for i in range(len(paths)):
        #     print('actions:', paths[i]['actions'] - fwd_paths[i]['actions'])
        #     print('rewards:', paths[i]['rewards'] - fwd_paths[i]['rewards'])

        # else:
        #     next_states = np.zeros((N, H, self.nx))  # N x H x nx
        #     act = np.array(act_list)  # dim of N x H x nu
        #     s0 = np.hstack((start_state['qp'], start_state['qv']))
        #     s0_batch = nm.repmat(s0, N, 1)
        #     next_states[:, 0, :] = s0_batch
        #     rewards = np.zeros((N, H))
        #     nn = 1
        #     for t in range(H):
        #         reward = self.reward_fn(s0_batch, act[:, t, :])
        #         rewards[:, t] = reward
        #         next_state = self.fwd_dyn(s0_batch, act[:, t, :])
        #         next_states[:, t, :] = next_state
        #         s0_batch = next_state
        #         nn += 1
        #
        #     for j in range(N):
        #         path = dict(actions=act[j, :, :],
        #                     rewards=rewards[j, :],
        #                     )
        #         paths.append(path)
        return paths

    def do_env_rollout(self, start_state, act_list):
        """
            1) Construct env with env_id and set it to start_state.
            2) Generate rollouts using act_list.
               act_list is a list with each element having size (H,m).
               Length of act_list is the number of desired rollouts.
        """
        paths = []
        H = act_list[0].shape[0]  # Horizon
        N = len(act_list)  # = K rollouts
        if self.step:
            self.env_cpy.reset_model()
            for i in range(N):
                self.env_cpy.set_env_state(start_state)
                obs = []
                act = []
                rewards = []
                states = []

                for k in range(H):
                    # obs.append(self.env_cpy.env._get_obs())
                    act.append(act_list[i][k])
                    # states.append(self.env_cpy.get_env_state())
                    s, r, d, ifo = self.env_cpy.step(act[-1])
                    rewards.append(r)

                # path = dict(observations=np.array(obs),
                #             actions=np.array(act),
                #             rewards=np.array(rewards),
                #             states=states)
                path = dict(actions=np.array(act),
                            rewards=np.array(rewards),
                            )
                paths.append(path)
        else:
            next_states = np.zeros((N, H, self.nx))  # N x H x nx
            act = np.array(act_list)  # dim of N x H x nu
            s0 = np.hstack((start_state['qp'], start_state['qv']))
            s0_batch = nm.repmat(s0, N, 1)
            next_states[:, 0, :] = s0_batch
            rewards = np.zeros((N, H))
            nn = 1
            for t in range(H):
                # t_rew = time.time()
                reward = self.reward_fn(s0_batch, act[:, t, :])
                # print('rew time:', time.time() - t_rew)
                rewards[:, t] = reward
                # t_nxt = time.time()
                next_state = self.fwd_dyn(s0_batch, act[:, t, :])
                # print('next state time:', time.time() - t_nxt)
                next_states[:, t, :] = next_state
                s0_batch = next_state
                nn += 1

            for j in range(N):
                path = dict(actions=act[j, :, :],
                            rewards=rewards[j, :],
                            )
                paths.append(path)
        return paths

    def generate_perturbed_actions(self, base_act, filter_coefs):
        """
        Generate perturbed actions around a base action sequence
        """
        sigma, beta_0, beta_1, beta_2 = filter_coefs
        eps = np.random.normal(loc=0, scale=1.0, size=base_act.shape) * sigma
        for i in range(2, eps.shape[0]):
            eps[i] = beta_0 * eps[i] + beta_1 * eps[i - 1] + beta_2 * eps[i - 2]
        return base_act + eps

    def generate_paths(self, start_state, N, base_act, filter_coefs, base_seed):
        """
        first generate enough perturbed actions
        then do rollouts with generated actions
        set seed inside this function for multiprocessing
        """
        np.random.seed(base_seed)
        act_list = []
        for i in range(N):
            act = self.generate_perturbed_actions(base_act, filter_coefs)
            act_list.append(act)
        paths = self.do_env_rollout(start_state, act_list)
        # paths = self.do_env_rollout(start_state, act_list)
        return paths

    def generate_paths_star(self, args_list):
        return self.generate_paths(*args_list)

    def gather_paths_parallel(self, start_state, base_act, filter_coefs, base_seed, paths_per_cpu, num_cpu=None):
        num_cpu = mp.cpu_count() if num_cpu is None else num_cpu
        args_list = []
        for i in range(num_cpu):
            cpu_seed = base_seed + i * paths_per_cpu
            args_list_cpu = [start_state, paths_per_cpu, base_act, filter_coefs, cpu_seed]
            args_list.append(args_list_cpu)

        # do multiprocessing
        results = self._try_multiprocess(args_list, num_cpu, max_process_time=300, max_timeouts=4)
        paths = []
        for result in results:
            for path in result:
                paths.append(path)
        return paths

    def _try_multiprocess(self, args_list, num_cpu, max_process_time, max_timeouts):
        # Base case
        if max_timeouts == 0:
            return None

        if num_cpu == 1:
            results = [self.generate_paths_star(args_list[0])]  # dont invoke multiprocessing unnecessarily
        else:
            pool = mp.Pool(processes=num_cpu, maxtasksperchild=1)
            parallel_runs = [pool.apply_async(self.generate_paths_star,
                                              args=(args_list[i],)) for i in range(num_cpu)]
            try:
                results = [p.get(timeout=max_process_time) for p in parallel_runs]
            except Exception as e:
                print(str(e))
                print("Timeout Error raised... Trying again")
                pool.close()
                pool.terminate()
                pool.join()
                return self._try_multiprocess(args_list, num_cpu, max_process_time, max_timeouts - 1)
            pool.close()
            pool.terminate()
            pool.join()
        return results

    def animate_result(self):
        self.env.reset()
        # self.env.reset(self.seed)
        # self.env.env.state = self.sol_state[0]
        self.env.set_env_state(self.sol_state[0])
        for k in range(len(self.sol_act)):
            self.env.env.mujoco_render_frames = True
            # self.env.env.env.mujoco_render_frames = True
            self.env.render()
            self.env.step(self.sol_act[k])
        self.env.env.mujoco_render_frames = False
        # self.env.env.env.mujoco_render_frames = False


def run_mppi(mppi, env, retrain_dynamics=None, retrain_after_iter=50, iter=200, render=True):
    dataset = np.zeros((retrain_after_iter, mppi.nx + mppi.nu))
    # dataset = np.zeros((iter, mppi.nx + mppi.nu))
    total_reward = 0
    states, actions = [], []
    nn = 0
    for i in range(iter):
        # state = env.env.state.copy()
        state = env.get_env_state().copy()
        s0 = np.hstack((state['qp'], state['qv']))
        # print('state:', s0)
        command_start = time.perf_counter()
        mppi.control(state)
        action = mppi.act_sequence[0]
        # print('next_state_dyn_model:', mppi.fwd_dyn(s0, action))
        action = np.clip(action, mppi.a_low, mppi.a_high)
        actions.append(action)
        print('action:', action)
        # action = torch.zeros(7)
        elapsed = time.perf_counter() - command_start
        # print('Elaspsed time:', elapsed)
        s, r, _, _ = env.step(action)
        mppi.advance_time(rew=r)
        # states.append(env.sim.get_state())
        # print('MJstate:', env.sim.get_state())
        print(i)
        # time.sleep(.2)
        total_reward += r
        # logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
        if render:
            env.render()
        di = i % retrain_after_iter
        # if nn <= 7:
        if retrain_dynamics and di == 0 and i > 0:
            retrain_dynamics(dataset)
            nn += 1
                # don't have to clear dataset since it'll be overridden, but useful for debugging
        dataset[di, :mppi.nx] = env.state_vector()
        dataset[di, mppi.nx:] = action
    # np.save('actions.npy', np.array(actions), allow_pickle=True)
    # np.save('states.npy', np.array(states), allow_pickle=True)
    return total_reward, dataset, actions


if __name__ == '__main__':

    job_data = {
            'env_name'      : 'SpaceRobot-v0',
            'H_total'       : 100,
            'num_traj'      : 1,
            'seed'          : 12345,
            'num_iter'      : 1,
            'plan_horizon'  : 16,
            'paths_per_cpu' : 32,
            'num_cpu'       : 1,
            'filter'        : {'beta_0': 0.25, 'beta_1': 0.8, 'beta_2': 0.0},
            'kappa'         : 5.0,
            'gamma'         : 1.0,
            'default_act'   : 'mean',
            'visualize'     : True,
            'exp_notes'     : '7-DOF robot arm reaching various spatial goals in a continual, reset-free, '
                              'non-episodic setting.',
        }
    scaler = MinMaxScaler()
    OUT_DIR = 'spaceRobot_job_polo/'  # args.output
    # Unpack args and make files for easy access
    ENV_NAME = job_data['env_name']
    PICKLE_FILE = OUT_DIR + '/trajectories.pickle'
    EXP_FILE = OUT_DIR + '/job_data.json'
    SEED = job_data['seed']
    with open(EXP_FILE, 'w') as f:
        json.dump(job_data, f, indent=4)
    if 'visualize' in job_data.keys():
        VIZ = job_data['visualize']
    else:
        VIZ = False

    def trigger_tqdm(inp, viz=False):
        if viz:
            return tqdm(inp)
        else:
            return inp

    # =======================================
    # Train loop
    en = gym.make(ENV_NAME)
    en.reset()
    en_cpy = copy.deepcopy(en)
    en_cpy.reset()
    target_loc = en_cpy.env.data.get_site_xpos('debrisSite')
    mean = np.zeros(en.action_dim)
    sigma = 1.0 * np.ones(en.action_dim)
    filter_coefs = [sigma, job_data['filter']['beta_0'], job_data['filter']['beta_1'], job_data['filter']['beta_2']]
    trajectories = []
    a_low = en.action_space.low
    a_high = en.action_space.high
    s_dim = en.observation_space.shape[0]
    a_dim = en.action_space.shape[0]

    def preprocess(data):
        X = np.hstack((data[:, 7:14], data[:, 20:27]))
        U = data[:, 27:]
        dX = np.diff(X, axis=0)  # state residual

        scalarX = StandardScaler()  # MinMaxScaler(feature_range=(-1,1))#StandardScaler()# RobustScaler()
        scalarU = MinMaxScaler(feature_range=(-1, 1))
        scalardX = MinMaxScaler(feature_range=(-1, 1))

        scalarX.fit(X)
        scalarU.fit(U)
        scalardX.fit(dX)

        normX = scalarX.transform(X)
        normU = scalarU.transform(U)
        normdX = scalardX.transform(dX)

        inputs = np.hstack((normX, normU))
        inputs = inputs[:-1]
        outputs = normdX

    def reward(x0, act):
        lam_a, lam_b = 0.001, 0
        en_cpy.env.set_env_state(x0)
        target_loc = en_cpy.env.data.get_site_xpos('debrisSite')
        endEff_loc = en_cpy.env.data.get_site_xpos('end_effector')
        base_linVel = en_cpy.env.data.get_site_xvelp('baseSite')
        base_angVel = en_cpy.env.data.get_site_xvelr('baseSite')
        act, base_linVel, base_angVel = np.squeeze(act), np.squeeze(base_linVel), np.squeeze(base_angVel)
        rw_vel = np.dot(base_angVel, base_angVel) + np.dot(base_linVel, base_linVel)
        return -np.linalg.norm((target_loc - endEff_loc)) - lam_a * np.dot(act, act) - lam_a * rw_vel

    def angle_normalize(x):
        return ((x + np.pi) % (2 * np.pi)) - np.pi

    def angular_diff_batch(a, b):
        """Angle difference from b to a (a - b)"""
        d = a - b
        d[d > np.pi] -= 2 * np.pi
        d[d < -np.pi] += 2 * np.pi
        return d

    def dyn_model(in_dim, out_dim):
        model = tf.keras.Sequential([
            tf.keras.Input(shape=(in_dim, )),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(out_dim),
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
        return model

    fwd_dyn_nn = dyn_model(21, 14)
    # fwd_dyn_nn = dyn_model(s_dim + a_dim, s_dim)

    def reward_batch(x0, act):
        lam_a, lam_b = 0.001, 0
        ss = x0.shape[0]
        reward = np.zeros(ss)
        s0 = x0.copy()
        state = dict()
        state['qp'], state['qv'], state['timestep'] = np.zeros(14), np.zeros(13), 0
        for i in range(ss):
            en_cpy.reset()
            state['qp'], state['qv'] = s0[i, :14], s0[i, 14:]
            en_cpy.env.set_env_state(state)
            endEff_loc = en_cpy.env.data.get_site_xpos('end_effector')
            base_linVel = en_cpy.env.data.get_site_xvelp('baseSite')
            base_angVel = en_cpy.env.data.get_site_xvelr('baseSite')
            # act, base_linVel, base_angVel = np.squeeze(act), np.squeeze(base_linVel), np.squeeze(base_angVel)
            rw_vel = np.dot(base_angVel, base_angVel) + np.dot(base_linVel, base_linVel)
            reward[i] = -np.linalg.norm((target_loc - endEff_loc)) - lam_a * np.dot(act[i], act[i]) - lam_a * rw_vel
        return reward

    def dynamics_batch(state, perturbed_action):
        u = np.clip(perturbed_action, a_low, a_high)
        next_state = state.copy()  # np.zeros_like(state)
        ang_manip, vel_manip = state[:, 7:14], state[:, 20:27]
        ang_manip = angle_normalize(ang_manip)
        xx = np.hstack((ang_manip, vel_manip, u))
        state_residual = fwd_dyn_nn.predict(xx)
        next_state[:, 7:14] += state_residual[:, :7]
        next_state[:, 20:27] += state_residual[:, 7:]
        return next_state

    def dynamics(state, perturbed_action):
        u = np.clip(perturbed_action, a_low, a_high)
        next_state = np.zeros_like(state)
        ang_manip, vel_manip = state[7:14], state[20:27]
        ang_manip = angle_normalize(ang_manip)
        xx = np.hstack((ang_manip, vel_manip, u))
        state_residual = fwd_dyn_nn.predict(xx.reshape(1, -1))
        # state_residual = scaler.inverse_transform(fwd_dyn_nn(xx.reshape(1, -1)))
        # output dtheta directly so can just add
        next_state[7:14], next_state[20:27] = state_residual[0][:7], state_residual[0][7:]
        # next_state = state + state_residual
        return next_state

    def train(dataset):
        """
        Trying to find the increment in states, f_(theta), from the equation
        s_{t+1} = s_t + dt * f_(theta)(s_t, a_t)

        states for a spacerobot: there is a free-floating base (passive joints) and a 7-DoF arm (active joints).
        base: (x, y, z, qx, qy, qz, qw) ; arm: (q1, q2, q3, q4, q5, q6, q7)
        and the corresponding velocities
        """


        """ 
        # Method 1:
        manip_joint_angles, manip_joint_vel, actions = dataset[:, 7:14], dataset[:, 20:27], dataset[:, 27:]
        manip_joint_angles = angle_normalize(manip_joint_angles)
        actions = np.clip(actions, a_low, a_high)
        xu = np.hstack((manip_joint_angles, manip_joint_vel, actions))
        dtheta_manip = manip_joint_angles[1:, :] - manip_joint_angles[:-1, :]
        # dtheta_manip = angular_diff_batch(manip_joint_angles[1:, :], manip_joint_angles[:-1, :])
        dtheta_dt_manip = manip_joint_vel[1:, :] - manip_joint_vel[:-1, :]
        Y = np.hstack((dtheta_manip, dtheta_dt_manip))  # x' - x residual
        xu = xu[:-1]  # make same size as Y
        fwd_dyn_nn.fit(xu, Y, epochs=100)
        
        """

        """
        # Method 2: Normalization
        # create scaler
        # fit and transform in one step
        # normalized = scaler.fit_transform(dataset)
        # nor_states, nor_actions = normalized[:, :s_dim], normalized[:, s_dim:]
        # Y = nor_states[1:, :] - nor_states[:-1, :]  # true_state_residual
        # xu = normalized[:-1]
        # fwd_dyn_nn.fit(xu, Y, epochs=100)
        """

        # """
        # Method 3: Standardization
        manip_joint_angles, manip_joint_vel, actions = dataset[:, 7:14], dataset[:, 20:27], dataset[:, 27:]
        manip_joint_angles = angle_normalize(manip_joint_angles)
        actions = np.clip(actions, a_low, a_high)
        mean_angles = np.mean(manip_joint_angles, axis=0)
        std_angles = np.std(manip_joint_angles, axis=0)
        angles_normalized = (manip_joint_angles - mean_angles)/std_angles

        mean_vel = np.mean(manip_joint_vel, axis=0)
        std_vel = np.std(manip_joint_vel, axis=0)
        vel_normalized = (manip_joint_vel - mean_vel) / std_vel
        xu = np.hstack((angles_normalized, vel_normalized, actions))
        dtheta = angles_normalized[1:,:] - angles_normalized[:-1, :]
        dtheta_dt = vel_normalized[1:,:] - vel_normalized[:-1, :]
        Y = np.hstack((dtheta, dtheta_dt))
        xu = xu[:-1]
        fwd_dyn_nn.fit(xu, Y, epochs=100)
        # """
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

    def bootstrap(bootstrapIter):
        # logger.info("bootstrapping with random action for %d actions", self.bootstrapIter)
        new_data = np.zeros((bootstrapIter, s_dim+a_dim))
        # new_data = np.zeros((bootstrapIter, num_arm_states+a_dim))
        for i in range(bootstrapIter):
            pre_action_state = en.state_vector()  # [num_base_states:]
            # pre_action_state = en.en.state_vector()[num_base_states:]
            action = np.random.uniform(low=a_low, high=a_high) * 0.6
            en.step(action)
            # en.render()
            new_data[i, :s_dim] = pre_action_state
            new_data[i, s_dim:] = action
        train(new_data)
        # logger.info("bootstrapping finished")
        en.reset()

    def normalize(data, mean, std):
        return (data - mean) / (std + 1e-10)

    def denormalize(data, mean, std):
        return data * (std + 1e-10) + mean

    dynamics_given = False

    if dynamics_given:
        agent = MPPI(en,
                      H=job_data['plan_horizon'],
                      paths_per_cpu=job_data['paths_per_cpu'],
                      num_cpu=job_data['num_cpu'],
                      kappa=job_data['kappa'],
                      gamma=job_data['gamma'],
                      mean=mean,
                      filter_coefs=filter_coefs,
                      default_act=job_data['default_act'],
                      seed=2145
                      )
        run_mppi(agent, en, iter=200, retrain_after_iter=20, render=True)
    else:
        bootstrap(1500)
        agent = MPPI(en, dynamics=dynamics_batch, reward=reward_batch,
                     H=job_data['plan_horizon'],
                     paths_per_cpu=job_data['paths_per_cpu'],
                     num_cpu=job_data['num_cpu'],
                     kappa=job_data['kappa'],
                     gamma=job_data['gamma'],
                     mean=mean,
                     filter_coefs=filter_coefs,
                     default_act=job_data['default_act'],
                     seed=2145
                     )
        run_mppi(agent, en, train, iter=200, retrain_after_iter=20, render=True)
    # run_mppi(agent1, en)
    # opt_actions = np.array(agent.sol_act)
    # np.save('opt_actions.npy', opt_actions, allow_pickle=True)




    #
    # ts = time.time()
    # for i in range(job_data['num_traj']):
    #     start_time = time.time()
    #     print("Currently optimizing trajectory : %i" % i)
    #     seed = job_data['seed'] + i * 12345
    #     en.reset()
    #     # en.reset(seed=seed)
    #
    #     agent = MPPI(en,
    #                  H=job_data['plan_horizon'],
    #                  paths_per_cpu=job_data['paths_per_cpu'],
    #                  num_cpu=job_data['num_cpu'],
    #                  kappa=job_data['kappa'],
    #                  gamma=job_data['gamma'],
    #                  mean=mean,
    #                  filter_coefs=filter_coefs,
    #                  default_act=job_data['default_act'],
    #                  seed=seed
    #                  )
    #
    #     for t in trigger_tqdm(range(job_data['H_total']), VIZ):
    #         agent.control(job_data['num_iter'])
    #         # agent.control(job_data['num_iter'])
    #
    #     end_time = time.time()
    #     print("Trajectory reward = %f" % np.sum(agent.sol_reward))
    #     print("Optimization time for this trajectory = %f" % (end_time - start_time))
    #     trajectories.append(agent)
    #     pickle.dump(trajectories, open(PICKLE_FILE, 'wb'))
    #
    # print("Time for trajectory optimization = %f seconds" % (time.time() - ts))
    # pickle.dump(trajectories, open(PICKLE_FILE, 'wb'))
    #
    # if VIZ:
    #     _ = input("Press enter to display optimized trajectory (will be played 10 times) : ")
    #     for i in range(10):
    #         [traj.animate_result() for traj in trajectories]

