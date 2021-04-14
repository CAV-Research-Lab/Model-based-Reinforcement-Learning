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
        action = act_sequence[-1].copy()
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
                reward = self.reward_fn(s0_batch, act[:, t1, :])
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
