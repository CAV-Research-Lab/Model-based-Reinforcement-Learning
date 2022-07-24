
import numpy as np
from utils import gather_paths_parallel
from xml.etree import ElementTree as ET
import copy
import os

class Trajectory:
    def __init__(self, env,env_p, H=32, seed=123):
        self.env, self.seed = env, seed
        self.n, self.m, self.H = env.observation_dim, env.action_dim, H
        self.env_p = env_p
        # following need to be populated by the trajectory optimization algorithm
        self.sol_state = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_obs = []

        self.env.reset_model(seed=self.seed)
        self.sol_state.append(self.env.get_env_state())
        self.sol_obs.append(self.env._get_obs())
        self.act_sequence = np.zeros((self.H, self.m))

    def update(self, paths):
        """
        This function should accept a set of trajectories
        and must update the solution trajectory
        """
        raise NotImplementedError

    def animate_rollout(self, t, act):
        """
        This function starts from time t in the solution trajectory
        and animates a given action sequence
        """
        self.env.set_env_state(self.sol_state[t])
        for k in range(act.shape[0]):
            try:
                self.env.env.env.mujoco_render_frames = True
            except AttributeError:
                self.env.render()
            self.env.set_env_state(self.sol_state[t+k])
            self.env.step(act[k])
            print(self.env.env_timestep)
            print(self.env.real_step)
        try:
            self.env.env.env.mujoco_render_frames = False
        except:
            pass

    def animate_result(self):
        self.env.reset()
        #self.env_p.set_env_state(self.sol_state[0])
        for k in range(len(self.sol_act)):
            self.env.mujoco_render_frames = True

            self.env.render()
            self.env.step(self.sol_act[k])
        #self.env.env.env.mujoco_render_frames = False

class MPPI(Trajectory):
    def __init__(self, env, env_p, mission, H, paths_per_cpu,
                 num_cpu=1,
                 kappa=1.0,
                 gamma=1.0,
                 mean=None,
                 filter_coefs=None,
                 default_act='repeat',
                 warmstart=True,
                 seed=123,
                 ):
        self.mission = mission
        self.env, self.seed = env, seed
        self.env_p = env_p
        self.n, self.m = env.observation_dim, env.action_dim
        self.H, self.paths_per_cpu, self.num_cpu = H, paths_per_cpu, num_cpu
        self.warmstart = warmstart

        self.mean, self.filter_coefs, self.kappa, self.gamma = mean, filter_coefs, kappa, gamma
        if mean is None:
            self.mean = np.zeros(self.m)
        if filter_coefs is None:
            self.filter_coefs = [np.ones(self.m), 1.0, 0.0, 0.0]
        self.default_act = default_act

        self.nVar = 7
        self.VarMin = 50
        self.VarMax = 6000
        self.MaxVelocity = 0.2*(self.VarMax - self.VarMin)
        self.MinVelocity = -1*self.MaxVelocity
        self.MaxIt = 200
        self.nPop =100
        self.w = 1
        self.wdamp = 0.9
        self.c1 = 2
        self.c2 = 2
        self.t1 = 0

        self.sol_state = []
        self.sol_state1 = []
        self.sol_act = []
        self.sol_reward = []
        self.sol_debris_site = []
        self.sol_obs = []
        self.sol_friction = []
        self.sigma = 200
        self.decay = 0.995
        self.best_reward = -1000.0

        self.env.reset()
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_obs.append(self.env.get_obs())
        self.act_sequence = np.ones((self.H, self.m)) * self.mean
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

    def advance_time(self, act_sequence=None):
        act_sequence = self.act_sequence if act_sequence is None else act_sequence
        # accept first action and step
        action = act_sequence[0].copy()
        state, r, _, _ = self.env.step(action)
        self.sol_act.append(action)
        self.sol_state.append(self.env.get_env_state().copy())
        self.sol_state1.append(state)
        self.sol_obs.append(self.env.get_obs())
        self.sol_reward.append(r)
        self.sol_debris_site.append(np.copy(self.env.data.get_site_xpos('end_effector')))
        name = 'space_robot_job/' + 'debris_site_' + str(self.mission) + '.npy'
        np.save(name, self.sol_debris_site)

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

    def do_rollouts(self, seed):
        paths = gather_paths_parallel(self.sol_state[-1],
                                      self.act_sequence,
                                      self.filter_coefs,
                                      seed,
                                      self.paths_per_cpu,
                                      self.num_cpu,
                                      )
        return paths

    def train_step(self, niter=1):
        t = len(self.sol_state) - 1
        for _ in range(niter):
            paths = self.do_rollouts(self.seed+t)
            self.update(paths)
        self.advance_time()
        if self.t1 < 200 and (self.mission == 2 or self.mission == 4):
            self.t1 += 1
            self.update_model()

    def save_xml(self,env):
        path = os.getcwd() + '\spaceRobot4mbrl_p.xml'
        b = env.model.get_xml()
        tree = ET.XML(b) 
        with open(path, "wb") as f: 
            f.write(ET.tostring(tree))

    def update_model(self):
        model_parameter = self.env_p.model.dof_damping[6:]
        model_parameter1 = copy.deepcopy(self.env_p.model.dof_damping)
        self.model_params = []
        paths = []
        self.sigma *= self.decay
        self.sigma = max(self.sigma, 30)
        for i in range(50) :
            self.model_params.append(self.generate_perturbed_parameters(model_parameter, filter_coefs=[self.sigma*np.array([1., 1., 1., 1., 1., 1., 1.]), 0.25, 0.8, 0.0]))
            self.env_p.reset()
            self.env_p.model.dof_damping[:] = [ 0.,  0.,  0.,  0.,  0.,  0.]+ list(self.model_params[-1])
            self.env_p.set_env_state(self.sol_state[0])
            rewards = 0
            actions = []
            for j in range(len(self.sol_state1)) :
                state, reward, done, info = self.env_p.step(self.sol_act[j])
                r = -2*np.linalg.norm(np.array(state) - np.array(self.sol_state1[j]))
                rewards += r

            path = dict(rewards=rewards,
                        model_param=self.model_params[-1])
            
            paths.append(path)

        num_traj = len(paths)
        act = np.array([paths[i]["model_param"] for i in range(num_traj)])
        R = np.array([paths[i]["rewards"] for i in range(num_traj)])
        max_indent_r = list(R).index(np.max(R)) 
        act_sequence = act[max_indent_r]
        self.sol_friction.append(act_sequence)
        name = 'space_robot_job/' + 'params_' + str(self.mission) + '.npy'
        np.save(name, self.sol_friction)

        #print(act_sequence.shape)
        self.bests= np.array([ 0.,  0.,  0.,  0.,  0.,  0.]+ list(act_sequence))
        self.best_params = 0.2*self.bests + 0.8*model_parameter1
        self.env_p.model.dof_damping[:] = np.clip(self.best_params, 0, 10000)
        self.save_xml(self.env_p)
        
    def generate_perturbed_parameters(self,base_parameters, filter_coefs):
        """
        Generate perturbed actions around a base action sequence
        """
        sigma, beta_0, beta_1, beta_2 = filter_coefs
        eps = np.random.normal(loc=0, scale=1.0, size=base_parameters.shape) * sigma
        for i in range(2, eps.shape[0]):
            eps[i] = beta_0*eps[i] + beta_1*eps[i-1] + beta_2*eps[i-2]
        return np.clip(base_parameters + eps,10,10000)
