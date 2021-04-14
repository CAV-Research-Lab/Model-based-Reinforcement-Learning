import gym
import numpy as np
import time


class MPPI:
    """ MMPI according to algorithm 2 in Williams et al., 2017
        'Information Theoretic MPC for Model-Based Reinforcement Learning' """

    def __init__(self, env_name='Pendulum-v0', K=200, T=20, gamma=0.98, U=None, lambda_=1.0, noise_mu=0, noise_sigma=1, u_init=1, render=False,
        downward_start=True, save=True):
        self.env = gym.make(env_name)
        self.env.reset()
        self.render = render
        self.rollouts = K  # N_SAMPLES
        self.horizon = T  # TIMESTEPS
        self.gamma = gamma
        self.a_dim = self.env.action_space.shape[0]
        """ To set initial guess value of U """
        if not U:
            self.low = self.env.action_space.low
            self.high = self.env.action_space.high
            self.U = np.squeeze(np.random.uniform(low=self.low, high=self.high, size=(self.horizon, self.a_dim)))
        self.lambda_ = lambda_
        self.noise_mu = noise_mu
        self.noise_sigma = noise_sigma
        self.u_init = u_init
        self.cost_total = np.zeros(shape=self.rollouts)
        self.optimized_actions = []

        if env_name == 'Pendulum-v0' and downward_start:
            self.env.env.state = [np.pi, 1]

        """
        MPPI requires env.get_state() and env.set_state() function in addition to the gym env functions
        For envs like 'Pendulum-v0', there is an already existing function 'env.env.state' which can be 
        used to get and set state.
        """
        self.x_init = self.get_state()

        if save:
            np.save('initial_state', self.x_init, allow_pickle=True)
        
        self.noise = self.get_noise(k=self.horizon, t=self.rollouts, a_dim=self.a_dim)

        # if noise_gaussian:
        #     self.noise = np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(self.rollouts*self.horizon, self.a_dim))
        # else:
        #     self.noise = np.full(shape=(self.rollouts, self.horizon), fill_value=0.9)

    def set_state(self, state):
        """
           This method has to be implemented for envs other than pendulum
           Refer: 'https://github.com/aravindr93/trajopt/blob/master/trajopt/envs/reacher_env.py#L87'
           The above is for  MuJoCo envs
        """
        # if env_name == 'Pendulum-v0':
        self.env.env.state = state

    def get_state(self):
        """
        'https://github.com/aravindr93/trajopt/blob/master/trajopt/envs/reacher_env.py#L81'
        'https://github.com/ashBabu/spaceRobot_RL/blob/master/spacecraftRobot/envs/spaceRobot.py#L148'
        """
        # if env_name == 'Pendulum-v0':
        return self.env.env.state

    def get_noise(self, k=1, t=1, a_dim=1):
        return np.random.normal(loc=self.noise_mu, scale=self.noise_sigma, size=(t, k, a_dim))

    def _compute_total_cost(self, state, k):
        self.set_state(state)
        cost = 0
        for t in range(self.horizon):
            perturbed_actions = self.U[t] + self.noise[k, t]
            next_state, reward, done, info = self.env.step(perturbed_actions)
            cost += self.gamma**t * -reward
        return cost

    def _ensure_non_zero(self, cost, beta, factor):
        return np.exp(-factor * (cost - beta))

    def control(self, iter=200):
        for _ in range(iter):
            for k in range(self.rollouts):
                self.cost_total[k] = self._compute_total_cost(k=k, state=self.x_init)

            beta = np.min(self.cost_total)  # minimum cost of all trajectories
            cost_total_non_zero = self._ensure_non_zero(cost=self.cost_total, beta=beta, factor=1/self.lambda_)

            eta = np.sum(cost_total_non_zero)
            omega = 1/eta * cost_total_non_zero

            for t1 in range(self.horizon):
                for k1 in range(self.rollouts):
                    self.U[t1] += omega[k1] * self.noise[k1, t1]
            self.set_state(self.x_init)
            # print('state b4', self.get_state())  # start vertically down (if downward start=True) and see the changes
            if self.U[0].ndim == 0:
                s, r, _, _ = self.env.step([self.U[0]])
            else:
                s, r, _, _ = self.env.step(self.U[0])
            # print('state after', self.get_state())
            self.optimized_actions.append(self.U[0])
            self.x_init = self.get_state()
            self.U = np.roll(self.U, -1, axis=0)  # shift all elements to the left
            self.U[-1] = 1.  #
            act = np.clip(self.U[0], self.low, self.high)
            print("action taken: %.2f cost received: %.2f" % (act, -r))
            if self.render:
                self.env.render()
            self.cost_total[:] = 0
        return self.optimized_actions

    def animate_result(self, state, action):
        self.env.reset()
        self.set_state(state)
        for k in range(len(action)):
            # env.env.env.mujoco_render_frames = True
            self.env.render()
            self.env.step([action[k]])
            time.sleep(0.2)
        # env.env.env.mujoco_render_frames = False


if __name__ == "__main__":
    # env = 'MountainCarContinuous-v0'
    env = "Pendulum-v0"
    # env = "FetchReach-v1"
    # env = 'Reacher-v2'
    Horizon = 10  # T
    n_rollouts = 30  # K
    noise_mu, noise_sigma, lambda_ = 0, 10, 1
    mppi_gym = MPPI(env_name=env, K=n_rollouts, T=Horizon, lambda_=lambda_, noise_mu=noise_mu,
                    render=False, noise_sigma=noise_sigma)

    U = mppi_gym.control(iter=200)
    x0 = np.load('initial_state.npy', allow_pickle=True)
    print('final')
    mppi_gym.animate_result(x0, U)
    print('hi')