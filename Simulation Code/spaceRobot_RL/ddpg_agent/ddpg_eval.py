import numpy as np
import gym
import tensorflow as tf
import matplotlib.pyplot as plt

from ddpg import DDPG
from pendulum import Pendulum
from actor_critic import Actor

if __name__ == '__main__':
    EPISODES = 5000
    GAMMA = 0.98
    ALPHA = 0.005
    EPSILON = 0.5
    EPSILON_DECAY = 0.1

    env = gym.make('Pendulum-v0')
    # env = Pendulum()
    a_dim = env.action_space.shape[0]
    layer_size = [32, 32]
    s_dim = env.observation_space.shape[0]
    ddpg = DDPG(env, s_dim=s_dim, a_dim=a_dim)
    # actor_trained.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')
    # x, y = np.random.rand(2, 3), np.array([0.8, 0.4])
    # actor_trained.train_on_batch(x, y)
    actor_trained = Actor(s_dim, a_dim).model()
    actor_trained.load_weights('training/target_actor_weights')
    # actor_untrained = ddpg.actor
    print('hi')

    def collect_data(act_net):
        a_all, states_all = [], []
        obs = env.reset()
        for t in range(1000):
            obs = np.squeeze(obs)
            if obs.ndim == 1:
                a = act_net(obs[None, :])
            else:
                a = act_net(obs)
            obs, _, done, _ = env.step(a)
            states_all.append(obs)
            a_all.append(a)
            env.render()
            if done:
                break
        states = np.squeeze(np.array(states_all))  # cos(theta), sin(theta), theta_dot
        a_all = np.squeeze(np.array(a_all))
        return states, a_all

    def plot_result(a_all, states, title='None'):
        fig, ax = plt.subplots(3, sharex=True)
        theta = np.arctan2(states[:, 1], states[:, 0])
        ax[0].set_ylabel('u')
        ax[0].plot(np.squeeze(a_all))
        ax[1].set_ylabel(u'$\\theta$')
        ax[1].plot(theta)
        # ax[1].plot(states[:, 0])
        ax[2].set_ylabel(u'$\omega$')
        ax[2].plot(states[:, 2])  # ang velocity
        fig.canvas.set_window_title(title)

    s_trained, a_trained = collect_data(actor_trained)
    # s_untrained, a_untrained, theta_untrained = collect_data(actor_untrained)

    plot_result(a_trained, s_trained, title='Trained_model')
    # plot_result(a_untrained, theta_untrained, s_untrained, title='UnTrained_model')

    plt.show()
    print('hi')
