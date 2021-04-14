"""
A fast Implementation of Deep Deterministic Policy Gradient in Tensorflow 2.

Original Paper:
    http://arxiv.org/pdf/1509.02971v2.pdf

Author: Ash Babu
credits:  Arthur Bouton, https://github.com/Bouty92
          https://github.com/keras-team/keras-io/blob/master/examples/rl/ddpg_pendulum.py
"""

import numpy as np
import gym
import tensorflow as tf
from looptools import Loop_handler, Monitor
from replaybuffer import ReplayBuffer
from actor_critic import Actor, Critic
from ounoise import OUNoise
import matplotlib.pyplot as plt


class DDPG:
    def __init__(self, env=gym.make('Pendulum-v0'), s_dim=2, a_dim=1, gamma=0.99, episodes=100, tau=0.001,
                 buffer_size=1e06, minibatch_size=64, actor_lr=0.001, critic_lr=0.001, save_name='final_weights',
                 render=False):
        self.save_name = save_name
        self.render = render
        self.env = env
        self.upper_bound = env.action_space.high[0]
        self.lower_bound = env.action_space.low[0]
        self.EPISODES = episodes
        self.MAX_TIME_STEPS = 200
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.GAMMA = gamma
        self.TAU = tau
        self.buffer_size = buffer_size
        self.minibatch_size = minibatch_size
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.ou_noise = OUNoise(mean=np.zeros(1))

        self.actor = Actor(self.s_dim, self.a_dim).model()
        self.target_actor = Actor(self.s_dim, self.a_dim).model()
        self.actor_opt = tf.keras.optimizers.Adam(learning_rate=self.actor_lr)
        self.target_actor.set_weights(self.actor.get_weights())

        self.critic = Critic(self.s_dim, self.a_dim).model()
        self.critic_opt = tf.keras.optimizers.Adam(learning_rate=self.critic_lr)
        self.target_critic = Critic(self.s_dim, self.a_dim).model()
        self.target_critic.set_weights(self.critic.get_weights())

        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def update_target(self):
        # Two methods to update the target actor
        # Method 1:
        self.target_actor.set_weights(np.array(self.actor.get_weights()) * self.TAU + np.array(
            self.target_actor.get_weights()) * (1 - self.TAU))
        self.target_critic.set_weights(np.array(self.critic.get_weights()) * self.TAU + np.array(
            self.target_critic.get_weights()) * (1 - self.TAU))
        """
        # Method 2:
        new_weights = []
        target_variables = self.target_critic.weights
        for i, variable in enumerate(self.critic.weights):
            new_weights.append(variable * self.TAU + target_variables[i] * (1 - self.TAU))

        self.target_critic.set_weights(new_weights)
        new_weights = []
        target_variables = self.target_actor.weights
        for i, variable in enumerate(self.actor.weights):
            new_weights.append(variable * self.TAU + target_variables[i] * (1 - self.TAU))
        self.target_actor.set_weights(new_weights)
        """
    def train_step(self):
        s_batch, a_batch, r_batch, d_batch, s2_batch = self.replay_buffer.sample_batch(
            self.minibatch_size)

        """
        mu_prime = self.target_actor(s2_batch)  # predictions by target actor
        Q_prime = self.target_critic([s2_batch, mu_prime])  # predictions by target critic
        y = np.zeros_like(Q_prime)
        for k in range(self.minibatch_size):
            if d_batch[k]:
                y[k] = r_batch[k]
            else:
                y[k] = r_batch[k] + self.GAMMA * Q_prime[k]
        # y = r_batch + gamma * Q_prime

        checkpoint_path = "training/cp_critic.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        # Create a callback that saves the model's weights
        cp_callback1 = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_dir,
                                                          save_weights_only=True,
                                                          verbose=1)
        self.critic.train_on_batch([s_batch, a_batch], y)
        # self.critic.fit([s_batch, a_batch], y, verbose=0, steps_per_epoch=8, callbacks=[cp_callback1])

        with tf.GradientTape(persistent=True) as tape:
            a = self.actor(s_batch)
            tape.watch(a)
            theta = self.actor.trainable_variables
            q = self.critic([s_batch, a])
        dq_da = tape.gradient(q, a)
        da_dtheta = tape.gradient(a, theta, output_gradients=-dq_da)
        self.actor_opt.apply_gradients(zip(da_dtheta, self.actor.trainable_variables))
        """

        with tf.GradientTape() as tape:
            target_actions = self.target_actor(s2_batch)
            y = r_batch + self.GAMMA * self.target_critic([s2_batch, target_actions])
            critic_value = self.critic([s_batch, a_batch])
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))
        critic_grad = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic_grad, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions = self.actor(s_batch)
            q = self.critic([s_batch, actions])  # critic_value
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(q)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_grad, self.actor.trainable_variables))
        self.update_target()
        return np.mean(q)

    def policy(self, s):
        # since batch normalization is done on self.actor, it is multiplied with upper_bound
        if s.ndim == 1:
            s = s[None, :]
        action = self.actor(s) * self.upper_bound + self.ou_noise()
        action = np.clip(action, self.lower_bound, self.upper_bound)
        return action

    def train(self):
        # To store reward history of each episode
        ep_reward_list = []
        # To store average reward history of last few episodes
        avg_reward_list = []
        monitor = Monitor([1, 1], titles=['Reward', 'Loss'], log=2)
        with Loop_handler() as interruption:   # to properly save even if ctrl+C is pressed
            for eps in range(self.EPISODES):
                episode_reward = 0
                s = self.env.reset()
                """
                if an env is created using the "gym.make" method, it will terminate after 200 steps
                """
                for t in range(self.MAX_TIME_STEPS):
                # done = False
                # while not done:
                    if self.render:
                        self.env.render()
                    a = self.policy(s)
                    s_, r, done, _ = self.env.step(a)
                    self.replay_buffer.add(np.reshape(s, (self.s_dim,)), np.reshape(a, (self.a_dim,)), r,
                                           done, np.reshape(s_, (self.s_dim,)))
                    episode_reward += r
                    if self.replay_buffer.size() > self.minibatch_size:
                        q = self.train_step()
                    s = s_.reshape(1, -1)
                    if interruption():
                        break
                ep_reward_list.append(episode_reward)
                # Mean of last 40 episodes
                avg_reward = np.mean(ep_reward_list[-40:])
                print("Episode * {} * Avg Reward is ==> {}".format(eps, avg_reward))
                avg_reward_list.append(avg_reward)
                monitor.add_data(avg_reward, q)

            self.save_weights(save_name=self.save_name)  # if you want to save weights
            self.plot_results(avg_reward=avg_reward_list, train=True)

    def save_weights(self, save_name='final_weights'):
        self.actor.save_weights("training/%s_actor.h5" % save_name)
        self.critic.save_weights("training/%s_critic.h5" % save_name )
        self.target_actor.save_weights("training/%s_target_actor.h5" % save_name)
        self.target_critic.save_weights("training/%s_target_critic.h5" % save_name)

        # to save in other format
        self.target_actor.save_weights('training/%s_actor_weights' % save_name, save_format='tf')
        self.target_critic.save_weights('training/%s_critic_weights' % save_name, save_format='tf')
        print('Training completed and network weights saved')

    # For evaluation of the policy learned
    def collect_data(self, act_net, iterations=1000):
        a_all, states_all = [], []
        obs = self.env.reset()
        for t in range(iterations):
            obs = np.squeeze(obs)
            if obs.ndim == 1:
                a = act_net(obs[None, :])
            else:
                a = act_net(obs)
            obs, _, done, _ = self.env.step(a)
            states_all.append(obs)
            a_all.append(a)
            # self.env.render()  # Uncomment this to see the actor in action (But not in python notebook)
            # if done:
            #     break
        states = np.squeeze(np.array(states_all))  # cos(theta), sin(theta), theta_dot
        a_all = np.squeeze(np.array(a_all))
        return states, a_all

    def plot_results(self, avg_reward=None, actions=None, states=None, train=False, title=None):
        # An additional way to visualize the avg episode rewards
        if train:
            plt.figure()
            plt.plot(avg_reward)
            plt.xlabel("Episode")
            plt.ylabel("Avg. Epsiodic Reward")
            plt.show()
        else:  # work only for Pendulum-v0 environment
            fig, ax = plt.subplots(3, sharex=True)
            theta = np.arctan2(states[:, 1], states[:, 0])
            ax[0].set_ylabel('u')
            ax[0].plot(np.squeeze(actions))
            ax[1].set_ylabel(u'$\\theta$')
            ax[1].plot(theta)
            # ax[1].plot(states[:, 0])
            ax[2].set_ylabel(u'$\omega$')
            ax[2].plot(states[:, 2])  # ang velocity
            fig.canvas.set_window_title(title)


if __name__ == '__main__':
    env = gym.make('Pendulum-v0')
    render = True
    s_dim, a_dim = env.observation_space.shape[0], env.action_space.shape[0]
    critic_lr, actor_lr = 0.002, 0.001
    total_episodes = 100
    gamma = 0.99  # Discount factor for future rewards
    tau = 0.005  # Used to update target networks
    train = True
    ddpg = DDPG(env, s_dim=s_dim, a_dim=a_dim, gamma=gamma, tau=tau,
                      actor_lr=actor_lr, critic_lr=critic_lr)
    if train:
        ddpg.train()
    else:
        tgt_critic = ddpg.target_critic.load_weights('training/target_critic_weights')
        actor_trained = Actor(s_dim, a_dim).model()
        actor_trained.load_weights('training/target_actor_weights')

        s_trained, a_trained = ddpg.collect_data(actor_trained)
        ddpg.plot_results(actions=a_trained, states=s_trained, train=False, title='Trained_model')
        plt.show()
    print('Done')
        





