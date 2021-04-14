import numpy as np


class OUNoise(object):
    def __init__(self, mean, std_deviation=0.2, theta=0.15, dt=1e-2, state=None):
        self.mu = mean
        self.theta = theta
        self.sigma = std_deviation
        self.dt = dt
        self.state = state
        self.reset()

    def reset(self):
        if self.state is not None:
            self.state_prev = self.state
        else:
            self.state_prev = np.zeros_like(self.mu)

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        # dx_{t}= \theta * (\mu -x_{t}) * dt+\sigma * dW_{t}, where W_t is a weiner process
        dx = self.theta * (self.mu - self.state_prev) * self.dt + np.sqrt(self.dt) * self.sigma * np.random.normal(size=self.mu.shape)
        self.state_prev += dx
        return self.state_prev


if __name__ == '__main__':
    mean, std_dev = np.zeros(1)+0.13, 0.2
    ou_ash = OUNoise(mean=mean, std_deviation=std_dev)
    print(ou_ash())
    print('hi')
