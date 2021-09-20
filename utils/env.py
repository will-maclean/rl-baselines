import gym
import numpy as np


class ParallelEnv:
    def __init__(self, env_fn, n_envs,):
        self.n_envs = n_envs
        self.envs = [env_fn() for _ in range(n_envs)]

        self.state_shape = env_fn().reset().shape

    def step(self, a):
        next_s = np.zeros(self.n_envs, *self.state_shape)
        r = np.zeros(self.n_envs, 1)
        d = np.zeros(self.n_envs, 1)
        inf = []

        for i in range(self.n_envs):
            next_state, reward, done, info = self.envs[i].step(a[i])

            r[i] = reward
            d[i] = done
            inf.append(info)

            if done:
                next_s[i] = self.envs[i].reset()

        return next_s, r, d, inf

    def reset(self):
        s = np.zeros(self.n_envs, *self.state_shape)

        for i in range(self.n_envs):
            s[i] = self.envs[i].reset()

        return s
