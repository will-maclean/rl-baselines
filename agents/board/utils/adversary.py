from abc import ABC, abstractmethod

import numpy as np


class Adversary(ABC):
    @abstractmethod
    def act(self, env, state):
        pass


class RandomAdversary(Adversary):
    def act(self, env, state):
        return np.random.choice(env.legal_actions)
