from abc import ABC, abstractmethod


class RLAgent(ABC):
    def __init__(self, env, name):
        self.name = name

    @abstractmethod
    def act(self, state, env, step=-1):
        pass

    @abstractmethod
    def train_step(self, batch, step):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass

    @abstractmethod
    def log(self, log_dict):
        pass

    @abstractmethod
    def config(self):
        pass
