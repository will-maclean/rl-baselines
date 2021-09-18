from abc import ABC, abstractmethod


class RLAgent(ABC):
    def __init__(self, env, name, device):
        self.name = name
        self.device = device

    @abstractmethod
    def act(self, state, env, step=-1):
        pass

    @abstractmethod
    def train_step(self, step):
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


class OfflineAgent(RLAgent, ABC):
    def __init__(self, env, name, memory, batch_size, device):
        RLAgent.__init__(self, env, name, device)
        self.memory = memory
        self.batch_size = batch_size
