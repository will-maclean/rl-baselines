from abc import ABC, abstractmethod

import numpy as np
import torch


class RLAgent(ABC):
    def __init__(self, env, name, device):
        self.name = name
        self.device = device

    @abstractmethod
    def act(self, state, env, step=-1):
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

    def wandb_watch(self):
        pass


class OfflineAgent(RLAgent, ABC):
    def __init__(self, env, name, memory, batch_size, device):
        RLAgent.__init__(self, env, name, device)
        self.memory = memory
        self.batch_size = batch_size

    def get_batch(self):
        states, actions, next_states, rewards, dones = self.memory.sample(self.batch_size)

        states = torch.from_numpy(np.stack([np.float32(state) for state in states])).to(self.device,
                                                                                        dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([np.float32(state) for state in next_states])).to(self.device,
                                                                                                  dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return states, actions, next_states, rewards, dones

    @abstractmethod
    def train_step(self, step):
        pass


class OnlineAgent(RLAgent, ABC):
    def __init__(self, env, name, device, memory):
        RLAgent.__init__(self, env, name, device)
        self.memory = memory

    @abstractmethod
    def train_step(self, step):
        pass

    def get_batch(self):
        states, actions, next_states, rewards, dones = self.memory.all()

        states = torch.from_numpy(np.stack([np.float32(state) for state in states])).to(self.device,
                                                                                        dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([np.float32(state) for state in next_states])).to(self.device,
                                                                                                  dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        return states, actions, next_states, rewards, dones

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def ready_to_train(self):
        pass
