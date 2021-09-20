from collections import deque
import random
from abc import ABC, abstractmethod

import numpy as np

from .structures import CircularQueue, SumTree


class ReplayBuffer(ABC):
    def __init__(self, max_memory):
        self.max_memory = max_memory

    @abstractmethod
    def sample(self, batch_size, *args):
        pass

    @abstractmethod
    def append(self, *experience):
        pass


class StandardReplayBuffer(ReplayBuffer):
    def __init__(self, max_mem_states):
        """
        ReplayMemory class to store agent experiences for training
        :param max_mem_states: maximum states to store in replay memory
        """
        super().__init__(max_mem_states)
        self.memory = deque(maxlen=max_mem_states)

    def sample(self, batch_size, *args):
        """
        Sample a set amount of experiences according to the batch size
        :param batch_size: size of batch of experiences to sample
        :return: experiences zipped up
        """

        return zip(*random.sample(self.memory, batch_size))

    def all(self):
        return zip(*self.memory)

    def append(self, *experience):
        """
        Append experience to the memory
        :param experience: Tuple of experience
        """
        self.memory.append(experience)

    def is_full(self):
        return len(self.memory) == self.max_memory

    def clear(self):
        self.memory.clear()


class PrioritisedReplayBuffer(ReplayBuffer):

    def __init__(self,
                 max_memory,
                 beta_frames=None,
                 alpha: float = 0.6,
                 beta0: float = 0.4,
                 epsilon: float = 1e-10
                 ):
        """
        Prioritised replay class instead of evenly sampling memories we have a higher chance of sampling events that had
        a larger TD error, events that behaved unexpectedly eg a sudden death when we thought we were going well

        Parameters
        ----------
        max_memory: we all know this one
        beta_frames: how many steps our until our beta anneals to 1
        alpha: our priority factor. alpha = 0 means standard memory behaviour
        beta0: initial bias factor
        epsilon: small offset for when priorities near zero (prevents any from reaching zero)
        """
        super().__init__(
            max_memory
        )
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta_frames = beta_frames if beta_frames is not None else max_memory
        self.max_memory = max_memory
        self.beta0 = beta0
        self.experience = CircularQueue(max_memory)  # circular queue in structures.py for storing experiences
        self.tree = SumTree(
            max_memory)  # Sum tree from structures.py used to sort priorities and speed probability calculation
        self.max_weight = 1  # original priority events are first appending with
        self.full = False
        self.count = 0

    def append(self, *args):
        """
        The circular queue class is appended with an experience which then returns an index which is used to address the
        leaf of the tree that priority is stored into. IS weights is also calculated and appended to a similar list.
        """
        index = self.experience.append(args)
        self.tree[index] = self.tree.max_priority ** self.alpha
        if not self.full:
            self.count += 1
        if self.count >= self.max_memory:
            self.full = True

    def sample(self, batch_size, *args):
        """
        Calls the batch sample function of the sum tree class and which will return indexes based on priorities stored
        """
        step = args[0]

        indexes, priorities = self.tree.sample_batch(batch_size)
        priorities /= self.tree.total_priority()
        weights = (self.max_memory * priorities) ** (-1 * self._calculate_beta(step))
        weights /= self.max_weight
        sampled_exp = []
        for index in indexes:
            sampled_exp.append(self.experience[index])

        self.max_weight = max(max(weights), self.max_weight)

        return weights, sampled_exp, indexes

    def update_priorities(self, td_errors: np.array, indexes):
        for index, td_error in zip(indexes, td_errors):
            self.tree[index] = abs(td_error)

    def _calculate_beta(self, step):
        gradient = (1 - self.beta0) / self.beta_frames
        return self.beta0 + step * gradient

    def __len__(self):
        if self.full:
            return self.max_memory
        else:
            return self.count
