from collections import deque
import random
from abc import ABC, abstractmethod


class ReplayBuffer(ABC):
    def __init__(self, max_memory):
        self.max_memory = max_memory

    @abstractmethod
    def sample(self, batch_size):
        pass

    @abstractmethod
    def append(self, experience):
        pass


class StandardReplayBuffer(ReplayBuffer):
    def __init__(self, max_mem_states):
        """
        ReplayMemory class to store agent experiences for training
        :param max_mem_states: maximum states to store in replay memory
        """
        super().__init__(max_mem_states)
        self.memory = deque(maxlen = max_mem_states)

    def sample(self, batch_size):
        """
        Sample a set amount of experiences according to the batch size
        :param batch_size: size of batch of experiences to sample
        :return: experiences zipped up
        """

        return zip(*random.sample(self.memory, batch_size))

    def append(self, experience):
        """
        Append experience to the memory
        :param experience: Tuple of experience
        """
        self.memory.append(experience)