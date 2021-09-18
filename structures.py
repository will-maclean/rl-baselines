import random

import numpy as np


class SumTree:

    def __init__(self, capacity):
        """
        numpy implementation of a sum tree

        Parameters
        ----------
        capacity: number of leaf nodes
        """
        self.layers = []

        layer_size = 1
        tree_depth = int(np.log2(capacity)) + 1
        for _ in range(tree_depth + 1):
            self.layers.append(np.zeros(layer_size))
            layer_size *= 2

        self.num_layers = len(self.layers)
        self.capacity = capacity
        self.max_priority = 1

    def total_priority(self):
        return self.layers[0][0]

    def sample(self, search_for=None):
        """
        Returns the index associated with a sampled priority value.
        Parameters
        ----------
        search_for: can set this to something in the range [0, max_value_in_tree] - not required though

        Returns
        -------
        Index sampled by probability distribution
        """
        if search_for is not None and (search_for < 0 or search_for > self.total_priority()):
            raise ValueError(str(search_for) + " is out of range")
        elif search_for is None:
            search_for = random.random() * self.total_priority()

        index = 0
        for layer in range(len(self.layers) - 1):
            left_child_index = index * 2

            if search_for < self.layers[layer + 1][left_child_index]:
                index = left_child_index
            else:
                search_for -= self.layers[layer + 1][left_child_index]
                index = left_child_index + 1

        return index

    def sample_batch(self, batch_size) -> ([], np.array):
        if batch_size < 1:
            raise ValueError("Need to have a positive batch size!")

        interval_size = self.total_priority() / batch_size
        indexes = []
        probs = []
        lower = 0
        for _ in range(batch_size):
            search_for = random.random() * interval_size + lower
            indexes.append(self.sample(search_for))
            probs.append(self[indexes[-1]])
            lower += interval_size

        return indexes, np.array(probs)

    def __getitem__(self, node_index):
        """
        Gets priority of leaf with specified node

        Parameters
        ----------
        node_index: index of leaf node

        Returns
        -------
        associated priority
        """

        return self.layers[-1][node_index]

    def __setitem__(self, node_index, value):
        """
        Set leaf node at node_index with the given value
        Parameters
        ----------
        node_index: node index to set
        value: value/priority to store

        Returns
        -------
        None
        """
        if value < 0:
            raise ValueError("Value must be > 0")
        self.layers[-1][node_index] = value

        index = node_index // 2
        for layer in range(len(self.layers) - 2, -1, -1):
            index_val = self.layers[layer + 1][2 * index] + self.layers[layer + 1][2 * index + 1]
            self.layers[layer][index] = index_val
            index //= 2

        self.max_priority = max(self.max_priority, value)


class CircularQueue:
    def __init__(self, length):
        """
        Circular Queue used by prioritised experience. Not your usual queue, in that there's no
        serve/push, and you can access items by index, but its 2020.

        Parameters
        ----------
        length: length of queue
        """
        self.array = [None for _ in range(length)]
        self.current = 0
        self.length = length

    def append(self, item) -> int:
        """
        Appends given item to the rear of the queue
        Parameters
        ----------
        item: item to append

        Returns
        -------
        index of the append
        """
        self.array[self.current] = item
        ret_val = self.current
        self.current = (self.current + 1) % self.length
        return ret_val

    def __getitem__(self, index):
        """
        Gets item at specified index
        Parameters
        ----------
        index: index to return

        Returns
        -------
        item at specified index
        """
        return self.array[index]

    def __setitem__(self, key, value):
        self.array[key] = value

