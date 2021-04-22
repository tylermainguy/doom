"""Replay memory for experience replay training of DQN."""

import random
from collections import deque

import torch
from torch import Tensor


class ReplayMemory:
    """
    Replay memory to be used by the network for training. Stores a fixed size
    buffer containing a history of transitions seen by the network.
    """

    def __init__(self, size):
        self.max_size = size
        self.buffer = deque(maxlen=size)

    def add_memory(self, state: Tensor, action: Tensor, new_state: Tensor, reward: Tensor):
        """
        Add memory (transition) to replay memory.

        Parameters
        ----------
        state : Tensor
            State (s) that action was taken from.
        action : Tensor
            Number corresponding to action.
        new_state : Tensor
            State (s') transitioned to.
        reward : Tensor
            Reward for action in state s.

        """

        self.buffer.append((state, action, new_state, reward))
        self.curr_size = len(self.buffer)

    def sample(self, batch_size: int) -> list:
        """
        Sample memories for training DQN.

        Parameters
        ----------
        batch_size : int
            Number of samples in a batch.

        Returns
        ----------
        list
            Batch of transitions.

        """
        # randomly sample batch
        sample_indices = set(random.sample(range(len(self.buffer)), batch_size))
        batch = [self.buffer[idx] for idx in sample_indices]

        return batch

    def __len__(self):
        return len(self.buffer)
