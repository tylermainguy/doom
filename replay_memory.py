import random
from collections import deque

import torch


class ReplayMemory:
    """
    Replay memory to be used by the network for training. Stores a fixed size
    buffer containing a history of transitions seen by the network.
    """

    def __init__(self, size):
        self.max_size = size
        self.buffer = deque(maxlen=size)

    def add_memory(self, state, action, new_state, reward):
        """Add memory (transition) to replay memory."""
        self.buffer.append((state, action, new_state, reward))
        self.curr_size = len(self.buffer)

    def sample(self, batch_size):
        """Sample memories for training."""
        sample_indices = set(random.sample(range(len(self.buffer)), batch_size))
        batch = [self.buffer[idx] for idx in sample_indices]

        return batch

    def __len__(self):
        return len(self.buffer)
