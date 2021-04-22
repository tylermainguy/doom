"""Deep Q-Network (DQN) Module."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class DQN(nn.Module):
    """Module implementing a deep-Q network."""

    def __init__(self, height, width, in_channels=4, num_actions=20):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=7)
        self.conv2 = nn.Conv2d(32, out_channels=32, kernel_size=4)

        self.fc1 = nn.Linear(32 * 51 * 71, 1024)
        self.fc2 = nn.Linear(1024, num_actions)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Batch of states.

        Returns
        ----------
        Tensor
            Q value for each action.

        """

        # convolutional layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        # flatten for fully connected layers
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(self.fc1(x))

        return self.fc2(x)
