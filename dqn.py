import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    """Module implementing a deep-Q network."""

    def __init__(self, height, width, in_channels=4, num_actions=20):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=32, kernel_size=7)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)

        # calculate size of output after convolutional layers
        def conv2d_size_out(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(width, kernel_size=7, stride=1), kernel_size=5, stride=1
            )
        )
        convh = conv2d_size_out(
            conv2d_size_out(
                conv2d_size_out(height, kernel_size=7, stride=1), kernel_size=5, stride=1
            )
        )

        self.fc4 = nn.Linear(convw * convh * 64, num_actions)  # num actions = 20

    def forward(self, x):
        """Forward pass."""
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        return self.fc4(x.view(x.size(0), -1))
