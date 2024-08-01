import torch
from torch import nn

from src.settings import MAX_FEATURES, NUM_LABELS


class FFNNClassifier(nn.Module):
    """
    Simple Neural Network with three fully connected layers.
    """

    def __init__(self):
        """
        Initialize the network layers.
        """
        super(FFNNClassifier, self).__init__()
        self.fc1 = nn.Linear(MAX_FEATURES, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, NUM_LABELS)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        :param x: Input tensor
        :return: Output tensor
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x
