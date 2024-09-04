import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


class GCNClassifier(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the GCNClassifier for multilabel classification.

        :param input_size: The number of input features.
        :param hidden_size: The number of hidden units in the hidden layers.
        :param output_size: The number of output classes.
        """
        super(GCNClassifier, self).__init__()
        self.conv1 = GCNConv(input_size, hidden_size)
        self.bn1 = BatchNorm1d(hidden_size)
        self.conv2 = GCNConv(hidden_size, hidden_size)
        self.bn2 = BatchNorm1d(hidden_size)
        self.conv3 = GCNConv(hidden_size, output_size)  # Output layer for multilabel classification

    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the network.

        :param data: The input data containing node features and edge indices.
        :return: The output logits of the network for multilabel classification.
        """
        x, edge_index = data.x, data.edge_index

        # Ensure edge_index is of type torch.long (int64)
        edge_index = edge_index.long()

        # First layer
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Second layer
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)

        # Output layer (raw logits)
        x = self.conv3(x, edge_index)

        # For multilabel classification, we apply sigmoid to get probabilities
        x = torch.sigmoid(x)

        return x  # Sigmoid output (probabilities between 0 and 1 for each class)
