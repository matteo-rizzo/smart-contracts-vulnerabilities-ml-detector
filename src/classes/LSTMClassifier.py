import numpy as np
import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """
    LSTM Classifier for text classification.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, pretrained_embeddings: np.ndarray,
                 output_size: int):
        """
        Initialize the LSTM Classifier.

        :param vocab_size: Size of the vocabulary.
        :param embedding_dim: Dimension of the embedding vectors.
        :param hidden_dim: Dimension of the hidden layer.
        :param pretrained_embeddings: Pretrained embeddings to initialize the embedding layer.
        :param output_size: Number of output classes.
        """
        super(LSTMClassifier, self).__init__()

        # Embedding layer initialized with pretrained embeddings
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(pretrained_embeddings, dtype=torch.float32))
        self.embedding.weight.requires_grad = True  # Set to False to freeze embeddings

        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)

        # Fully connected output layer
        self.fc = nn.Linear(hidden_dim, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM Classifier.

        :param x: Input tensor.
        :return: Output tensor after passing through the LSTM and fully connected layers.
        """
        # Convert input to embeddings
        embedded = self.embedding(x.long())

        # Pass embeddings through LSTM
        _, (hidden, _) = self.lstm(embedded)

        # Use the last hidden state for classification
        hidden = hidden[-1]

        # Pass the last hidden state through the fully connected layer
        output = self.fc(hidden)

        return output
