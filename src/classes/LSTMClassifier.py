import torch
from torch import nn

from src.settings import NUM_LABELS


class LSTMClassifier(nn.Module):
    """
    LSTM Classifier for text classification.
    """

    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, pretrained_embeddings: np.ndarray):
        """
        Initialize the LSTM Classifier.

        :param vocab_size: Size of the vocabulary.
        :param embedding_dim: Dimension of the embedding vectors.
        :param hidden_dim: Dimension of the hidden layer.
        :param pretrained_embeddings: Pretrained embeddings to initialize the embedding layer.
        """
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight = nn.Parameter(torch.tensor(pretrained_embeddings, dtype=torch.float32))
        self.embedding.weight.requires_grad = True  # Optionally freeze the embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, NUM_LABELS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM Classifier.

        :param x: Input tensor.
        :return: Output tensor after passing through the LSTM and fully connected layers.
        """
        embedded = self.embedding(x.long())
        _, (hidden, _) = self.lstm(embedded)
        hidden = hidden[-1]  # Get the last layer's hidden state

        output = self.fc(hidden)

        return output
