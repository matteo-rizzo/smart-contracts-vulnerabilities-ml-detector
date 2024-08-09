from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm


class WordEmbeddingProcessor:
    def __init__(self, glove_file: str, embedding_dim: int):
        """
        Initialize the WordEmbeddingProcessor with necessary configurations.

        :param glove_file: Path to the GloVe embeddings file.
        :param embedding_dim: Dimensionality of GloVe embeddings used.
        """
        self.glove_file = glove_file
        self.embedding_dim = embedding_dim
        self.vocabulary = {}
        self.embedding_matrix = None

    def load_glove_embeddings(self) -> Dict[str, np.ndarray]:
        """
        Load GloVe embeddings from a file.

        :return: Dictionary mapping words to their corresponding embedding vectors.
        """
        embeddings = {}
        with open(self.glove_file, 'r', encoding='utf-8') as file:
            for line in tqdm(file, desc="Loading GloVe Embeddings"):
                parts = line.split()
                word = parts[0]
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[word] = vector
        return embeddings

    def create_vocabulary(self, texts: List[str]) -> None:
        """
        Create a vocabulary from a list of texts.

        :param texts: List of texts to create the vocabulary from.
        """
        word_count = Counter(word for sentence in texts for word in sentence.lower().split())
        self.vocabulary = {word: i + 1 for i, word in enumerate(word_count)}  # start indexing from 1
        self.vocabulary['<PAD>'] = 0  # Padding value

    def create_embedding_matrix(self, glove_embeddings: Dict[str, np.ndarray]) -> None:
        """
        Create an embedding matrix from the GloVe embeddings and vocabulary.

        :param glove_embeddings: Dictionary mapping words to their corresponding GloVe embeddings.
        """
        self.embedding_matrix = np.zeros((len(self.vocabulary), self.embedding_dim))
        for word, i in tqdm(self.vocabulary.items(), desc='Creating Embedding Matrix'):
            embedding_vector = glove_embeddings.get(word)
            if embedding_vector is not None:
                self.embedding_matrix[i] = embedding_vector

    def text_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert a list of texts to sequences of integers based on the vocabulary.

        :param texts: List of texts to convert.
        :return: List of sequences of integers.
        """
        return [[self.vocabulary.get(word, self.vocabulary['<PAD>']) for word in text.lower().split()] for text in
                texts]

    def pad_sequences(self, sequences: List[List[int]]) -> List[List[int]]:
        """
        Pad sequences to ensure they all have the same length.

        :param sequences: List of sequences to pad.
        :return: List of padded sequences.
        """
        max_seq_len = max(len(seq) for seq in sequences)
        return [seq + [self.vocabulary['<PAD>']] * (max_seq_len - len(seq)) for seq in sequences]

    def process_embeddings(self, texts: List[str]) -> Tuple[int, int, np.ndarray]:
        """
        Process the texts to generate vocabulary, load embeddings, and create the embedding matrix.

        :param texts: List of texts to process.
        :return: Tuple containing the vocabulary length, embedding dimension, and embedding matrix.
        """
        self.create_vocabulary(texts)
        glove_embeddings = self.load_glove_embeddings()
        self.create_embedding_matrix(glove_embeddings)
        return len(self.vocabulary), self.embedding_dim, self.embedding_matrix
