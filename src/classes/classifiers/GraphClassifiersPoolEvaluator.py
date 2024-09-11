from typing import List, Dict, Tuple

import numpy as np
from torch_geometric.data import Data

from src.classes.classifiers.ClassifiersPoolEvaluator import ClassifiersPoolEvaluator


class GraphClassifiersPoolEvaluator(ClassifiersPoolEvaluator):
    def __init__(self, inputs: List[Data], labels: List[List[int]], classifiers: Dict[str, object], num_folds: int,
                 random_seed: int, pca_components: int = None):
        """
        Public method: Initialize the GraphClassifiersPoolEvaluator with graph inputs, labels, classifiers, number
        of folds, random seed, and optional PCA components.
        """
        # Extract features and labels from graph dataset
        features, extracted_labels = self._extract_features_and_labels(inputs)

        # Initialize the base class with the extracted features
        super().__init__(features, np.array(extracted_labels), classifiers, num_folds, random_seed, pca_components)

    @staticmethod
    def _extract_features_and_labels(graph_dataset: List[Data]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from the graph dataset, ensuring labels are 2D for stratification.
        """
        if not graph_dataset:
            raise ValueError("The graph dataset is empty.")

        num_graphs = len(graph_dataset)

        # Determine max nodes for feature padding
        max_nodes = max(data.x.shape[0] for data in graph_dataset)

        # Pre-allocate features
        feature_dim = graph_dataset[0].x.shape[1]
        padded_features = np.zeros((num_graphs, max_nodes * feature_dim), dtype=np.float32)

        # Determine max label shape (multi-dimensional labels)
        max_label_shape = max(data.y.shape for data in graph_dataset)
        label_shape = (num_graphs,) + max_label_shape
        padded_labels = np.zeros(label_shape, dtype=np.float32)

        for i, data in enumerate(graph_dataset):
            num_nodes = data.x.shape[0]
            flattened_features = data.x.numpy().flatten()
            padded_features[i, :num_nodes * feature_dim] = flattened_features

            # Pad labels to max_label_shape
            label_len = data.y.shape
            padded_labels[i, :label_len[0], ...] = data.y.numpy()

        # Reshape or squeeze the labels to ensure they are 2D for stratification
        padded_labels = np.squeeze(padded_labels)

        # If the labels are still not 2D, reshape them appropriately
        if len(padded_labels.shape) == 3:
            padded_labels = padded_labels.reshape((num_graphs, -1))  # Flatten the last two dimensions

        return padded_features, padded_labels
