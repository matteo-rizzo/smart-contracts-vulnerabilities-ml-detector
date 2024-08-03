from typing import Tuple, Union

import numpy as np
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from torch import Tensor
from torch.utils.data import Subset
from torch.utils.data.sampler import WeightedRandomSampler


class ClassBalancer:
    @staticmethod
    def compute_weights(counts: np.ndarray) -> np.ndarray:
        """
        Compute weights for each class to handle class imbalance.

        :param counts: Array containing the count of each class.
        :return: Array containing the computed weights for each class.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            weights = np.divide(1.0, counts, out=np.zeros_like(counts, dtype=float), where=counts != 0)
        weights = np.where(weights == 0, 1e-10, weights)  # Ensure no zero weights
        return weights

    @staticmethod
    def make_balanced_sampler(subset: Subset) -> WeightedRandomSampler:
        """
        Create a balanced sampler for the subset to handle class imbalance.

        :param subset: The subset of the original dataset for which to create the sampler.
        :return: A WeightedRandomSampler instance.
        """
        labels = subset.dataset.tensors[-1][subset.indices]
        label_counts = labels.sum(axis=0).numpy()
        weights = ClassBalancer.compute_weights(label_counts)
        sample_weights = np.dot(labels.numpy(), weights)
        return WeightedRandomSampler(sample_weights, len(sample_weights))

    @staticmethod
    def calculate_class_weights(subset: Subset) -> torch.FloatTensor:
        """
        Calculate class weights for the subset to handle class imbalance.

        :param subset: The subset of the original dataset for which to calculate class weights.
        :return: A torch.FloatTensor containing the class weights.
        """
        labels = subset.dataset.tensors[-1][subset.indices]
        class_sample_counts = labels.sum(axis=0).numpy()
        weights = ClassBalancer.compute_weights(class_sample_counts)
        return torch.FloatTensor(weights)

    @staticmethod
    def train_test_split(x: Union[Tensor, np.ndarray], y: Union[Tensor, np.ndarray], test_size: float,
                         random_state: int) -> Tuple:
        """
        Split the data into training and test sets in a balanced manner for multilabel data using
        iterative stratification.

        :param x: Features array.
        :param y: Labels array.
        :param test_size: Proportion of the dataset to include in the test split.
        :param random_state: The random seed for reproducibility
        :return: Train and test datasets.
        """
        mskf = MultilabelStratifiedKFold(n_splits=int(1 / test_size), shuffle=True, random_state=random_state)
        train_indices, test_indices = next(mskf.split(x, y))

        x_train, x_test = x[train_indices], x[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        return x_train, x_test, y_train, y_test
