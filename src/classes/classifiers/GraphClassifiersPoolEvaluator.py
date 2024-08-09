from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from torch_geometric.data import Data

from src.classes.training.MetricsHandler import MetricsHandler


class GraphClassifiersPoolEvaluator:
    def __init__(self, inputs: List[Data], labels: List[List[int]], classifiers: Dict[str, object], num_folds: int,
                 random_seed: int):
        """
        Initialize the ClassifiersPoolEvaluator with inputs, labels, classifiers, number of folds, and random seed.

        :param inputs: List of graph data objects.
        :param labels: List of labels corresponding to the graph data objects.
        :param classifiers: Dictionary of classifiers to evaluate.
        :param num_folds: Number of folds for k-fold cross-validation.
        :param random_seed: Random seed for reproducibility.
        """
        self.__classifiers = classifiers
        self.__num_folds = num_folds
        self.__random_seed = random_seed
        self.__inputs = inputs
        self.__labels = np.array(labels)

        print(f"Inputs shape: {len(inputs)}, Labels shape: {self.__labels.shape}")

        # Extract features and labels
        self.__features, self.__labels = self._extract_features_and_labels(self.__inputs)
        print(f"Extracted features shape: {self.__features.shape}, Extracted labels shape: {self.__labels.shape}")

    @staticmethod
    def _extract_features_and_labels(graph_dataset: List[Data]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract features and labels from the graph dataset.

        :param graph_dataset: The graph dataset.
        :return: A tuple of features and labels as numpy arrays.
        """
        features = []
        labels = []

        for data in graph_dataset:
            features.append(data.x.numpy())  # Assuming `data.x` contains the features as a tensor
            labels.append(data.y[0].numpy())  # Assuming `data.y` contains the labels as a tensor

        # Determine the maximum shape for padding over the first dimension (number of nodes)
        max_nodes = max(f.shape[0] for f in features)

        # Pad features to ensure they all have the same shape over the first dimension
        padded_features = np.array([np.pad(f, ((0, max_nodes - f.shape[0]), (0, 0)), 'constant') for f in features])

        # Flatten the features for classifier compatibility
        flattened_features = padded_features.reshape(len(padded_features), -1)

        # Convert labels to a numpy array
        labels = np.array(labels)

        return flattened_features, labels

    def __evaluate_fold(self, classifier: OneVsRestClassifier, train_index: List[int], test_index: List[int],
                        fold_num: int) -> Dict[str, float]:
        """
        Evaluate a single fold during cross-validation.

        :param classifier: The classifier to be evaluated.
        :param train_index: Indices of the training samples.
        :param test_index: Indices of the test samples.
        :param fold_num: The fold number.
        :return: A dictionary containing the evaluation metrics for the fold.
        """
        x_train, x_test = self.__features[train_index], self.__features[test_index]
        y_train, y_test = self.__labels[train_index], self.__labels[test_index]

        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)

        metrics = MetricsHandler.compute_metrics(y_test, predictions)
        print(f"Results for fold {fold_num} | ", end="")
        MetricsHandler.print_metrics(metrics)
        return metrics

    def __k_fold_cv(self, classifier: OneVsRestClassifier) -> pd.DataFrame:
        """
        Perform k-fold cross-validation on a given classifier.

        :param classifier: The classifier to be evaluated.
        :return: A DataFrame containing the results of each fold.
        """
        mskf = MultilabelStratifiedKFold(n_splits=self.__num_folds, shuffle=True, random_state=self.__random_seed)
        indices = list(range(len(self.__inputs)))
        results = [self.__evaluate_fold(classifier, train_index, test_index, fold_num)
                   for fold_num, (train_index, test_index) in enumerate(mskf.split(indices, self.__labels), 1)]
        return pd.DataFrame(results)

    def pool_evaluation(self, log_dir: str = "") -> None:
        """
        Evaluate all classifiers in the pool and save the results.

        :param log_dir: Directory to save the evaluation logs.
        """
        for classifier_name, classifier in self.__classifiers.items():
            print(f"\nTesting classifier: {classifier_name}\n")

            metrics_df = self.__k_fold_cv(OneVsRestClassifier(classifier))
            MetricsHandler.save_results(metrics_df, f"{classifier_name}.csv", log_dir=log_dir)

            # Print average metrics across folds
            average_metrics = metrics_df.mean().to_dict()
            print(f"\nAverage results for {classifier_name}:")
            MetricsHandler.print_metrics(average_metrics)
