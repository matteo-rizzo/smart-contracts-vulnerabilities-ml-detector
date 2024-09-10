from typing import Dict, List

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier

from src.classes.training.MetricsHandler import MetricsHandler


class ClassifiersPoolEvaluator:
    def __init__(self, inputs: np.ndarray, labels: np.ndarray, classifiers: Dict[str, object], num_folds: int,
                 random_seed: int, pca_components: int = None):
        """
        Public method: Initialize the ClassifiersPoolEvaluator with inputs, labels, classifiers, number of folds,
        random seed, and optional PCA components.
        """
        self.classifiers = classifiers
        self.num_folds = num_folds
        self.random_seed = random_seed
        self.inputs = inputs
        self.labels = labels
        self.pca_components = pca_components

        if self.pca_components is not None:
            self._apply_pca()  # Protected method since it's internal and related to pre-processing

    def _apply_pca(self) -> None:
        """
        Protected method: Apply PCA to reduce the dimensionality of the input data.
        """
        print(f"Applying PCA to reduce dimensionality to {self.pca_components} components.")
        pca = PCA(n_components=self.pca_components, random_state=self.random_seed)
        self.inputs = pca.fit_transform(self.inputs)
        print(f"New shape of inputs after PCA: {self.inputs.shape}")

    def _evaluate_fold(self, classifier: OneVsRestClassifier, train_index: List[int], test_index: List[int],
                       fold_num: int) -> Dict[str, float]:
        """
        Protected method: Evaluate a single fold during cross-validation.
        """
        x_train, x_test = self.inputs[train_index], self.inputs[test_index]
        y_train, y_test = self.labels[train_index], self.labels[test_index]

        classifier.fit(x_train, y_train)
        predictions = classifier.predict(x_test)

        metrics = MetricsHandler.compute_metrics(y_test, predictions)
        print(f"Results for fold {fold_num} | ", end="")
        MetricsHandler.print_metrics(metrics)
        return metrics

    def _k_fold_cv(self, classifier: OneVsRestClassifier) -> pd.DataFrame:
        """
        Protected method: Perform k-fold cross-validation on a given classifier.
        """
        mskf = MultilabelStratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=self.random_seed)
        results = [self._evaluate_fold(classifier, train_index, test_index, fold_num)
                   for fold_num, (train_index, test_index) in enumerate(mskf.split(self.inputs, self.labels), 1)]
        return pd.DataFrame(results)

    def pool_evaluation(self, log_dir: str = "") -> None:
        """
        Public method: Evaluate all classifiers in the pool and save the results.
        """
        for classifier_name, classifier in self.classifiers.items():
            print(f"\nTesting classifier: {classifier_name}\n")

            metrics_df = self._k_fold_cv(OneVsRestClassifier(classifier))
            MetricsHandler.save_results(metrics_df, f"{classifier_name}.csv", log_dir=log_dir)

            average_metrics = metrics_df.mean().to_dict()
            print(f"\nAverage results for {classifier_name}:")
            MetricsHandler.print_metrics(average_metrics)
