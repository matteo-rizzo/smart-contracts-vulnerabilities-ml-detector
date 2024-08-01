from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier

from src.settings import CLASSIFIERS, VECTORIZER, NUM_FOLDS, RANDOM_SEED
from src.utility import save_results, compute_metrics


class ClassifiersPoolEvaluator:
    """
    ClassifiersPoolEvaluator class for evaluating a pool of classifiers using TF-IDF features and k-fold cross-validation.
    """

    def __init__(self, inputs, labels):
        """
        Initialize the ClassifiersPoolEvaluator with TF-IDF vectorizer and a dictionary of classifiers.
        """
        # Define a dictionary of classifiers to evaluate

        # Transform the documents into TF-IDF features
        self.X = VECTORIZER.fit_transform(inputs)

        # Transform the labels into a numpy array
        self.y = np.array(labels)

    def __evaluate_fold(self, classifier: OneVsRestClassifier, train_index: List[int], test_index: List[int],
                        fold_num: int) -> Dict[str, float]:
        """
        Evaluate a classifier on a single fold of cross-validation.

        :param classifier: The classifier to be evaluated.
        :param train_index: Indices for the training data.
        :param test_index: Indices for the test data.
        :param fold_num: The fold number.
        :return: A dictionary of computed metrics.
        """
        X_train, X_test = self.X[train_index], self.X[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]

        # Train the classifier on the training data
        classifier.fit(X_train, y_train)
        # Make predictions on the test data
        predictions = classifier.predict(X_test)

        # Compute metrics using the provided utility function
        metrics = compute_metrics(y_test, predictions)
        print(f"Results for fold {fold_num} | "
              f"Precision: {metrics['precision']:.4f}, "
              f"Recall: {metrics['recall']:.4f}, "
              f"F1: {metrics['f1']:.4f}")
        return metrics

    def __k_fold_cv(self, classifier: OneVsRestClassifier) -> pd.DataFrame:
        """
        Perform k-fold cross-validation on a given classifier.

        :param classifier: The classifier to be evaluated.
        :return: A DataFrame containing the results of each fold.
        """
        kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        # Evaluate the classifier on each fold and collect the results
        results = []
        for fold_num, (train_index, test_index) in enumerate(kf.split(self.X), 1):
            metrics = self.__evaluate_fold(classifier, train_index, test_index, fold_num)
            results.append(metrics)
        # Return the results as a DataFrame
        return pd.DataFrame(results)

    def pool_evaluation(self) -> None:
        """
        Run the evaluation for each classifier defined in self.classifiers.
        """
        # Run the evaluation for each classifier defined in self.classifiers
        for classifier_name, classifier in CLASSIFIERS.items():
            print(f"\nTesting classifier: {classifier_name}\n")
            # Evaluate the classifier and get the metrics DataFrame
            metrics_df = self.__k_fold_cv(OneVsRestClassifier(classifier))
            # Save the results using the provided utility function
            save_results(metrics_df, f"{classifier_name}.csv")
