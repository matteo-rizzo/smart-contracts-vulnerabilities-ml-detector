from typing import Dict, List

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.multiclass import OneVsRestClassifier

from src.classes.ClassBalancer import ClassBalancer
from src.classes.MetricsHandler import MetricsHandler


class ClassifiersPoolEvaluator:
    def __init__(self, inputs: np.ndarray, labels: np.ndarray, classifiers: Dict[str, object], num_folds: int,
                 random_seed: int):
        """
        Initialize the ClassifiersPoolEvaluator with inputs, labels, classifiers, number of folds, and random seed.

        :param inputs: Array of input features.
        :param labels: Array of labels corresponding to the input features.
        :param classifiers: Dictionary of classifiers to evaluate.
        :param num_folds: Number of folds for k-fold cross-validation.
        :param random_seed: Random seed for reproducibility.
        """
        self.__classifiers = classifiers
        self.__num_folds = num_folds
        self.__random_seed = random_seed
        self.__x = inputs
        self.__y = labels

        print("Computing class weights...")
        self.__class_weights = self.__compute_class_weights()

    def __compute_class_weights(self) -> Dict[str, np.ndarray]:
        """
        Compute class weights for the dataset.

        :return: A dictionary containing the computed class weights.
        """
        class_sample_counts = self.__y.sum(axis=0)
        weights = ClassBalancer.compute_weights(class_sample_counts)
        return {classifier_name: weights for classifier_name in self.__classifiers.keys()}

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
        x_train, x_test = self.__x[train_index], self.__x[test_index]
        y_train, y_test = self.__y[train_index], self.__y[test_index]

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
        results = [self.__evaluate_fold(classifier, train_index, test_index, fold_num)
                   for fold_num, (train_index, test_index) in enumerate(mskf.split(self.__x, self.__y), 1)]
        return pd.DataFrame(results)

    def __apply_class_weights(self, classifier_name: str, classifier: object) -> object:
        """
        Apply class weights to the classifier if applicable.

        :param classifier_name: The name of the classifier.
        :param classifier: The classifier instance.
        :return: The classifier with class weights applied if applicable.
        """
        if classifier_name in self.__class_weights and 'class_weight' in classifier.get_params():
            classifier.set_params(class_weight=dict(enumerate(self.__class_weights[classifier_name])))
        return classifier

    def pool_evaluation(self, log_dir: str = "") -> None:
        """
        Evaluate all classifiers in the pool and save the results.

        :param log_dir: Directory to save the evaluation logs.
        """
        for classifier_name, classifier in self.__classifiers.items():
            print(f"\nTesting classifier: {classifier_name}\n")

            classifier = self.__apply_class_weights(classifier_name, classifier)
            metrics_df = self.__k_fold_cv(OneVsRestClassifier(classifier))
            MetricsHandler.save_results(metrics_df, f"{classifier_name}.csv", log_dir=log_dir)

            # Print average metrics across folds
            average_metrics = metrics_df.mean().to_dict()
            print(f"\nAverage results for {classifier_name}:")
            MetricsHandler.print_metrics(average_metrics)
