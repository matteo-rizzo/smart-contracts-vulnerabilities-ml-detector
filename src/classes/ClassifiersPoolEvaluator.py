from typing import Dict, List

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier

from src.classes.ClassBalancer import ClassBalancer
from src.classes.MetricsHandler import MetricsHandler


class ClassifiersPoolEvaluator:

    def __init__(self, inputs: List[str], labels: List[List[int]], classifiers: Dict[str, object],
                 vectorizer: TfidfVectorizer, num_folds: int, random_seed: int):
        """
        Initialize the ClassifiersPoolEvaluator with TF-IDF vectorizer and a dictionary of classifiers.

        :param inputs: List of input documents.
        :param labels: List of labels corresponding to the input documents.
        :param classifiers: Dictionary of classifiers to evaluate.
        :param vectorizer: TF-IDF vectorizer for transforming input documents.
        :param num_folds: Number of folds for k-fold cross-validation.
        :param random_seed: Random seed for reproducibility.
        """
        self.__classifiers = classifiers
        self.__num_folds = num_folds
        self.__random_seed = random_seed

        # Transform the documents into TF-IDF features
        print("Transforming input documents into TF-IDF features...")
        self.x = vectorizer.fit_transform(inputs).toarray()

        # Transform the labels into a numpy array
        print("Converting labels to numpy array...")
        self.y = np.array(labels)

        # Compute class weights
        print("Computing class weights...")
        self.class_weights = self.__compute_class_weights()

    def __compute_class_weights(self) -> Dict[str, np.ndarray]:
        """
        Compute class weights for the dataset.

        :return: A dictionary containing the computed class weights.
        """
        class_sample_counts = self.y.sum(axis=0)
        weights = ClassBalancer.compute_weights(class_sample_counts)
        return {classifier_name: weights for classifier_name in self.__classifiers.keys()}

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

        # Splitting the dataset into training and testing parts
        x_train, x_test = self.x[train_index], self.x[test_index]
        y_train, y_test = self.y[train_index], self.y[test_index]

        # Train the classifier on the training data
        classifier.fit(x_train, y_train)
        # Make predictions on the test data
        predictions = classifier.predict(x_test)

        # Compute metrics using the provided utility function
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
        # Evaluate the classifier on each fold and collect the results
        results = []
        for fold_num, (train_index, test_index) in enumerate(mskf.split(self.x, self.y), 1):
            metrics = self.__evaluate_fold(classifier, train_index, test_index, fold_num)
            results.append(metrics)
        # Return the results as a DataFrame
        return pd.DataFrame(results)

    def pool_evaluation(self, log_dir="") -> None:
        """
        Run the evaluation for each classifier defined in self.__classifiers.
        """
        # Run the evaluation for each classifier defined in self.__classifiers
        for classifier_name, classifier in self.__classifiers.items():
            print(f"\nTesting classifier: {classifier_name}\n")

            # Apply class weights if they are provided for the classifier
            if classifier_name in self.class_weights:
                class_weight = self.class_weights[classifier_name]
                if 'class_weight' in classifier.get_params().keys():
                    classifier.set_params(class_weight=dict(enumerate(class_weight)))

            # Evaluate the classifier and get the metrics DataFrame
            metrics_df = self.__k_fold_cv(OneVsRestClassifier(classifier))
            # Save the results using the provided utility function
            MetricsHandler.save_results(metrics_df, f"{classifier_name}.csv", log_dir=log_dir)
