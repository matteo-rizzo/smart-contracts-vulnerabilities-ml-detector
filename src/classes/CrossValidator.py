from typing import Any, List, Dict
import numpy as np
from numpy import floating
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Subset, RandomSampler
import matplotlib.pyplot as plt
from src.classes.Trainer import Trainer
from src.utility import save_results


class CrossValidator:
    """
    CrossValidator class for handling k-fold cross-validation of a model.
    """

    def __init__(self, trainer: Trainer, train_data: TensorDataset, test_data: TensorDataset, num_epochs: int,
                 num_folds: int, batch_size: int):
        """
        Initialize the CrossValidator with trainer, training data, test data, and configuration parameters.

        :param trainer: An instance of the Trainer class.
        :param train_data: The training dataset.
        :param test_data: The test dataset.
        :param num_epochs: Number of epochs to train the model.
        :param num_folds: Number of folds for cross-validation.
        :param batch_size: Batch size for data loading.
        """
        self.__trainer = trainer
        self.__train_data = train_data
        self.__test_data = test_data
        self.__num_epochs = num_epochs
        self.__num_folds = num_folds
        self.__batch_size = batch_size

    def __train_and_evaluate(self, train_dataloader: DataLoader, test_dataloader: DataLoader) -> Dict[str, List[float]]:
        """
        Train and evaluate the model for a specified number of epochs.

        :param train_dataloader: DataLoader for the training data.
        :param test_dataloader: DataLoader for the validation data.
        :return: A dictionary containing lists of training and validation losses and metrics for each epoch.
        """
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_metrics': [],
            'val_metrics': []
        }

        for epoch in range(self.__num_epochs):
            print(f"\n --- Epoch {epoch + 1}/{self.__num_epochs} ---")

            # Train the model and print training metrics
            avg_train_loss, avg_train_metrics = self.__trainer.run_epoch(train_dataloader, train_mode=True)
            training_history['train_loss'].append(avg_train_loss)
            training_history['train_metrics'].append(avg_train_metrics)
            print(f"\n TRAIN | Loss: {avg_train_loss:.4f} |"
                  f" Precision: {avg_train_metrics['precision']:.4f},"
                  f" Recall: {avg_train_metrics['recall']:.4f},"
                  f" F1: {avg_train_metrics['f1']:.4f}\n")

            # Evaluate the model on the validation set and print validation metrics
            avg_test_loss, avg_test_metrics = self.__trainer.run_epoch(test_dataloader, train_mode=False)
            training_history['val_loss'].append(avg_test_loss)
            training_history['val_metrics'].append(avg_test_metrics)
            print(f" VALID | Loss: {avg_test_loss:.4f} |"
                  f" Precision: {avg_test_metrics['precision']:.4f},"
                  f" Recall: {avg_test_metrics['recall']:.4f},"
                  f" F1: {avg_test_metrics['f1']:.4f}\n")

        return training_history

    def __evaluate_on_test_set(self, test_dataloader: DataLoader) -> dict[str, floating[Any]]:
        """
        Evaluate the model on the test set.

        :param test_dataloader: DataLoader for the test data.
        :return: A dictionary of test set metrics.
        """
        avg_test_loss, avg_test_metrics = self.__trainer.run_epoch(test_dataloader, train_mode=False)

        # Print test set metrics
        print(f"\nTest Set Evaluation | Loss: {avg_test_loss:.4f} |"
              f" Precision: {avg_test_metrics['precision']:.4f},"
              f" Recall: {avg_test_metrics['recall']:.4f},"
              f" F1: {avg_test_metrics['f1']:.4f}\n")

        return avg_test_metrics

    def __plot_metrics(self, history: Dict[str, List[float]], fold: int, log_id: str, log_dir: str) -> None:
        """
        Plot training and validation metrics and save the plots.

        :param history: A dictionary containing lists of training and validation losses and metrics for each epoch.
        :param fold: The current fold number.
        :param log_id: Identifier for logging purposes, typically the model name.
        :param log_dir: The logging directory.
        """
        epochs = range(1, self.__num_epochs + 1)
        metrics = ['precision', 'recall', 'f1']

        fig, axs = plt.subplots(3, 1, figsize=(10, 15))

        for i, metric in enumerate(metrics):
            train_metric = [m[metric] for m in history['train_metrics']]
            val_metric = [m[metric] for m in history['val_metrics']]
            axs[i].plot(epochs, train_metric, label=f'Training {metric.capitalize()}')
            axs[i].plot(epochs, val_metric, label=f'Validation {metric.capitalize()}')
            axs[i].set_title(f'Training and Validation {metric.capitalize()} - Fold {fold + 1}')
            axs[i].set_xlabel('Epochs')
            axs[i].set_ylabel(metric.capitalize())
            axs[i].legend()

        plt.tight_layout()
        plt.savefig(f"{log_dir}/{log_id}_fold_{fold + 1}_metrics.png")
        plt.close()

        # Plot training and validation loss
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, history['train_loss'], label='Training Loss')
        plt.plot(epochs, history['val_loss'], label='Validation Loss')
        plt.title(f'Training and Validation Loss - Fold {fold + 1}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(f"{log_dir}/{log_id}_fold_{fold + 1}_loss.png")
        plt.close()

    def k_fold_cv(self, log_id: str = "bert", log_dir: str = "") -> None:
        """
        Perform k-fold cross-validation.

        :param log_dir: The logging directory
        :param log_id: Identifier for logging purposes, typically the model name.
        """
        kf = KFold(n_splits=self.__num_folds, shuffle=True)
        fold_metrics = []

        # Iterate over each fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.__train_data)):
            # Create data loaders for training and validation sets
            train_subsampler = Subset(self.__train_data, train_idx)
            val_subsampler = Subset(self.__train_data, val_idx)

            train_loader = DataLoader(
                train_subsampler,
                sampler=RandomSampler(train_subsampler),
                batch_size=self.__batch_size
            )
            val_loader = DataLoader(
                val_subsampler,
                batch_size=self.__batch_size  # No need for shuffling
            )

            print(f"Starting Fold {fold + 1}/{self.__num_folds}")

            # Train and evaluate the model for the current fold
            history = self.__train_and_evaluate(train_loader, val_loader)

            # Plot and save metrics for the current fold
            self.__plot_metrics(history, fold, log_id, log_dir)

            # Evaluate on the test set after each fold
            metrics = self.__evaluate_on_test_set(
                DataLoader(self.__test_data, batch_size=self.__batch_size, shuffle=False))
            fold_metrics.append(metrics)

            # Reset the model to untrained
            self.__trainer.reset_model()

        # Calculate average and standard deviation of each metric across all folds
        metric_keys = fold_metrics[0].keys()  # Assuming all metrics dictionaries have the same structure
        average_metrics = {key: np.mean([metric[key] for metric in fold_metrics]) for key in metric_keys}
        std_dev_metrics = {key: np.std([metric[key] for metric in fold_metrics]) for key in metric_keys}

        # Print average metrics and their standard deviations
        print("Average Metrics Over All Folds:")
        for key, value in average_metrics.items():
            print(f"{key}: {value:.4f} (±{std_dev_metrics[key]:.4f})")

        # Save metrics to CSV file
        save_results(fold_metrics, filename=f"{log_id}.csv", log_dir=log_dir)
