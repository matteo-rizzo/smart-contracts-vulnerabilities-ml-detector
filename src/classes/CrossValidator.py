import numpy as np
from sklearn.model_selection import KFold
from torch.utils.data import TensorDataset, DataLoader, Subset, RandomSampler

from src.classes.Trainer import Trainer
from src.settings import NUM_EPOCHS, NUM_FOLDS, BATCH_SIZE
from src.utility import save_results


class CrossValidator:
    """
    CrossValidator class for handling k-fold cross-validation of a model.
    """

    def __init__(self, trainer: Trainer, train_data: TensorDataset, test_data: TensorDataset):
        """
        Initialize the CrossValidator with trainer, training data, and test data.

        :param trainer: An instance of the Trainer class.
        :param train_data: The training dataset.
        :param test_data: The test dataset.
        """
        self.__trainer = trainer
        self.__train_data = train_data
        self.__test_data = test_data

    def __train_and_evaluate(self, train_dataloader: DataLoader, test_dataloader: DataLoader) -> None:
        """
        Train and evaluate the model for a specified number of epochs.

        :param train_dataloader: DataLoader for the training data.
        :param test_dataloader: DataLoader for the validation data.
        """
        for epoch in range(NUM_EPOCHS):
            print(f"\n --- Epoch {epoch + 1}/{NUM_EPOCHS} ---")

            # Train the model and print training metrics
            avg_train_loss, avg_train_metrics = self.__trainer.run_epoch(train_dataloader, train_mode=True)
            print(f"\n TRAIN | Loss: {avg_train_loss:.4f} |"
                  f" Precision: {avg_train_metrics['precision']:.4f},"
                  f" Recall: {avg_train_metrics['recall']:.4f},"
                  f" F1: {avg_train_metrics['f1']:.4f}\n")

            # Evaluate the model on the validation set and print validation metrics
            avg_test_loss, avg_test_metrics = self.__trainer.run_epoch(test_dataloader, train_mode=False)
            print(f" VALID | Loss: {avg_test_loss:.4f} |"
                  f" Precision: {avg_test_metrics['precision']:.4f},"
                  f" Recall: {avg_test_metrics['recall']:.4f},"
                  f" F1: {avg_test_metrics['f1']:.4f}\n")

    def __evaluate_on_test_set(self, test_dataloader: DataLoader) -> Dict[str, float]:
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

    def k_fold_cv(self, log_id: str = "bert") -> None:
        """
        Perform k-fold cross-validation.

        :param log_id: Identifier for logging purposes, typically the model name.
        """
        kf = KFold(n_splits=NUM_FOLDS, shuffle=True)
        fold_metrics = []

        # Iterate over each fold
        for fold, (train_idx, val_idx) in enumerate(kf.split(self.__train_data)):
            # Create data loaders for training and validation sets
            train_subsampler = Subset(self.__train_data, train_idx)
            val_subsampler = Subset(self.__train_data, val_idx)

            train_loader = DataLoader(
                train_subsampler,
                sampler=RandomSampler(train_subsampler),
                batch_size=BATCH_SIZE
            )
            val_loader = DataLoader(
                val_subsampler,
                batch_size=BATCH_SIZE  # No need for shuffling
            )

            print(f"Starting Fold {fold + 1}/{NUM_FOLDS}")

            # Train and evaluate the model for the current fold
            self.__train_and_evaluate(train_loader, val_loader)

            # Evaluate on the test set after each fold
            metrics = self.__evaluate_on_test_set(DataLoader(self.__test_data, batch_size=BATCH_SIZE, shuffle=False))
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
        save_results(fold_metrics, filename=f"{log_id}.csv")
