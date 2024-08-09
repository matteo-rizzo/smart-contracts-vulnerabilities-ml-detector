from typing import Tuple, Dict, Any

import numpy as np
import torch
from numpy import floating
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

from src.classes.training.MetricsHandler import MetricsHandler
from src.classes.training.Trainer import Trainer
from src.settings import DEVICE


class GCNTrainer(Trainer):
    """
    Trainer class for handling the training and evaluation of a Graph Convolutional Network (GCN).
    """

    def _evaluate_batch(self, batch: Data) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate a single batch of graph data.

        :param batch: A Data object containing the batch of graphs.
        :return: A tuple containing the loss and a dictionary of metrics.
        """
        # Move batch to the appropriate device (CPU/GPU)
        batch = batch.to(DEVICE)

        # Prepare inputs for the model
        labels = batch.y

        # Disable gradient computation for evaluation
        with torch.no_grad():
            outputs = self._model(batch)

            # Compute the loss
            loss = self._loss_fn(outputs, labels)

            # Convert outputs to binary predictions
            predictions = (outputs > 0).float().cpu().numpy()

            # Ensure labels are in the correct format
            true_labels = labels.cpu().numpy()

            # Compute batch metrics
            batch_metrics = MetricsHandler.compute_metrics(true_labels, predictions)

        # Return the loss and metrics
        return loss.item(), batch_metrics

    def _train_batch(self, batch: Data) -> Tuple[float, Dict[str, float]]:
        """
        Train a single batch of graph data.

        :param batch: A Data object containing the batch of graphs.
        :return: A tuple containing the loss and a dictionary of metrics.
        """
        # Move batch to the appropriate device (CPU/GPU)
        batch = batch.to(DEVICE)

        # Prepare inputs for the model
        labels = batch.y

        # Zero the parameter gradients
        self._model.zero_grad()

        # Forward pass
        outputs = self._model(batch)

        # Compute the loss
        loss = self._loss_fn(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        self._optimizer.step()

        # Convert outputs to binary predictions
        predictions = (outputs > 0).float().detach().cpu().numpy()

        # Ensure labels are in the correct format
        true_labels = labels.detach().cpu().numpy()

        # Compute batch metrics
        batch_metrics = MetricsHandler.compute_metrics(true_labels, predictions)

        return loss.item(), batch_metrics

    def run_epoch(self, dataloader: DataLoader, train_mode: bool = True) -> tuple[
        floating[Any], dict[str, floating[Any]]]:
        """
        Run a single epoch of training or evaluation.

        :param dataloader: DataLoader providing the data for the epoch.
        :param train_mode: Boolean flag indicating whether to train or evaluate.
        :return: A tuple containing the average loss and a dictionary of average metrics.
        """
        # Set the mode for the epoch (Training or Testing)
        phase = 'Training' if train_mode else 'Testing'
        self._model.train() if train_mode else self._model.eval()

        losses, metrics_list = [], []

        # Iterate over the data loader
        for batch in tqdm(dataloader, desc=phase):
            loss, batch_metrics = self._train_batch(batch) if train_mode else self._evaluate_batch(batch)

            # Accumulate the loss and metrics
            losses.append(loss)
            metrics_list.append(batch_metrics)

        # Compute average loss and metrics for the epoch
        avg_loss = np.mean(losses)
        avg_metrics = {metric: np.mean([m[metric] for m in metrics_list]) for metric in metrics_list[0]}

        return avg_loss, avg_metrics
