import argparse
import os
import random
from argparse import ArgumentParser
from typing import List, Any, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from src.settings import LABELS, SUBSET, FILE_TYPE, RANDOM_SEED, PATH_TO_DATASET, NUM_FOLDS, TEST_SIZE


def compute_metrics(true_labels: List[Any], pred_labels: List[Any]) -> Dict[str, float]:
    """
    Compute evaluation metrics for the given true and predicted labels.

    :param true_labels: The ground truth labels.
    :param pred_labels: The predicted labels.
    :return: A dictionary containing precision, recall, and F1 score.
    """
    return {
        "precision": precision_score(true_labels, pred_labels, average='samples', zero_division=0),
        "recall": recall_score(true_labels, pred_labels, average='samples', zero_division=0),
        "f1": f1_score(true_labels, pred_labels, average='samples', zero_division=0)
    }


def save_results(results: List[Dict[str, Any]], filename: str, log_dir: str) -> None:
    """
    Save the results to a CSV file.

    :param log_dir: The logging directory.
    :param results: The results to save, typically a list of dictionaries.
    :param filename: The name of the file to save the results to.
    """
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(log_dir, filename), index=False)
    print(f"All fold results saved to '{os.path.join(log_dir, filename)}'")


def make_reproducible(random_seed: int):
    """
    Set random seeds for reproducibility.

    :param random_seed: The seed to use for random number generation.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.benchmark = False


def get_file_ext(file_type: str) -> str:
    """
    Determine file extension based on file type.

    :param file_type: The type of file.
    :return: The file extension corresponding to the file type.
    :raises ValueError: If the file type is not supported.
    """
    if file_type == "source":
        return ".sol"
    elif file_type == "runtime":
        return ".rt.hex"
    elif file_type == "bytecode":
        return ".hex"
    else:
        raise ValueError(f"File type '{file_type}' has no supported file extension!")


def get_file_id(file_type: str) -> str:
    """
    Determine file id based on file type.

    :param file_type: The type of file.
    :return: The file id corresponding to the file type.
    :raises ValueError: If the file type is not supported.
    """
    if file_type == "source":
        return "sol"
    elif file_type == "runtime":
        return "runtime"
    elif file_type == "bytecode":
        return "bytecode"
    else:
        raise ValueError(f"File type '{file_type}' has no supported file ID!")


def get_num_labels(dataset: str) -> int:
    """
    Get the number of labels for the given dataset.

    :param dataset: The name of the dataset.
    :return: The number of labels in the dataset.
    """
    return LABELS[dataset]


def init_arg_parser() -> ArgumentParser:
    """
    Initialize and return an argument parser.

    :return: The argument parser with configured arguments.
    """
    parser = argparse.ArgumentParser(description="Set configurations for the CGT dataset processing.")
    parser.add_argument("--subset", type=str, default=SUBSET, help="Subset dataset to consider within CGT")
    parser.add_argument("--file_type", type=str, default=FILE_TYPE, help="File type",
                        choices=["source", "runtime", "bytecode"])
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--path_to_dataset", type=str, default=PATH_TO_DATASET, help="Path to dataset")
    parser.add_argument("--num_folds", type=int, default=NUM_FOLDS, help="Number of folds")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE, help="Test size")
    return parser


def make_log_dir(experiment_id: str):
    """
    Create the log directory if it doesn't exist.

    :param experiment_id: The ID of the experiment.
    :return: The path to the log directory.
    """
    log_dir = os.path.join("log", f"experiment_{experiment_id}")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        print(f"Log directory created at {log_dir}")
    else:
        print(f"Log directory already exists at {log_dir}")
    return log_dir
