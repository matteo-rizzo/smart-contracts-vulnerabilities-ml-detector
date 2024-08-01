import os
import random
from typing import List, Any, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score

from src.settings import LOG_DIR


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


def save_results(results: List[Dict[str, Any]], filename: str) -> None:
    """
    Save the results to a CSV file.

    :param results: The results to save, typically a list of dictionaries.
    :param filename: The name of the file to save the results to.
    """
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(LOG_DIR, filename), index=False)
    print(f"All fold results saved to '{os.path.join(LOG_DIR, filename)}'")


def make_reproducible(random_seed: int):
    # Setting random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.backends.cudnn.benchmark = False


def get_file_ext(file_type: str) -> str:
    # Determine file extension based on file type
    if file_type == "source":
        file_ext = ".sol"
    elif file_type == "runtime":
        file_ext = ".rt.hex"
    elif file_type == "bytecode":
        file_ext = ".hex"
    else:
        file_ext = None
