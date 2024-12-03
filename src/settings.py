import os
import warnings

import torch


def warn(*args, **kwargs):
    pass


warnings.warn = warn

"""
CGT Repo and Paper: <https://github.com/gsalzer/cgt?tab=readme-ov-file>
"""

# Set the device for torch (use GPU if available)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device {DEVICE}")

# Default configuration constants
RANDOM_SEED = 0
PATH_TO_DATASET = os.path.join("dataset", "manually-verified")
DATASET_NAME = "manually_verified.csv"
MAX_FEATURES = 128
PCA_COMPONENTS = 10
USE_CLASS_WEIGHTS = False
BATCH_SIZE = 32
NUM_FOLDS = 5
NUM_EPOCHS = 25
LR = 0.0001
TEST_SIZE = 0.1
FILE_TYPE = "source"
SUBSET = ""
LABEL_TYPE = "property"
