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
PATH_TO_DATASET = os.path.join("dataset", "cgt")
MAX_FEATURES = 256
PCA_COMPONENTS = 10
BATCH_SIZE = 1
NUM_FOLDS = 5
NUM_EPOCHS = 25
LR = 0.001
TEST_SIZE = 0.1
FILE_TYPE = "source"
SUBSET = "CodeSmells"
LABEL_TYPE = "property"
