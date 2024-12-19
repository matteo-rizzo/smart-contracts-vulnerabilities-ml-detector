import os
import warnings

import nltk
import torch

nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# Set the device for torch (use GPU if available)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device {DEVICE}")

# Reproducibility
RANDOM_SEED = 0

# Default configuration constants
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

# Explainability
LLM_MODE = "azure"
EMBEDDING_MODE = "local"
