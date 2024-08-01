import os
import random

import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

"""
DATASET     ASSESSMENTS
CodeSmells        10395
Zeus               7315
eThor               702
ContractFuzzer      367
SolidiFI            343
EverEvolvingG       292
Doublade            276
NPChecker           212
JiuZhou             165
SBcurated           129
SWCregistry         116
EthRacer            109
NotSoSmartC          34

Subset dataset to consider within CGT
"""
SUBSET = "Zeus"
print(f"Subset of CGT: {SUBSET}")


# File configurations
FILE_TYPE = "source"  # Can be 'source', 'runtime', 'bytecode'
FILE_EXT = None
if FILE_TYPE == "source":
    FILE_EXT = ".sol"
elif FILE_TYPE == "runtime":
    FILE_EXT = ".rt.hex"
elif FILE_TYPE == "bytecode":
    FILE_EXT = ".hex"
FILE_ID = "sol"  # Can be 'sol ,'sol2' for 'source', otherwise 'runtime', 'bytecode'

print(f"File type: {FILE_TYPE}")
print(f"File extension: {FILE_EXT}")
print(f"File ID: {FILE_ID}")

# Creating the log directory if it doesn't exist
LOG_DIR = os.path.join("log", FILE_TYPE)
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)
    print(f"Log directory created at {LOG_DIR}")
else:
    print(f"Log directory already exists at {LOG_DIR}")

# Setting the device for torch
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device {DEVICE}")

# Setting random seeds for reproducibility
RANDOM_SEED = 0
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.backends.cudnn.benchmark = False
print(f"Random seeds set to {RANDOM_SEED}")

# Path to dataset
PATH_TO_DATASET = os.path.join("..", "dataset", "cgt")
print(f"Path to dataset: {PATH_TO_DATASET}")

# Model and training configurations
BERT_MODEL_TYPE = 'microsoft/codebert-base'
print(f"BERT model type: {BERT_MODEL_TYPE}")

MAX_FEATURES = 500
print(f"Max features: {MAX_FEATURES}")

BATCH_SIZE = 1
print(f"Batch size: {BATCH_SIZE}")

NUM_FOLDS = 10
print(f"Number of folds: {NUM_FOLDS}")

NUM_EPOCHS = 25
print(f"Number of epochs: {NUM_EPOCHS}")

NUM_LABELS = 20  # 7 for Zeus, 20 for CodeSmell, 160 overall
print(f"Number of labels: {NUM_LABELS}")

LR = 0.001
print(f"Learning rate: {LR}")

TEST_SIZE = 0.1
print(f"Test size: {TEST_SIZE}")

VECTORIZER = TfidfVectorizer(max_features=MAX_FEATURES)

CLASSIFIERS = {
    "svm": SVC(kernel='linear', probability=True),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=LR, max_depth=3),
    "logistic_regression": LogisticRegression(),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}
