import os

import torch

"""
CGT Repo and Paper: <https://github.com/gsalzer/cgt?tab=readme-ov-file>

| Set                | Entries | Total Assessments | Weaknesses |
|--------------------|---------|-------------------|------------|
| CodeSmells         | 587     | 11740             | 20         |
| ContractFuzzer     | 379     | 379               | 7          |
| Doublade           | 319     | 319               | 5          |
| eThor              | 720     | 720               | 1          |
| EthRacer           | 127     | 127               | 2          |
| EverEvolvingGame   | 344     | 344               | 5          |
| NPChecker          | 50      | 250               | 5          |
| Zeus               | 1524    | 10533             | 7          |
| JiuZhou            | 168     | 168               | 53         |
| NotSoSmartC        | 31      | 34                | 18         |
| SBcurated          | 143     | 145               | 10         |
| SolidiFI           | 350     | 350               | 7          |
| SWCregistry        | 117     | 117               | 33         |

"""

# Labels for the dataset
LABELS = {
    "CodeSmells": 20,
    "Zeus": 7,
    "eThor": 1,
    "ContractFuzzer": 7,
    "SolidiFI": 7,
    "EverEvolvingG": 5,
    "Doublade": 5,
    "NPChecker": 5,
    "JiuZhou": 53,
    "SBcurated": 10,
    "SWCregistry": 22,
    "EthRacer": 2,
    "NotSoSmartC": 18
}

# Set the device for torch (use GPU if available)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device {DEVICE}")

# Default configuration constants
RANDOM_SEED = 0
PATH_TO_DATASET = os.path.join("dataset", "cgt")
MAX_FEATURES = 256
BATCH_SIZE = 1
NUM_FOLDS = 2
NUM_EPOCHS = 2
LR = 0.001
TEST_SIZE = 0.1
FILE_TYPE = "source"
SUBSET = "CodeSmells"
