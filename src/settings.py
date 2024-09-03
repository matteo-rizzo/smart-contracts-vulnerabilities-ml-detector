import os
import warnings

import torch


def warn(*args, **kwargs):
    pass


warnings.warn = warn

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

# Set the device for torch (use GPU if available)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device {DEVICE}")

# Default configuration constants
RANDOM_SEED = 0
PATH_TO_DATASET = os.path.join("dataset", "cgt")
MAX_FEATURES = 256
BATCH_SIZE = 1
NUM_FOLDS = 10
NUM_EPOCHS = 100
LR = 0.00001
TEST_SIZE = 0.1
FILE_TYPE = "cfg"
SUBSET = "CodeSmells"
