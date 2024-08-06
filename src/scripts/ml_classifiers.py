from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.classes.ClassifiersPoolEvaluator import ClassifiersPoolEvaluator
from src.classes.DataPreprocessor import DataPreprocessor
from src.classes.MultimodalVectorizer import MultimodalVectorizer
from src.settings import MAX_FEATURES
from src.utility import make_reproducible, init_arg_parser, make_log_dir

MULTIMODAL = True
MULTIMODAL_FILE_TYPES = ["source", "bytecode"]

"""
Define classifiers to be used with optimized parameters:

1. Support Vector Machine (SVM):
   - kernel: 'linear' - a linear kernel is chosen for simplicity and effectiveness in high-dimensional spaces.
   - probability: True - enables probability estimates.
   - C: 1.0 - regularization parameter, the trade-off between achieving a low training error and a low testing error.
   - gamma: 'scale' - kernel coefficient for 'rbf', 'poly', and 'sigmoid'.
   - random_state: config["random_seed"] - ensures reproducibility.

2. Random Forest:
   - n_estimators: 200 - the number of trees in the forest.
   - max_depth: 10 - the maximum depth of the tree.
   - min_samples_split: 5 - the minimum number of samples required to split an internal node.
   - min_samples_leaf: 2 - the minimum number of samples required to be at a leaf node.
   - max_features: 'sqrt' - the number of features to consider when looking for the best split.
   - random_state: config["random_seed"] - ensures reproducibility.

3. Gradient Boosting:
   - n_estimators: 150 - the number of boosting stages to be run.
   - learning_rate: 0.05 - shrinks the contribution of each tree by this value.
   - max_depth: 5 - the maximum depth of the individual regression estimators.
   - min_samples_split: 2 - the minimum number of samples required to split an internal node.
   - min_samples_leaf: 1 - the minimum number of samples required to be at a leaf node.
   - random_state: config["random_seed"] - ensures reproducibility.

4. Logistic Regression:
   - C: 1.0 - inverse of regularization strength.
   - solver: 'liblinear' - algorithm to use in the optimization problem.
   - max_iter: 100 - maximum number of iterations taken for the solvers to converge.
   - random_state: config["random_seed"] - ensures reproducibility.

5. K-Nearest Neighbors (KNN):
   - n_neighbors: 10 - the number of neighbors to use.
   - weights: 'distance' - weight points by the inverse of their distance.
   - metric: 'minkowski' - the distance metric to use.

6. XGBoost:
   - eval_metric: 'mlogloss' - evaluation metric for the training data.
   - use_label_encoder: False - avoid label encoding warnings.
   - n_estimators: 150 - the number of gradient boosted trees.
   - max_depth: 6 - maximum depth of a tree.
   - learning_rate: 0.1 - step size shrinkage.
   - colsample_bytree: 0.8 - subsample ratio of columns when constructing each tree.
   - subsample: 0.8 - subsample ratio of the training instances.
   - random_state: config["random_seed"] - ensures reproducibility.
"""


def initialize_classifiers(random_seed: int) -> Dict[str, object]:
    """
    Initializes classifiers with the given random seed.

    :param random_seed: The random seed for reproducibility.
    :return: A dictionary of classifier instances.
    """
    return {
        "svm": SVC(kernel='linear', probability=True, C=1.0, gamma='scale', random_state=random_seed),
        "logistic_regression": LogisticRegression(C=1.0, solver='liblinear', max_iter=100, random_state=random_seed),
        "knn": KNeighborsClassifier(n_neighbors=10, weights='distance', metric='minkowski'),
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, max_features='sqrt', random_state=random_seed
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5,
            min_samples_split=2, min_samples_leaf=1, random_state=random_seed
        ),
        "xgboost": XGBClassifier(
            eval_metric='mlogloss', use_label_encoder=False, n_estimators=150,
            max_depth=6, learning_rate=0.1, colsample_bytree=0.8,
            subsample=0.8, random_state=random_seed
        )
    }


def main(config: Dict):
    preprocessor = DataPreprocessor(
        path_to_dataset=config['path_to_dataset'],
        file_types=MULTIMODAL_FILE_TYPES if config["multimodal"] else [config['file_type']],
        subset=config['subset']
    )
    print("Loading and processing data...")
    preprocessor.load_and_process_data()

    vectorizer = MultimodalVectorizer(max_features=config["max_features"], multimodal=config["multimodal"])

    print("Transforming input documents into TF-IDF features...")
    x = vectorizer.transform_inputs(preprocessor.get_inputs())
    print(f"TF-IDF feature matrix shape: {x.shape}")

    print("Converting labels to numpy array...")
    y = np.array(preprocessor.get_labels())
    print(f"Labels shape: {y.shape}")

    evaluator = ClassifiersPoolEvaluator(
        inputs=x, labels=y,
        classifiers=initialize_classifiers(config['random_seed']),
        num_folds=config['num_folds'],
        random_seed=config['random_seed']
    )
    print("Starting pool evaluation...")
    evaluator.pool_evaluation(log_dir=config['log_dir'])


if __name__ == '__main__':
    parser = init_arg_parser()
    parser.add_argument("--max_features", type=int, default=MAX_FEATURES, help="Maximum features for TF-IDF vectorizer")
    parser.add_argument("--multimodal", type=bool, default=MULTIMODAL, help="Process multiple modalities at once")

    args = parser.parse_args()
    config = vars(args)

    make_reproducible(config["random_seed"])

    experiment_id = f"{config['subset']}_{'multimodal' if config['multimodal'] else config['file_type']}"
    config["log_dir"] = make_log_dir(experiment_id)

    print("Configuration:")
    for arg, value in config.items():
        print(f"{arg}: {value}")

    print("Classifiers initialized:")
    classifiers = initialize_classifiers(config["random_seed"])
    for name, clf in classifiers.items():
        print(f" - {name}: {clf}")

    main(config)
