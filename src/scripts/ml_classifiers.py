from typing import Dict

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.classes.ClassifiersPoolEvaluator import ClassifiersPoolEvaluator
from src.classes.DataPreprocessor import DataPreprocessor
from src.settings import MAX_FEATURES
from src.utility import make_reproducible, get_file_ext, get_num_labels, init_arg_parser, get_file_id, make_log_dir

# Define classifiers to be used
CLASSIFIERS = {
    "svm": SVC(kernel='linear', probability=True),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.01, max_depth=3),
    "logistic_regression": LogisticRegression(),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "xgboost": XGBClassifier(eval_metric='mlogloss')
}


def main(config: Dict):
    # Initialize the DataPreprocessor
    print("Initializing DataPreprocessor...")
    preprocessor = DataPreprocessor(
        file_type=config['file_type'],
        path_to_dataset=config['path_to_dataset'],
        file_ext=config['file_ext'],
        file_id=config['file_id'],
        num_labels=config['num_labels'],
        subset=config['subset']
    )

    # Load and process the data
    print("Loading and processing data...")
    preprocessor.load_and_process_data()

    # Access processed data
    print("Accessing processed data...")
    inputs = preprocessor.get_inputs()
    labels = preprocessor.get_labels()

    # Initialize the TF-IDF vectorizer
    print("Initializing the TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=config["max_features"])

    # Initialize the ClassifiersPoolEvaluator
    print("Initializing ClassifiersPoolEvaluator...")
    evaluator = ClassifiersPoolEvaluator(
        inputs=inputs,
        labels=labels,
        classifiers=CLASSIFIERS,
        vectorizer=vectorizer,
        num_folds=config['num_folds'],
        random_seed=config['random_seed']
    )

    # Run the pool evaluation
    print("Starting pool evaluation...")
    evaluator.pool_evaluation(log_dir=config['log_dir'])


if __name__ == '__main__':
    # Initialize argument parser and add custom arguments
    parser = init_arg_parser()
    parser.add_argument("--max_features", type=int, default=MAX_FEATURES, help="Maximum features for TF-IDF vectorizer")

    # Parse command-line arguments
    args = parser.parse_args()
    config = vars(args)

    # Ensure reproducibility by setting the random seed
    make_reproducible(config["random_seed"])

    # Get the file extension based on the file type
    config["file_ext"] = get_file_ext(config["file_type"])

    # Get the file ID based on the file type
    config["file_id"] = get_file_id(config["file_type"])

    # Get the number of labels based on the subset of data to consider
    config["num_labels"] = get_num_labels(config["subset"])

    # Create the logging directory
    config["log_dir"] = make_log_dir(experiment_id=f"{config['subset']}_{config['file_type']}")

    # Print all configurations for verification
    print("Configuration:")
    for arg in config:
        print(f"{arg}: {getattr(args, arg)}")

    # Output the chosen classifiers for verification
    print("Classifiers initialized:")
    for name, clf in CLASSIFIERS.items():
        print(f" - {name}: {clf}")

    # Execute the main function with the provided configurations
    main(config)
