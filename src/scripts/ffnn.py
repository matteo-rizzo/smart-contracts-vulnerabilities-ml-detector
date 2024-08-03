from typing import Dict

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import TensorDataset

from src.classes.ClassBalancer import ClassBalancer
from src.classes.CrossValidator import CrossValidator
from src.classes.DataPreprocessor import DataPreprocessor
from src.classes.FFNNClassifier import FFNNClassifier
from src.classes.Trainer import Trainer
from src.settings import BATCH_SIZE, NUM_EPOCHS, LR, MAX_FEATURES
from src.utility import make_reproducible, get_file_ext, get_num_labels, init_arg_parser, make_log_dir, get_file_id


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

    # Initialize the FFNNClassifier
    print("Initializing the FFNNClassifier...")
    model = FFNNClassifier(input_size=config["max_features"], output_size=config["num_labels"])

    # Initialize the TF-IDF vectorizer
    print("Initializing the TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=config["max_features"])

    # Convert the data to PyTorch tensors
    x = vectorizer.fit_transform(inputs).toarray()
    y = np.array(labels)

    # Split the data into training and test sets using ClassBalancer
    print("Splitting data into training and test sets...")
    x_train, x_test, y_train, y_test = ClassBalancer.train_test_split(
        x, y, test_size=config['test_size'], random_state=config['random_seed']
    )

    train_data = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32))

    # Start cross-validation
    print("Starting cross-validation...")
    cross_validator = CrossValidator(
        Trainer(model),
        train_data,
        test_data,
        num_epochs=config['num_epochs'],
        num_folds=config['num_folds'],
        batch_size=config['batch_size']
    )

    print("Starting k-fold cross-validation...")
    cross_validator.k_fold_cv(log_id="ffnn", log_dir=config["log_dir"])


if __name__ == '__main__':
    # Initialize argument parser and add custom arguments
    parser = init_arg_parser()
    parser.add_argument("--max_features", type=int, default=MAX_FEATURES, help="Maximum features for TF-IDF vectorizer")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=LR, help="Learning rate")

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

    # Execute the main function with the provided configurations
    main(config)
