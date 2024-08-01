import os
from typing import Dict

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset

from src.classes.CrossValidator import CrossValidator
from src.classes.DataPreprocessor import DataPreprocessor
from src.classes.EmbeddingProcessor import EmbeddingProcessor
from src.classes.LSTMClassifier import LSTMClassifier
from src.classes.Trainer import Trainer
from src.settings import BATCH_SIZE, NUM_EPOCHS, LR
from src.utility import make_reproducible, get_file_ext, get_num_labels, init_arg_parser, get_file_id, make_log_dir

PATH_TO_GLOVE = os.path.join("asset", "glove.6B.100d.txt")


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

    # Initialize the EmbeddingProcessor with GloVe embeddings
    print("Initializing EmbeddingProcessor...")
    embedding_processor = EmbeddingProcessor(glove_file=config['path_to_glove'], embedding_dim=100)
    vocab_len, embed_dim, embed_matrix = embedding_processor.process_embeddings(inputs)
    sequences = embedding_processor.text_to_sequences(inputs)
    seq_padded = embedding_processor.pad_sequences(sequences)

    # Initialize the LSTMClassifier
    print("Initializing LSTMClassifier...")
    model = LSTMClassifier(
        vocab_size=vocab_len,
        embedding_dim=embed_dim,
        hidden_dim=128,
        pretrained_embeddings=embed_matrix,
        output_size=config["num_labels"]
    )

    # Convert the data to PyTorch tensors
    x = torch.FloatTensor(seq_padded)
    y = torch.FloatTensor(labels)

    # Split the data into training and test sets
    print("Splitting data into training and test sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=config['test_size'], random_state=config['random_seed']
    )
    train_data = TensorDataset(x_train, y_train)
    test_data = TensorDataset(x_test, y_test)

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
    cross_validator.k_fold_cv(log_id="lstm", log_dir=config['log_dir'])


if __name__ == '__main__':
    # Parse command-line arguments for configurations
    parser = init_arg_parser()
    parser.add_argument("--path_to_glove", type=str, default=PATH_TO_GLOVE, help="Path to Glove embeddings")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=LR, help="Learning rate")

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
