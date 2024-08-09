from typing import Dict

from src.classes.classifiers.GCNClassifier import GCNClassifier
from src.classes.data.DataPreprocessor import DataPreprocessor
from src.classes.data.GraphDataset import GraphDataset
from src.classes.training.ClassBalancer import ClassBalancer
from src.classes.training.CrossValidator import CrossValidator
from src.classes.training.GCNTrainer import GCNTrainer
from src.settings import BATCH_SIZE, NUM_EPOCHS, LR
from src.utility import make_reproducible, init_arg_parser, make_log_dir


def main(config: Dict):
    # Initialize the DataPreprocessor
    print("Initializing DataPreprocessor...")
    preprocessor = DataPreprocessor(
        path_to_dataset=config['path_to_dataset'],
        file_types=[config['file_type']],
        subset=config['subset']
    )

    print("Loading and processing data...")
    result = preprocessor.load_and_process_data()
    if result is None:
        raise ValueError("No graphs or labels found in the dataset.")

    graphs, labels = result

    print(f"Number of graphs: {len(graphs)}")
    print(f"Number of labels: {len(labels)}")
    print(f"Example labels: {labels[0]}")

    print("Initializing GCNClassifier model...")
    model = GCNClassifier(
        input_size=graphs[0].x.shape[1],
        hidden_size=128,
        output_size=preprocessor.get_num_labels()
    )

    print("Splitting data into training and test sets...")
    x_train, x_test, _, _ = ClassBalancer.train_test_split(
        graphs, labels, test_size=config['test_size'], random_state=config['random_seed']
    )

    print("Creating GraphDataset objects for training and testing...")
    train_dataset = GraphDataset(x_train)
    test_dataset = GraphDataset(x_test)

    print("Initializing CrossValidator...")
    cross_validator = CrossValidator(
        GCNTrainer(model), train_dataset, test_dataset,
        num_epochs=config['num_epochs'],
        num_folds=config['num_folds'],
        batch_size=config['batch_size']
    )

    print("Starting k-fold cross-validation...")
    cross_validator.k_fold_cv(log_id="gcn", log_dir=config["log_dir"], use_class_weights=False)


if __name__ == '__main__':
    parser = init_arg_parser()
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=LR, help="Learning rate")

    args = parser.parse_args()
    config = vars(args)

    make_reproducible(config["random_seed"])
    config["log_dir"] = make_log_dir(experiment_id=f"{config['subset']}_{config['file_type']}")

    print("Configuration:")
    for arg in config:
        print(f"{arg}: {getattr(args, arg)}")

    main(config)
