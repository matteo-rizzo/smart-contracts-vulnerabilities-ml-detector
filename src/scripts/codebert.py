import argparse
from typing import Dict

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from src.classes.BertModelTrainer import BERTModelTrainer
from src.classes.CrossValidator import CrossValidator
from src.classes.DataPreprocessor import DataPreprocessor
from src.settings import SUBSET, FILE_TYPE, FILE_ID, RANDOM_SEED, PATH_TO_DATASET, BERT_MODEL_TYPE, BATCH_SIZE, \
    NUM_FOLDS, NUM_EPOCHS, NUM_LABELS, LR, TEST_SIZE
from src.utility import make_reproducible, get_file_ext


def main(config: Dict):
    # Usage
    preprocessor = DataPreprocessor()
    preprocessor.load_and_process_data()

    # Access processed data using getters
    inputs = preprocessor.get_inputs()
    labels = preprocessor.get_labels()

    model = RobertaForSequenceClassification.from_pretrained(
        config['bert_model_type'],
        num_labels=config['num_labels'],
        ignore_mismatched_sizes=True
    )

    model.config.problem_type = "multi_label_classification"
    model.to(config['device'])

    tokenizer = RobertaTokenizer.from_pretrained(config['bert_model_type'], ignore_mismatched_sizes=True)

    x, y = tokenizer(
        inputs,
        add_special_tokens=True,
        max_length=512,
        return_token_type_ids=False,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    ), labels

    # Split the data into training and test sets
    x_train, x_test, y_train, y_test = train_test_split(x['input_ids'], y, test_size=config['test_size'])

    # Split attention masks for training and test sets
    train_masks, test_masks, _, _ = train_test_split(x['attention_mask'], y, test_size=config['test_size'])

    # Create datasets for training and testing
    train_data = TensorDataset(x_train, train_masks, torch.tensor(y_train).float())
    test_data = TensorDataset(x_test, test_masks, torch.tensor(y_test).float())
    CrossValidator(BERTModelTrainer(model), train_data, test_data).k_fold_cv(log_id="bert")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Set configurations for the CGT dataset processing.")

    parser.add_argument("--subset", type=str, default=SUBSET, help="Subset dataset to consider within CGT")
    parser.add_argument("--file_type", type=str, default=FILE_TYPE, choices=["source", "runtime", "bytecode"],
                        help="File type")
    parser.add_argument("--file_id", type=str, default=FILE_ID, help="File ID")
    parser.add_argument("--random_seed", type=int, default=RANDOM_SEED, help="Random seed for reproducibility")
    parser.add_argument("--path_to_dataset", type=str, default=PATH_TO_DATASET, help="Path to dataset")
    parser.add_argument("--bert_model_type", type=str, default=BERT_MODEL_TYPE, help="BERT model type")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_folds", type=int, default=NUM_FOLDS, help="Number of folds")
    parser.add_argument("--num_epochs", type=int, default=NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--num_labels", type=int, default=NUM_LABELS, help="Number of labels")
    parser.add_argument("--learning_rate", type=float, default=LR, help="Learning rate")
    parser.add_argument("--test_size", type=float, default=TEST_SIZE, help="Test size")

    args = parser.parse_args()
    config = vars(args)

    make_reproducible(config["random_seed"])
    config["file_ext"] = get_file_ext(config["file_type"])

    # Print all configurations by iterating over the parsed arguments
    print("Configurations:")
    for arg in config:
        print(f"{arg}: {getattr(args, arg)}")

    main(config)
