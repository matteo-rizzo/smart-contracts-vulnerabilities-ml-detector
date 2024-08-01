import os
import re
from typing import Dict

import pandas as pd
from tqdm import tqdm

from src.settings import FILE_TYPE, PATH_TO_DATASET, FILE_EXT, FILE_ID, NUM_LABELS, SUBSET


class DataPreprocessor:
    def __init__(self):
        """
        Initialize the DataPreprocessor with necessary configurations.
        """
        self.inputs = {}
        self.labels = {}
        self.gt = {}

    @staticmethod
    def preprocess_hex(hex_data: str) -> str:
        """
        Convert hex data to a readable ASCII string.

        :param hex_data: Hexadecimal string to preprocess.
        :return: Preprocessed string in ASCII format.
        """
        byte_data = bytes.fromhex(hex_data.strip())
        return ' '.join(f'{byte:02x}' for byte in byte_data)

    @staticmethod
    def preprocess_solidity_code(code: str) -> str:
        """
        Preprocess Solidity code by removing comments and blank lines.

        :param code: Solidity code string to preprocess.
        :return: Preprocessed Solidity code.
        """
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        lines = code.split('\n')
        non_blank_lines = [line for line in lines if line.strip() != '']
        return '\n'.join(non_blank_lines)

    def preprocess(self, data: str) -> str:
        """
        Preprocess data based on the file type.

        :param data: String data to preprocess.
        :return: Preprocessed data.
        """
        if FILE_TYPE == "source":
            return self.preprocess_solidity_code(data)
        return self.preprocess_hex(data)

    @staticmethod
    def _load_file(file_id: str) -> str:
        """
        Load the content of a file given its ID.

        :param file_id: ID of the file to load.
        :return: Content of the file as a string.
        """
        path_to_file = os.path.join(PATH_TO_DATASET, FILE_TYPE, str(file_id) + FILE_EXT)
        if os.path.exists(path_to_file):
            with open(path_to_file, 'r', encoding="utf8") as file:
                return file.read()
        return ""

    def init_inputs_and_gt(self, data: pd.DataFrame) -> None:
        """
        Initialize inputs and ground truth (gt) from the dataset.

        :param data: A pandas DataFrame containing the dataset.
        """
        for _, row in tqdm(data.iterrows(), desc="Initializing inputs and groundtruth data"):
            item_id, file_id = row["id"], row["fp_" + FILE_ID]
            file_content = self._load_file(file_id)
            if file_content:
                self.inputs[item_id] = self.preprocess(file_content)
                self.labels[item_id] = [0] * NUM_LABELS
                prop = row["property"].lower()
                if prop not in self.gt:
                    self.gt[prop] = len(self.gt)

    def set_labels(self, data: pd.DataFrame) -> None:
        """
        Set labels for the dataset based on ground truth (gt).

        :param data: A pandas DataFrame containing the dataset.
        """
        for _, row in tqdm(data.iterrows(), desc="Setting up the labels"):
            item_id, file_id = row["id"], row["fp_" + FILE_ID]
            if self._load_file(file_id):
                prop = row["property"].lower()
                if row['property_holds'] == 't':
                    self.labels[item_id][self.gt[prop]] = 1

    def load_and_process_data(self) -> None:
        """
        Load and preprocess the dataset, initializing the TF-IDF vectorizer.
        """
        dataset = pd.read_csv(os.path.join(PATH_TO_DATASET, "consolidated.csv"), sep=";")
        dataset = dataset[dataset["dataset"] == SUBSET]

        self.init_inputs_and_gt(dataset)
        self.set_labels(dataset)
        print("Initialization complete!")

    def get_inputs(self) -> Dict:
        """
        Get the processed inputs.

        :return: Dictionary of processed inputs.
        """
        return self.inputs

    def get_labels(self) -> Dict:
        """
        Get the labels.

        :return: Dictionary of labels.
        """
        return self.labels


if __name__ == '__main__':
    # Usage
    preprocessor = DataPreprocessor()
    preprocessor.load_and_process_data()

    # Access processed data using getters
    INPUTS = preprocessor.get_inputs()
    LABELS = preprocessor.get_labels()
