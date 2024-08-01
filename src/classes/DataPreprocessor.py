import os
import re
from typing import List

import pandas as pd
from tqdm import tqdm


class DataPreprocessor:
    def __init__(self, file_type: str, path_to_dataset: str, file_ext: str, file_id: str, num_labels: int, subset: str):
        """
        Initialize the DataPreprocessor with necessary configurations.

        :param file_type: Type of the file (e.g., "source" or other).
        :param path_to_dataset: Path to the dataset directory.
        :param file_ext: File extension (e.g., ".sol" or ".hex").
        :param file_id: Identifier for the file.
        :param num_labels: Number of labels.
        :param subset: Subset of the dataset to process.
        """
        self.file_type = file_type
        self.path_to_dataset = path_to_dataset
        self.file_ext = file_ext
        self.file_id = file_id
        self.num_labels = num_labels
        self.subset = subset
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
        if self.file_type == "source":
            return self.preprocess_solidity_code(data)
        return self.preprocess_hex(data)

    def _load_file(self, file_id: str) -> str:
        """
        Load the content of a file given its ID.

        :param file_id: ID of the file to load.
        :return: Content of the file as a string.
        """
        path_to_file = os.path.join(self.path_to_dataset, self.file_type, str(file_id) + self.file_ext)
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
            item_id, file_id = row["id"], row["fp_" + self.file_id]
            file_content = self._load_file(file_id)
            if file_content:
                self.inputs[item_id] = self.preprocess(file_content)
                self.labels[item_id] = [0] * self.num_labels
                prop = row["property"].lower()
                if prop not in self.gt:
                    self.gt[prop] = len(self.gt)

    def set_labels(self, data: pd.DataFrame) -> None:
        """
        Set labels for the dataset based on ground truth (gt).

        :param data: A pandas DataFrame containing the dataset.
        """
        for _, row in tqdm(data.iterrows(), desc="Setting up the labels"):
            item_id, file_id = row["id"], row["fp_" + self.file_id]
            if self._load_file(file_id):
                prop = row["property"].lower()
                if row['property_holds'] == 't':
                    self.labels[item_id][self.gt[prop]] = 1

    def load_and_process_data(self) -> None:
        """
        Load and preprocess the dataset, initializing the TF-IDF vectorizer.
        """
        dataset = pd.read_csv(os.path.join(self.path_to_dataset, "consolidated.csv"), sep=";")
        dataset = dataset[dataset["dataset"] == self.subset]

        self.init_inputs_and_gt(dataset)
        self.set_labels(dataset)
        print("Initialization complete!")

    def get_inputs(self) -> List[str]:
        """
        Get the processed inputs.

        :return: List of processed inputs.
        """
        return list(self.inputs.values())

    def get_labels(self) -> List[List[int]]:
        """
        Get the labels.

        :return: List of labels.
        """
        return list(self.labels.values())
