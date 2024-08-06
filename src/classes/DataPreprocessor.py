import os
import re
from typing import List, Dict, Union

import pandas as pd
from tqdm import tqdm

from src.utility import get_file_config


class DataPreprocessor:
    def __init__(self, path_to_dataset: str, file_types: List[str], subset: str):
        """
        Initialize the DataPreprocessor with necessary configurations for multiple file types.

        :param path_to_dataset: Path to the dataset directory.
        :param file_types: List of file types (e.g., ["sol", "bytecode", "runtime"]).
        :param subset: Subset of the dataset to process.
        """
        self.__path_to_dataset = path_to_dataset
        self.__file_types = file_types
        self.__file_configs = {file_type: get_file_config(file_type) for file_type in file_types}
        self.__subset = subset
        self.__inputs = {file_type: [] for file_type in file_types}
        self.__labels = []
        self.__gt = {}
        self.__item_ids = []

    @staticmethod
    def __preprocess_hex(hex_data: str) -> str:
        """
        Convert hex data to a readable ASCII string.

        :param hex_data: Hexadecimal string.
        :return: Preprocessed string.
        """
        byte_data = bytes.fromhex(hex_data.strip())
        return ' '.join(f'{byte:02x}' for byte in byte_data)

    @staticmethod
    def __preprocess_solidity_code(code: str) -> str:
        """
        Preprocess Solidity code by removing comments and blank lines.

        :param code: Solidity code as a string.
        :return: Preprocessed Solidity code.
        """
        code = re.sub(r'//.*', '', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        lines = code.split('\n')
        non_blank_lines = [line for line in lines if line.strip() != '']
        return '\n'.join(non_blank_lines)

    def __preprocess(self, data: str, file_type: str) -> str:
        """
        Preprocess data based on the file type.

        :param data: Data as a string.
        :param file_type: Type of file (e.g., "sol", "bytecode").
        :return: Preprocessed data.
        """
        file_config = self.__file_configs[file_type]
        if file_config["type"] == "source":
            return self.__preprocess_solidity_code(data)
        return self.__preprocess_hex(data)

    def __load_file(self, file_id: str, file_type: str) -> str:
        """
        Load the content of a file given its ID and type.

        :param file_id: ID of the file.
        :param file_type: Type of file (e.g., "sol", "bytecode").
        :return: Content of the file.
        """
        file_config = self.__file_configs[file_type]
        path_to_file = os.path.join(self.__path_to_dataset, file_config["type"], f"{file_id}{file_config['ext']}")
        if os.path.exists(path_to_file):
            with open(path_to_file, 'r', encoding="utf8") as file:
                return file.read()
        return ""

    def __initialize_label_mapping(self, data: pd.DataFrame) -> int:
        """
        Initialize the label mapping for the properties.

        :param data: DataFrame containing the data.
        :return: Number of unique labels.
        """
        unique_properties = data['property'].str.lower().unique()
        self.__gt = {prop: idx for idx, prop in enumerate(unique_properties)}
        return len(self.__gt)

    def __process_file_contents(self, group: pd.DataFrame) -> Union[Dict[str, str], bool]:
        """
        Process the file contents for a given group.

        :param group: DataFrame group containing the files.
        :return: Dictionary of file contents or False if a file is missing.
        """
        file_contents = {}
        for file_type in self.__file_configs:
            file_col = f"fp_{self.__file_configs[file_type]['id']}"
            file_id = group.iloc[0][file_col]  # Assuming the same file ID for all properties of the same item
            file_content = self.__load_file(file_id, file_type)
            if not file_content:
                return False
            file_contents[file_type] = self.__preprocess(file_content, file_type)
        return file_contents

    def __initialize_labels(self, group: pd.DataFrame, num_labels: int) -> List[int]:
        """
        Initialize labels for a given group.

        :param group: DataFrame group containing the properties.
        :param num_labels: Number of unique labels.
        :return: List of initialized labels.
        """
        labels = [0] * num_labels
        for _, row in group.iterrows():
            prop = row["property"].lower()
            if row['property_holds'] == 't':
                labels[self.__gt[prop]] = 1
        return labels

    def __filter_and_preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out rows with NaN values, empty files, or where 'property_holds' is 'f'.
        Preprocess file contents and initialize labels.

        :param data: DataFrame containing the data to be processed.
        :return: DataFrame after filtering and preprocessing.
        """
        # Drop rows with NaN in any of the file columns
        for file_type in self.__file_configs:
            file_col = f"fp_{self.__file_configs[file_type]['id']}"
            data = data.dropna(subset=[file_col])

        num_labels = self.__initialize_label_mapping(data)

        # Group data by id
        grouped_data = data.groupby('id')

        valid_rows = []
        for item_id, group in tqdm(grouped_data, desc="Processing items"):
            file_contents = self.__process_file_contents(group)
            if not file_contents:
                continue

            for file_type, content in file_contents.items():
                self.__inputs[file_type].append(content)

            labels = self.__initialize_labels(group, num_labels)
            self.__labels.append(labels)
            self.__item_ids.append(item_id)
            valid_rows.append(group)

        return pd.concat(valid_rows)

    def load_and_process_data(self) -> None:
        """
        Load and preprocess the dataset, filtering NaNs if specified, and initializing inputs and labels.
        """
        dataset = pd.read_csv(os.path.join(self.__path_to_dataset, "consolidated.csv"), sep=";")
        dataset = dataset[dataset["dataset"] == self.__subset]

        processed_data = self.__filter_and_preprocess_data(dataset)
        print(f"Processed data shape: {processed_data.shape}")
        print(f"Number of samples: {len(self.__labels)}")
        print(f"Sample labels: {self.__labels[:5]}")
        print("Initialization complete!")

    def get_inputs(self) -> Union[List[str], Dict[str, List[str]]]:
        """
        Get the processed inputs.

        :return: Processed inputs.
        """
        if len(self.__file_types) == 1:
            return self.__inputs[self.__file_types[0]]
        return self.__inputs

    def get_labels(self) -> List[List[int]]:
        """
        Get the labels.

        :return: List of labels.
        """
        return self.__labels
