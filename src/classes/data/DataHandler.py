from typing import List, Dict, Union

import torch
from torch_geometric.data import Data


class DataHandler:
    """
    Class to manage the addition of processed data and labels, and provides access to inputs and labels.

    :param file_types: List of file types to handle.
    :type file_types: List[str]
    """

    def __init__(self, file_types: List[str]):
        self.__inputs = {file_type: [] for file_type in file_types}
        self.__labels = []
        self.__item_ids = []

    def add_data(self, item_id: str, file_contents: Union[Dict[str, str], List[Data]], labels: List[int]):
        """
        Adds processed data and labels to the respective storage.

        :param item_id: Unique identifier for the item.
        :type item_id: str
        :param file_contents: Processed file contents or AST data.
        :type file_contents: Union[Dict[str, str], List[Data]]
        :param labels: List of labels associated with the item.
        :type labels: List[int]
        """
        if isinstance(file_contents, list):
            self.__add_ast_data(file_contents, labels)
        else:
            self.__add_file_contents(file_contents, labels)
            self.__item_ids.append(item_id)

    def __add_ast_data(self, ast_data: List[Data], labels: List[int]):
        """
        Adds AST data to the storage.

        :param ast_data: List of AST data.
        :type ast_data: List[Data]
        :param labels: List of labels associated with the AST data.
        :type labels: List[int]
        """
        labels_tensor = torch.tensor(labels, dtype=torch.float)
        for data in ast_data:
            num_nodes = data.num_nodes

            # Pre-allocate the repeated_labels tensor
            repeated_labels = labels_tensor.unsqueeze(0).expand(num_nodes, -1)

            if repeated_labels.shape[0] != num_nodes:
                raise ValueError(f"Label shape mismatch: expected {num_nodes}, got {repeated_labels.shape[0]}")

            data.y = repeated_labels
            self.__inputs["ast"].append(data)
            self.__labels.append(labels)

    def __add_file_contents(self, file_contents: Dict[str, str], labels: List[int]):
        """
        Adds file contents to the storage.

        :param file_contents: Dictionary of file contents.
        :type file_contents: Dict[str, str]
        :param labels: List of labels associated with the file contents.
        :type labels: List[int]
        """
        for file_type, content in file_contents.items():
            self.__inputs[file_type].append(content)
        self.__labels.append(labels)

    def get_inputs(self) -> Dict[str, List[Union[str, Data]]]:
        """
        Retrieves the stored inputs.

        :return: Dictionary of stored inputs.
        :rtype: Dict[str, List[Union[str, Data]]]
        """
        return self.__inputs

    def get_labels(self) -> List[List[int]]:
        """
        Retrieves the stored labels.

        :return: List of stored labels.
        :rtype: List[List[int]]
        """
        return self.__labels
