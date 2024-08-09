from typing import List, Any

from torch_geometric.data import Data, Dataset

from src.classes.data.ASTAugmentor import ASTAugmentor


class GraphDataset(Dataset):
    def __init__(self, graphs: List[Data], augment: bool = False, num_augmentations: int = 5):
        """
        Initializes the GraphDataset with a list of graph data objects and optional augmentation.

        :param graphs: A list of graph data objects to be used in the dataset.
        :type graphs: List[Data]
        :param augment: Whether to apply augmentation to the graphs, defaults to False.
        :type augment: bool
        :param num_augmentations: The number of augmentations to generate for each graph, defaults to 5.
        :type num_augmentations: int
        """
        super().__init__()
        self.graphs: List[Data] = graphs
        self.augment = augment
        self.num_augmentations = num_augmentations

        if self.augment:
            self.augmentor = ASTAugmentor()
            self.graphs = self._augment_graphs()

    def _augment_graphs(self) -> List[Any]:
        """
        Generates augmented graph data objects.

        :return: A list of augmented graph data objects.
        :rtype: List[Any]
        """
        return self.augmentor.generate_augmented_asts(self.graphs, self.num_augmentations)

    def __len__(self) -> int:
        """
        Returns the number of graph data objects in the dataset.

        :return: The number of graph data objects in the dataset.
        :rtype: int
        """
        return len(self.graphs)

    def __getitem__(self, idx: int) -> Data:
        """
        Retrieves the graph data object at the specified index.

        :param idx: The index of the graph data object to retrieve.
        :type idx: int
        :return: The graph data object at the specified index.
        :rtype: Data
        """
        return self.graphs[idx]

    def get_labels(self) -> List[List[int]]:
        """
        Retrieves the labels for all graph data objects.

        :return: The labels for all graph data objects.
        :rtype: List[List[int]]
        """
        return [graph.y.tolist() for graph in self.graphs]
