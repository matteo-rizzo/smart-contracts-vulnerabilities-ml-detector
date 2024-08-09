from typing import Dict, Optional

import networkx as nx
import torch
from torch_geometric.data import Data

from src.classes.data.GraphFeatureExtractor import GraphFeatureExtractor


class AST2GraphConverter:

    @classmethod
    def add_nodes_edges(cls, node: Dict, graph: nx.DiGraph, parent: Optional[int] = None):
        """
        Recursively add nodes and edges to the graph from the AST JSON object.

        :param node: The current node in the AST.
        :type node: Dict
        :param graph: The graph to which nodes and edges are added.
        :type graph: nx.DiGraph
        :param parent: The parent node ID.
        :type parent: Optional[int]
        """
        node_id = len(graph)
        try:
            features = GraphFeatureExtractor.extract_features(node)
            # Ensure the feature vector has a consistent size
            features = cls.pad_or_truncate_features(features, target_length=100)
            graph.add_node(node_id, features=features)
            if parent is not None:
                graph.add_edge(parent, node_id)
        except Exception as e:
            print(f"Error extracting features for node ID {node_id}: {e}")

        for key, value in node.items():
            if isinstance(value, dict):
                cls.add_nodes_edges(value, graph, node_id)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        cls.add_nodes_edges(item, graph, node_id)

    @staticmethod
    def pad_or_truncate_features(features: list, target_length: int) -> list:
        """
        Pads or truncates the features list to the target length.

        :param features: List of features to be padded or truncated.
        :param target_length: The desired length of the feature list.
        :return: A list of features of the target length.
        """
        if len(features) > target_length:
            return features[:target_length]
        elif len(features) < target_length:
            return features + [0] * (target_length - len(features))
        return features

    def ast_to_graph(self, ast_json: Dict) -> Data:
        """
        Convert an AST JSON object to a PyTorch Geometric Data object.

        :param ast_json: The AST JSON object to convert.
        :type ast_json: Dict
        :return: The corresponding PyTorch Geometric Data object.
        :rtype: Data
        """
        graph = nx.DiGraph()

        self.add_nodes_edges(ast_json, graph)

        if len(graph) == 0:
            raise ValueError("The graph is empty. No nodes were added.")

        try:
            edge_index = torch.tensor(list(graph.edges)).t().contiguous()
            x = torch.stack([torch.tensor(graph.nodes[n]['features'], dtype=torch.float) for n in graph.nodes])

            data = Data(x=x, edge_index=edge_index)
            return data
        except Exception as e:
            print(f"Error converting AST to graph: {e}")
