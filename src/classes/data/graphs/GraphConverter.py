from typing import Dict, Optional

import networkx as nx
import torch
from torch_geometric.data import Data

from src.classes.data.graphs.ASTFeatureExtractor import ASTFeatureExtractor
from src.classes.data.graphs.CFGFeatureExtractor import CFGFeatureExtractor
from src.classes.data.graphs.GraphFeatureExtractor import GraphFeatureExtractor


class GraphConverter(GraphFeatureExtractor):

    @classmethod
    def add_nodes_edges(cls, node: Dict, graph: nx.DiGraph, extractor: GraphFeatureExtractor,
                        parent: Optional[int] = None):
        """
        Recursively add nodes and edges to the graph from the JSON object (AST or CFG).

        :param node: The current node in the JSON structure (AST or CFG).
        :type node: Dict
        :param graph: The graph to which nodes and edges are added.
        :type graph: nx.DiGraph
        :param extractor: The feature extractor to use (AST or CFG).
        :type extractor: GraphFeatureExtractorBase
        :param parent: The parent node ID.
        :type parent: Optional[int]
        """
        node_id = len(graph)
        try:
            features = extractor.extract_features(node)
            features = cls.pad_or_truncate_features(features, target_length=100)
            graph.add_node(node_id, features=features)
            if parent is not None:
                graph.add_edge(parent, node_id)
        except Exception as e:
            print(f"Error extracting features for node ID {node_id}: {e}")

        for key, value in node.items():
            if isinstance(value, dict):
                cls.add_nodes_edges(value, graph, extractor, node_id)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        cls.add_nodes_edges(item, graph, extractor, node_id)

    def convert_to_graph(self, json_data: Dict, extractor: GraphFeatureExtractor) -> Data:
        """
        Convert a JSON object (AST or CFG) to a PyTorch Geometric Data object.

        :param json_data: The JSON object to convert.
        :type json_data: Dict
        :param extractor: The feature extractor to use (AST or CFG).
        :type extractor: GraphFeatureExtractorBase
        :return: The corresponding PyTorch Geometric Data object.
        :rtype: Data
        """
        graph = nx.DiGraph()

        self.add_nodes_edges(json_data, graph, extractor)

        if len(graph) == 0:
            raise ValueError("The graph is empty. No nodes were added.")

        try:
            edge_index = torch.tensor(list(graph.edges)).t().contiguous()
            x = torch.stack([torch.tensor(graph.nodes[n]['features'], dtype=torch.float) for n in graph.nodes])

            data = Data(x=x, edge_index=edge_index)
            return data
        except Exception as e:
            print(f"Error converting JSON to graph: {e}")
            raise

    def ast_to_graph(self, ast_json: Dict) -> Data:
        """
        Convert an AST JSON object to a PyTorch Geometric Data object using the AST feature extractor.

        :param ast_json: The AST JSON object to convert.
        :type ast_json: Dict
        :return: The corresponding PyTorch Geometric Data object.
        :rtype: Data
        """
        return self.convert_to_graph(ast_json, ASTFeatureExtractor())

    def cfg_to_graph(self, cfg_json: Dict) -> Data:
        """
        Convert a CFG JSON object to a PyTorch Geometric Data object using the CFG feature extractor.

        :param cfg_json: The CFG JSON object to convert.
        :type cfg_json: Dict
        :return: The corresponding PyTorch Geometric Data object.
        :rtype: Data
        """
        return self.convert_to_graph(cfg_json, CFGFeatureExtractor())
