from typing import Dict, Any

import networkx as nx


class CFGProcessor:
    @staticmethod
    def build_cfg_graph(cfg_json: Dict[str, Any]) -> nx.DiGraph:
        """
        Convert a CFG JSON representation into a NetworkX directed graph.

        :param cfg_json: JSON representation of a CFG.
        :return: A NetworkX directed graph.
        """
        graph = nx.DiGraph()

        # Extract nodes safely
        nodes = cfg_json.get("nodes", [])
        edges = cfg_json.get("edges", [])

        # Add nodes
        for node in nodes:
            clean_node = str(node).strip()  # Ensure node is a string and clean up formatting
            graph.add_node(clean_node)

        # Add edges with safety checks
        for edge in edges:
            source = str(edge.get("source", "")).strip()
            target = str(edge.get("target", "")).strip()

            if source and target:  # Ensure both source and target exist
                graph.add_edge(source, target)

        return graph

    @staticmethod
    def compute_similarity(cfg1: Dict[str, Any], cfg2: Dict[str, Any]) -> float:
        """
        Compute similarity between two CFG JSON objects using Graph Edit Distance (GED).

        :param cfg1: The first CFG JSON object.
        :param cfg2: The second CFG JSON object.
        :return: A similarity score between 0 and 1.
        """
        graph1 = CFGProcessor.build_cfg_graph(cfg1)
        graph2 = CFGProcessor.build_cfg_graph(cfg2)

        ged = nx.graph_edit_distance(graph1, graph2, timeout=5)

        # Handle None case (timeout scenario)
        if ged is None:
            ged = max(len(graph1.nodes), len(graph2.nodes))  # Use worst-case edit distance

        max_size = max(len(graph1.nodes), len(graph2.nodes), 1)  # Avoid division by zero
        return 1 - (ged / max_size)
