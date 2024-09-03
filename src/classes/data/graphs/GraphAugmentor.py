import copy
import random
from typing import Any, Dict, List, Set, Union

from tqdm import tqdm


class GraphAugmentor:
    """
    General-purpose class to apply augmentations to graph-like structures (e.g., ASTs, CFGs).

    Augmentations include substituting nodes, inserting nodes, deleting nodes, renaming identifiers,
    reordering nodes, and adding no-op nodes.
    """

    def __init__(self):
        """
        Initialize the GraphAugmentor with predefined strategies.
        """
        self.substitutions: Dict[str, str] = {}  # Maps node types to their substitutions
        self.insertions: Dict[str, Any] = {}  # Maps node attributes to nodes to be inserted
        self.deletions: Set[str] = set()  # Set of node types to be deleted
        self.renames: Dict[str, str] = {}  # Maps old identifiers to new identifiers

    def substitute_nodes(self, graph: Any, substitutions: Dict[str, str]) -> Any:
        """
        Substitute certain nodes in the graph with other semantically equivalent nodes.

        :param graph: The graph structure to augment.
        :type graph: Any
        :param substitutions: A dictionary mapping node types to their substitutions.
        :type substitutions: Dict[str, str]
        :return: The augmented graph structure.
        :rtype: Any
        """
        if isinstance(graph, dict):
            node_type = graph.get('nodeType')
            if node_type in substitutions:
                graph['nodeType'] = substitutions[node_type]
            for key, value in graph.items():
                graph[key] = self.substitute_nodes(value, substitutions)
        elif isinstance(graph, list):
            for i in range(len(graph)):
                graph[i] = self.substitute_nodes(graph[i], substitutions)
        return graph

    def insert_nodes(self, graph: Any, insertions: Dict[str, Any]) -> Any:
        """
        Insert certain nodes into the graph.

        :param graph: The graph structure to augment.
        :type graph: Any
        :param insertions: A dictionary mapping node attributes to the nodes to be inserted.
        :type insertions: Dict[str, Any]
        :return: The augmented graph structure.
        :rtype: Any
        """
        if isinstance(graph, dict):
            for key, value in graph.items():
                if key in insertions and isinstance(graph[key], list):
                    graph[key].insert(random.randint(0, len(graph[key])), insertions[key])
                else:
                    graph[key] = self.insert_nodes(value, insertions)
        elif isinstance(graph, list):
            for i in range(len(graph)):
                graph[i] = self.insert_nodes(graph[i], insertions)
        return graph

    def delete_nodes(self, graph: Any, deletions: Set[str]) -> Any:
        """
        Delete certain nodes from the graph.

        :param graph: The graph structure to augment.
        :type graph: Any
        :param deletions: A set of node types to be deleted.
        :type deletions: Set[str]
        :return: The augmented graph structure.
        :rtype: Any
        """
        if isinstance(graph, dict):
            node_type = graph.get('nodeType')
            if node_type in deletions:
                return None  # Return None to indicate the node should be deleted
            for key, value in list(graph.items()):  # list() to avoid runtime error due to modification
                graph[key] = self.delete_nodes(value, deletions)
            # Remove keys with None values
            graph = {k: v for k, v in graph.items() if v is not None}
        elif isinstance(graph, list):
            graph = [self.delete_nodes(item, deletions) for item in graph if item is not None]
        return graph

    def rename_identifiers(self, graph: Any, renames: Dict[str, str], scope: Set[str] = None) -> Any:
        """
        Rename variables/functions in the graph in a context-aware manner.

        :param graph: The graph structure to augment.
        :type graph: Any
        :param renames: A dictionary mapping old identifiers to new identifiers.
        :type renames: Dict[str, str]
        :param scope: The current scope of variable/function names to avoid conflicts.
        :type scope: Set[str]
        :return: The augmented graph structure.
        :rtype: Any
        """
        if scope is None:
            scope = set()

        if isinstance(graph, dict):
            if 'name' in graph and graph['name'] in renames:
                new_name = renames[graph['name']]
                if new_name not in scope:
                    graph['name'] = new_name
                    scope.add(new_name)
            for key, value in graph.items():
                graph[key] = self.rename_identifiers(value, renames, scope)
        elif isinstance(graph, list):
            for i in range(len(graph)):
                graph[i] = self.rename_identifiers(graph[i], renames, scope)
        return graph

    def reorder_statements(self, graph: Any) -> Any:
        """
        Randomly reorder statements/nodes in the graph while maintaining logical dependencies.

        :param graph: The graph structure to augment.
        :type graph: Any
        :return: The augmented graph structure.
        :rtype: Any
        """
        if isinstance(graph, dict) and 'body' in graph:
            if isinstance(graph['body'], list):
                dependent_statements = [stmt for stmt in graph['body'] if self.has_dependency(stmt, graph['body'])]
                independent_statements = [stmt for stmt in graph['body'] if not self.has_dependency(stmt, graph['body'])]
                random.shuffle(independent_statements)
                graph['body'] = dependent_statements + independent_statements
            else:
                self.reorder_statements(graph['body'])
        elif isinstance(graph, list):
            for item in graph:
                self.reorder_statements(item)
        return graph

    @staticmethod
    def has_dependency(statement: Any, statements: List[Any]) -> bool:
        """
        Determine if a statement/node has dependencies on other statements/nodes in the list.

        :param statement: The statement/node to check for dependencies.
        :type statement: Any
        :param statements: The list of statements/nodes in the block.
        :type statements: List[Any]
        :return: True if the statement/node has dependencies, False otherwise.
        :rtype: bool
        """
        # Example logic: Check if a variable used in the statement is defined in another statement
        if isinstance(statement, dict):
            used_vars = set()
            defined_vars = set()
            for stmt in statements:
                if 'name' in stmt and stmt['name'] in used_vars:
                    return True
                if 'name' in stmt:
                    defined_vars.add(stmt['name'])
            if 'name' in statement and statement['name'] in defined_vars:
                return True
        return False

    def add_no_op_nodes(self, graph: Any) -> Any:
        """
        Add no-op nodes/statements to the graph where logically appropriate.

        :param graph: The graph structure to augment.
        :type graph: Any
        :return: The augmented graph structure.
        :rtype: Any
        """
        no_op_node = {'nodeType': 'ExpressionStatement', 'expression': {'nodeType': 'Literal', 'value': '0'}}
        if isinstance(graph, dict) and 'body' in graph:
            if isinstance(graph['body'], list):
                insertion_point = random.randint(0, len(graph['body']))
                graph['body'].insert(insertion_point, no_op_node)
            else:
                self.add_no_op_nodes(graph['body'])
        elif isinstance(graph, list):
            for item in graph:
                self.add_no_op_nodes(item)
        return graph

    def apply_augmentation(self, graph: Any) -> Any:
        """
        Apply random augmentations to the graph.

        :param graph: The graph structure to augment.
        :type graph: Any
        :return: The augmented graph structure.
        :rtype: Any
        """
        graph = copy.deepcopy(graph)
        if random.random() > 0.5:
            graph = self.substitute_nodes(graph, self.substitutions)
        if random.random() > 0.5:
            graph = self.insert_nodes(graph, self.insertions)
        if random.random() > 0.5:
            graph = self.delete_nodes(graph, self.deletions)
        if random.random() > 0.5:
            graph = self.rename_identifiers(graph, self.renames)
        if random.random() > 0.5:
            graph = self.reorder_statements(graph)
        if random.random() > 0.5:
            graph = self.add_no_op_nodes(graph)
        return graph

    def generate_augmented_graphs(self, dataset: List[Any], num_augmentations: int = 5) -> List[Any]:
        """
        Generate augmented graphs for each graph in the dataset.

        :param dataset: A list of graphs to augment.
        :type dataset: List[Any]
        :param num_augmentations: The number of augmentations to generate for each graph.
        :type num_augmentations: int
        :return: A list of augmented graphs.
        :rtype: List[Any]
        """
        augmented_dataset: List[Any] = []
        for graph in tqdm(dataset, desc="Generating augmented graphs"):
            augmented_dataset.append(graph)
            for _ in range(num_augmentations):
                augmented_graph = self.apply_augmentation(graph)
                augmented_dataset.append(augmented_graph)
        return augmented_dataset
