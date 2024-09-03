import hashlib
from typing import List, Any


class GraphFeatureExtractor:
    @staticmethod
    def hash_feature(value: Any, num_bins: int = 1000) -> int:
        """
        Helper function to hash a value into a fixed number of bins.

        :param value: The value to hash.
        :param num_bins: The number of bins to hash the value into.
        :return: The hashed value as an integer.
        """
        return int(hashlib.md5(str(value).encode()).hexdigest(), 16) % num_bins

    @staticmethod
    def extract_features(node: dict, depth: int = 0) -> List[int]:
        """
        Extracts features from a given AST node.

        :param node: The AST node to extract features from.
        :type node: dict
        :param depth: The current depth of the node in the AST.
        :return: A list of extracted features.
        :rtype: List[int]
        """
        features = []

        # Extract node type
        node_type = node.get("nodeType", "Unknown")
        features.append(GraphFeatureExtractor.hash_feature(node_type))

        # General features
        features.append(depth)  # Node depth
        features.append(len(node.get('children', [])))  # Number of children
        features.append(depth * len(node.get('children', [])))  # Node complexity

        # Specific feature extraction based on node type
        if node_type == 'ContractDefinition':
            features += GraphFeatureExtractor.extract_contract_features(node)
        elif node_type == 'FunctionDefinition':
            features += GraphFeatureExtractor.extract_function_features(node)
        elif node_type == 'VariableDeclaration':
            features += GraphFeatureExtractor.extract_variable_features(node)
        elif node_type == 'PragmaDirective':
            features += GraphFeatureExtractor.extract_pragma_features(node)
        elif node_type == 'Mapping':
            features += GraphFeatureExtractor.extract_mapping_features(node)
        elif node_type == 'BinaryOperation':
            features += GraphFeatureExtractor.extract_binary_operation_features(node)

        # Extract features from child nodes recursively
        for child in node.get('children', []):
            features += GraphFeatureExtractor.extract_features(child, depth + 1)

        return features

    @staticmethod
    def extract_contract_features(node: dict) -> List[int]:
        """
        Extract features specific to ContractDefinition nodes.

        :param node: The ContractDefinition AST node.
        :return: A list of contract-specific features.
        """
        features = []
        features.append(GraphFeatureExtractor.hash_feature(node.get('contractKind', '')))
        features.append(GraphFeatureExtractor.hash_feature(node.get('name', '')))
        features.append(GraphFeatureExtractor.hash_feature(node.get('fullyImplemented', '')))
        features.append(len(node.get('linearizedBaseContracts', [])))  # Inheritance depth
        features.append(len(node.get('baseContracts', [])))  # Number of base contracts
        return features

    @staticmethod
    def extract_function_features(node: dict) -> List[int]:
        """
        Extract features specific to FunctionDefinition nodes.

        :param node: The FunctionDefinition AST node.
        :return: A list of function-specific features.
        """
        features = [GraphFeatureExtractor.hash_feature(node.get('name', '')),
                    GraphFeatureExtractor.hash_feature(node.get('visibility', '')),
                    GraphFeatureExtractor.hash_feature(node.get('stateMutability', '')),
                    int(node.get('isConstructor', False)), int(node.get('payable', False)),
                    len(node.get('modifiers', [])), len(node.get('parameters', {}).get('parameters', []))]

        # Detecting loops and conditionals within the function body
        function_body = node.get('body', {})
        if function_body:
            features.append(GraphFeatureExtractor.count_node_types(function_body, 'ForStatement'))
            features.append(GraphFeatureExtractor.count_node_types(function_body, 'WhileStatement'))
            features.append(GraphFeatureExtractor.count_node_types(function_body, 'IfStatement'))
            features.append(GraphFeatureExtractor.count_node_types(function_body, 'FunctionCall', 'require'))
            features.append(GraphFeatureExtractor.count_node_types(function_body, 'FunctionCall', 'assert'))

        return features

    @staticmethod
    def extract_variable_features(node: dict) -> List[int]:
        """
        Extract features specific to VariableDeclaration nodes.

        :param node: The VariableDeclaration AST node.
        :return: A list of variable-specific features.
        """
        features = [GraphFeatureExtractor.hash_feature(node.get('name', '')),
                    GraphFeatureExtractor.hash_feature(node.get('visibility', '')),
                    GraphFeatureExtractor.hash_feature(node.get('storageLocation', '')),
                    GraphFeatureExtractor.hash_feature(node.get('typeDescriptions', {}).get('typeString', '')),
                    int(node.get('constant', False)), int(node.get('stateVariable', False))]
        return features

    @staticmethod
    def extract_pragma_features(node: dict) -> List[int]:
        """
        Extract features specific to PragmaDirective nodes.

        :param node: The PragmaDirective AST node.
        :return: A list of pragma-specific features.
        """
        features = []
        for literal in node.get('literals', []):
            features.append(GraphFeatureExtractor.hash_feature(literal))
        return features

    @staticmethod
    def extract_mapping_features(node: dict) -> List[int]:
        """
        Extract features specific to Mapping nodes.

        :param node: The Mapping AST node.
        :return: A list of mapping-specific features.
        """
        features = [GraphFeatureExtractor.hash_feature(node.get('typeDescriptions', {}).get('typeString', ''))]
        return features

    @staticmethod
    def extract_binary_operation_features(node: dict) -> List[int]:
        """
        Extract features specific to BinaryOperation nodes.

        :param node: The BinaryOperation AST node.
        :return: A list of binary operation-specific features.
        """
        features = [GraphFeatureExtractor.hash_feature(node.get('operator', '')),
                    GraphFeatureExtractor.hash_feature(node.get('typeDescriptions', {}).get('typeString', ''))]
        return features

    @staticmethod
    def count_node_types(node: dict, node_type: str, function_name: str = None) -> int:
        """
        Count the occurrences of a specific node type (e.g., ForStatement) within a subtree.

        :param node: The root node of the subtree to search.
        :param node_type: The type of node to count.
        :param function_name: If provided, only counts the function calls with this name.
        :return: The count of nodes of the specified type within the subtree.
        """
        count = 0
        if node.get('nodeType', '') == node_type:
            if function_name:
                if node.get('expression', {}).get('name', '') == function_name:
                    count += 1
            else:
                count += 1

        for child in node.get('children', []):
            count += GraphFeatureExtractor.count_node_types(child, node_type, function_name)

        return count
