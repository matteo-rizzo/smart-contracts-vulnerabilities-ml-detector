from typing import List

from src.classes.data.graphs.GraphFeatureExtractor import GraphFeatureExtractor


class ASTFeatureExtractor(GraphFeatureExtractor):
    @staticmethod
    def extract_features(node: dict, depth: int = 0) -> List[int]:
        """
        Extracts features from a given AST node.

        :param node: The AST node to extract features from.
        :param depth: The current depth of the node in the AST.
        :return: A list of extracted features.
        """
        features = []

        # Extract node type
        node_type = node.get("nodeType", "Unknown")
        features.append(ASTFeatureExtractor.hash_feature(node_type))

        # General features
        features.append(depth)  # Node depth
        features.append(len(node.get('children', [])))  # Number of children
        features.append(depth * len(node.get('children', [])))  # Node complexity

        # Specific feature extraction based on node type
        if node_type == 'ContractDefinition':
            features += ASTFeatureExtractor.extract_contract_features(node)
        elif node_type == 'FunctionDefinition':
            features += ASTFeatureExtractor.extract_function_features(node)
        elif node_type == 'VariableDeclaration':
            features += ASTFeatureExtractor.extract_variable_features(node)
        elif node_type == 'PragmaDirective':
            features += ASTFeatureExtractor.extract_pragma_features(node)
        elif node_type == 'Mapping':
            features += ASTFeatureExtractor.extract_mapping_features(node)
        elif node_type == 'BinaryOperation':
            features += ASTFeatureExtractor.extract_binary_operation_features(node)

        # Extract features from child nodes recursively
        for child in node.get('children', []):
            features += ASTFeatureExtractor.extract_features(child, depth + 1)

        return features

    @staticmethod
    def extract_contract_features(node: dict) -> List[int]:
        """
        Extract features specific to ContractDefinition nodes.
        """
        features = [ASTFeatureExtractor.hash_feature(node.get('contractKind', '')),
                    ASTFeatureExtractor.hash_feature(node.get('name', '')),
                    ASTFeatureExtractor.hash_feature(node.get('fullyImplemented', '')),
                    len(node.get('linearizedBaseContracts', [])), len(node.get('baseContracts', []))]
        return features

    @staticmethod
    def extract_function_features(node: dict) -> List[int]:
        """
        Extract features specific to FunctionDefinition nodes.
        """
        features = [
            ASTFeatureExtractor.hash_feature(node.get('name', '')),
            ASTFeatureExtractor.hash_feature(node.get('visibility', '')),
            ASTFeatureExtractor.hash_feature(node.get('stateMutability', '')),
            int(node.get('isConstructor', False)),
            int(node.get('payable', False)),
            len(node.get('modifiers', [])),
            len(node.get('parameters', {}).get('parameters', []))
        ]

        # Detecting loops and conditionals within the function body
        function_body = node.get('body', {})
        if function_body:
            features.append(ASTFeatureExtractor.count_node_types(function_body, 'ForStatement'))
            features.append(ASTFeatureExtractor.count_node_types(function_body, 'WhileStatement'))
            features.append(ASTFeatureExtractor.count_node_types(function_body, 'IfStatement'))
            features.append(ASTFeatureExtractor.count_node_types(function_body, 'FunctionCall', 'require'))
            features.append(ASTFeatureExtractor.count_node_types(function_body, 'FunctionCall', 'assert'))

        return features

    @staticmethod
    def extract_variable_features(node: dict) -> List[int]:
        """
        Extract features specific to VariableDeclaration nodes.
        """
        features = [
            ASTFeatureExtractor.hash_feature(node.get('name', '')),
            ASTFeatureExtractor.hash_feature(node.get('visibility', '')),
            ASTFeatureExtractor.hash_feature(node.get('storageLocation', '')),
            ASTFeatureExtractor.hash_feature(node.get('typeDescriptions', {}).get('typeString', '')),
            int(node.get('constant', False)),
            int(node.get('stateVariable', False))
        ]
        return features

    @staticmethod
    def extract_pragma_features(node: dict) -> List[int]:
        """
        Extract features specific to PragmaDirective nodes.
        """
        features = []
        for literal in node.get('literals', []):
            features.append(ASTFeatureExtractor.hash_feature(literal))
        return features

    @staticmethod
    def extract_mapping_features(node: dict) -> List[int]:
        """
        Extract features specific to Mapping nodes.
        """
        features = [ASTFeatureExtractor.hash_feature(node.get('typeDescriptions', {}).get('typeString', ''))]
        return features

    @staticmethod
    def extract_binary_operation_features(node: dict) -> List[int]:
        """
        Extract features specific to BinaryOperation nodes.
        """
        features = [
            ASTFeatureExtractor.hash_feature(node.get('operator', '')),
            ASTFeatureExtractor.hash_feature(node.get('typeDescriptions', {}).get('typeString', ''))
        ]
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
            count += ASTFeatureExtractor.count_node_types(child, node_type, function_name)

        return count
