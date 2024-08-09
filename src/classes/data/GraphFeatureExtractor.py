import hashlib
from typing import Dict, List, Any


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
    def extract_features(node: Dict) -> List[int]:
        """
        Extracts features from a given AST node.

        :param node: The AST node to extract features from.
        :type node: Dict
        :return: A list of extracted features.
        :rtype: List[int]
        """
        # Initialize features with default values
        name_feature, value_feature = [0], [0]
        src_feature, type_desc_features = [0, 0], [0, 0]
        state_mutability_feature, visibility_feature = [0], [0]

        # Extract basic features
        node_type = node.get('nodeType', 'Unknown')
        type_feature = [GraphFeatureExtractor.hash_feature(node_type)]

        # Extract additional features if they exist
        if 'name' in node:
            name_feature = [GraphFeatureExtractor.hash_feature(node.get('name', ''))]
        if 'value' in node:
            value_feature = [GraphFeatureExtractor.hash_feature(node.get('value', ''))]

        # Extract src features (start, end, and length if available)
        if 'src' in node:
            src_value = node['src']
            if isinstance(src_value, str):
                start, length, *_ = map(int, src_value.split(':'))
                src_feature = [start, length]
            elif isinstance(src_value, dict):
                start = src_value.get('start', 0)
                length = src_value.get('length', 0)
                src_feature = [start, length]

        # Extract typeDescriptions features if they exist
        if 'typeDescriptions' in node:
            type_desc = node['typeDescriptions']
            type_desc_features = [
                GraphFeatureExtractor.hash_feature(type_desc.get('typeString', '')),
                GraphFeatureExtractor.hash_feature(type_desc.get('typeIdentifier', ''))
            ]

        # Extract stateMutability if it exists
        if 'stateMutability' in node:
            state_mutability_feature = [GraphFeatureExtractor.hash_feature(node.get('stateMutability', ''))]

        # Extract visibility if it exists
        if 'visibility' in node:
            visibility_feature = [GraphFeatureExtractor.hash_feature(node.get('visibility', ''))]

        # Combine all features into a single feature vector
        return (type_feature + name_feature + value_feature + src_feature +
                type_desc_features + state_mutability_feature + visibility_feature)
