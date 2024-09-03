from typing import List

from src.classes.data.graphs.GraphFeatureExtractor import GraphFeatureExtractor


class CFGFeatureExtractor(GraphFeatureExtractor):
    @staticmethod
    def extract_features(node: dict) -> List[int]:
        """
        Extracts features from a given CFG node.

        :param node: The CFG node to extract features from.
        :return: A list of extracted features.
        """
        features = []

        # Basic block identification
        block_id = node.get("blockId", "Unknown")
        features.append(CFGFeatureExtractor.hash_feature(block_id))

        # Opcode-related features
        opcodes = node.get("opcodes", [])
        opcode_hashes = [CFGFeatureExtractor.hash_feature(opcode) for opcode in opcodes]
        features.extend(opcode_hashes)

        # Control flow specific features
        features.append(len(opcodes))  # Number of opcodes in the block
        features.append(len(node.get("successors", [])))  # Number of successors (outgoing edges)

        # Add specific CFG-related features
        features.append(CFGFeatureExtractor.hash_feature(node.get("type", "Unknown")))  # Control flow type
        features.append(CFGFeatureExtractor.extract_loop_features(node))
        features.append(CFGFeatureExtractor.extract_branching_features(node))

        return features

    @staticmethod
    def extract_loop_features(node: dict) -> int:
        """
        Extract features related to loops in the CFG.

        :param node: The CFG node to check for loops.
        :return: A hashed feature indicating the presence and type of loop.
        """
        loop_type = node.get("loopType", "None")
        return CFGFeatureExtractor.hash_feature(loop_type)

    @staticmethod
    def extract_branching_features(node: dict) -> int:
        """
        Extract features related to branching in the CFG.

        :param node: The CFG node to check for branching.
        :return: A hashed feature indicating the presence and type of branching.
        """
        branching_type = node.get("branchingType", "None")
        return CFGFeatureExtractor.hash_feature(branching_type)

    @staticmethod
    def extract_arithmetic_operation_features(node: dict) -> int:
        """
        Extract features related to arithmetic operations in the CFG.

        :param node: The CFG node to check for arithmetic operations.
        :return: A hashed feature indicating the presence and type of arithmetic operation.
        """
        arithmetic_operations = node.get("arithmeticOperations", [])
        operations_hash = sum([CFGFeatureExtractor.hash_feature(op) for op in arithmetic_operations])
        return operations_hash
