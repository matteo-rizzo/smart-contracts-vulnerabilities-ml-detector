from typing import Dict, Any

from zss import Node, simple_distance


class ASTProcessor:
    @staticmethod
    def build_ast_tree(ast_json: Dict[str, Any]) -> Node:
        if not isinstance(ast_json, dict) or "type" not in ast_json:
            return Node("")

        root = Node(ast_json["type"])
        for child in ast_json.get("children", []):
            root.addkid(ASTProcessor.build_ast_tree(child))
        return root

    @staticmethod
    def compute_similarity(ast1: Dict[str, Any], ast2: Dict[str, Any]) -> float:
        tree1 = ASTProcessor.build_ast_tree(ast1)
        tree2 = ASTProcessor.build_ast_tree(ast2)
        ted = simple_distance(tree1, tree2)
        max_size = max(len(ast1), len(ast2), 1)
        return 1 - (ted / max_size)
