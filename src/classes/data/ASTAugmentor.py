import copy
import random
from typing import Any, Dict, List, Set

from tqdm import tqdm


class ASTAugmentor:

    def __init__(self):
        """
        Initialize the ASTAugmentor with predefined strategies.
        """
        self.substitutions: Dict[str, str] = {'FunctionDefinition': 'ModifierDefinition'}
        self.insertions: Dict[str, Any] = {
            'body': {'nodeType': 'ExpressionStatement', 'expression': {'nodeType': 'Literal', 'value': '0'}}
        }
        self.deletions: Set[str] = {'ModifierDefinition'}
        self.renames: Dict[str, str] = {'oldVarName': 'newVarName', 'oldFuncName': 'newFuncName'}

    def substitute_nodes(self, ast: Any, substitutions: Dict[str, str]) -> Any:
        """
        Substitute certain nodes in the AST with other semantically equivalent nodes.

        :param ast: The abstract syntax tree to augment.
        :type ast: Any
        :param substitutions: A dictionary mapping node types to their substitutions.
        :type substitutions: Dict[str, str]
        :return: The augmented abstract syntax tree.
        :rtype: Any
        """
        if isinstance(ast, dict):
            for key, value in ast.items():
                if key in substitutions:
                    ast[key] = substitutions[key]
                else:
                    ast[key] = self.substitute_nodes(value, substitutions)
        elif isinstance(ast, list):
            for i in range(len(ast)):
                ast[i] = self.substitute_nodes(ast[i], substitutions)
        return ast

    def insert_nodes(self, ast: Any, insertions: Dict[str, Any]) -> Any:
        """
        Insert certain nodes into the AST.

        :param ast: The abstract syntax tree to augment.
        :type ast: Any
        :param insertions: A dictionary mapping node attributes to the nodes to be inserted.
        :type insertions: Dict[str, Any]
        :return: The augmented abstract syntax tree.
        :rtype: Any
        """
        if isinstance(ast, dict):
            for key, value in ast.items():
                if key in insertions:
                    ast[key] = [value, insertions[key]]
                else:
                    ast[key] = self.insert_nodes(value, insertions)
        elif isinstance(ast, list):
            for i in range(len(ast)):
                ast[i] = self.insert_nodes(ast[i], insertions)
        return ast

    def delete_nodes(self, ast: Any, deletions: Set[str]) -> Any:
        """
        Delete certain nodes from the AST.

        :param ast: The abstract syntax tree to augment.
        :type ast: Any
        :param deletions: A set of node types to be deleted.
        :type deletions: Set[str]
        :return: The augmented abstract syntax tree.
        :rtype: Any
        """
        if isinstance(ast, dict):
            keys_to_delete = [key for key in ast if key in deletions]
            for key in keys_to_delete:
                del ast[key]
            for key, value in ast.items():
                ast[key] = self.delete_nodes(value, deletions)
        elif isinstance(ast, list):
            ast = [self.delete_nodes(item, deletions) for item in ast if item not in deletions]
        return ast

    def rename_identifiers(self, ast: Any, renames: Dict[str, str]) -> Any:
        """
        Rename variables/functions in the AST.

        :param ast: The abstract syntax tree to augment.
        :type ast: Any
        :param renames: A dictionary mapping old identifiers to new identifiers.
        :type renames: Dict[str, str]
        :return: The augmented abstract syntax tree.
        :rtype: Any
        """
        if isinstance(ast, dict):
            for key, value in ast.items():
                if key == 'name' and value in renames:
                    ast[key] = renames[value]
                else:
                    ast[key] = self.rename_identifiers(value, renames)
        elif isinstance(ast, list):
            for i in range(len(ast)):
                ast[i] = self.rename_identifiers(ast[i], renames)
        return ast

    def reorder_statements(self, ast: Any) -> Any:
        """
        Randomly reorder statements in the AST.

        :param ast: The abstract syntax tree to augment.
        :type ast: Any
        :return: The augmented abstract syntax tree.
        :rtype: Any
        """
        if isinstance(ast, dict) and 'body' in ast:
            if isinstance(ast['body'], list):
                random.shuffle(ast['body'])
            else:
                self.reorder_statements(ast['body'])
        elif isinstance(ast, list):
            for item in ast:
                self.reorder_statements(item)
        return ast

    def add_no_op_statements(self, ast: Any) -> Any:
        """
        Add no-op statements to the AST.

        :param ast: The abstract syntax tree to augment.
        :type ast: Any
        :return: The augmented abstract syntax tree.
        :rtype: Any
        """
        no_op_statement = {'nodeType': 'ExpressionStatement', 'expression': {'nodeType': 'Literal', 'value': '0'}}
        if isinstance(ast, dict) and 'body' in ast:
            if isinstance(ast['body'], list):
                ast['body'].append(no_op_statement)
            else:
                self.add_no_op_statements(ast['body'])
        elif isinstance(ast, list):
            for item in ast:
                self.add_no_op_statements(item)
        return ast

    def apply_augmentation(self, ast: Any) -> Any:
        """
        Apply random augmentations to the AST.

        :param ast: The abstract syntax tree to augment.
        :type ast: Any
        :return: The augmented abstract syntax tree.
        :rtype: Any
        """
        if random.random() > 0.5:
            ast = self.substitute_nodes(ast, self.substitutions)
        if random.random() > 0.5:
            ast = self.insert_nodes(ast, self.insertions)
        if random.random() > 0.5:
            ast = self.delete_nodes(ast, self.deletions)
        if random.random() > 0.5:
            ast = self.rename_identifiers(ast, self.renames)
        if random.random() > 0.5:
            ast = self.reorder_statements(ast)
        if random.random() > 0.5:
            ast = self.add_no_op_statements(ast)
        return ast

    def generate_augmented_asts(self, dataset: List[Any], num_augmentations: int = 5) -> List[Any]:
        """
        Generate augmented ASTs for each AST in the dataset.

        :param dataset: A list of abstract syntax trees to augment.
        :type dataset: List[Any]
        :param num_augmentations: The number of augmentations to generate for each AST.
        :type num_augmentations: int
        :return: A list of augmented abstract syntax trees.
        :rtype: List[Any]
        """
        augmented_dataset: List[Any] = []
        for ast in tqdm(dataset, desc="Generating augmented ASTs"):
            augmented_dataset.append(ast)
            for _ in range(num_augmentations):
                augmented_ast = self.apply_augmentation(copy.deepcopy(ast))
                augmented_dataset.append(augmented_ast)
        return augmented_dataset