import glob
import json
import os
import re
from typing import Dict, Any, List, Tuple

from tqdm import tqdm

from src.classes.utils.DebugLogger import DebugLogger


class ContractFileProcessor:
    """
    A class for processing and grouping JSON files representing AST and CFG structures of smart contracts.
    """

    def __init__(self, ast_dir: str = None, cfg_dir: str = None):
        """
        Initialize the ContractFileProcessor.

        :param ast_dir: Directory containing AST JSON files.
        :param cfg_dir: Directory containing CFG JSON files.
        """
        self.logger = DebugLogger()
        self.ast_dir = ast_dir
        self.cfg_dir = cfg_dir
        self.grouped: Dict[str, Dict[str, Any]] = {}

    def process_files_in_dir(self, directory: str, key: str, progress_desc: str, logger_prefix: str = "") -> None:
        """
        Process JSON files in a given directory and update the grouped dictionary.

        :param directory: The directory to search files in (expects a structure like <label>/<file>.json).
        :param key: The key to store the loaded JSON data ("ast" or "cfg").
        :param progress_desc: Description used for the tqdm progress bar.
        :param logger_prefix: Prefix string for logging warnings (e.g. "Input ").
        """
        if not directory:
            return

        file_paths = glob.glob(os.path.join(directory, "*", "*.json"))
        for file_path in tqdm(file_paths, desc=progress_desc):
            contract_id = os.path.basename(file_path).split('.')[0].split('-')[0]
            label = os.path.basename(os.path.dirname(file_path)).lower()

            if contract_id not in self.grouped:
                self.grouped[contract_id] = {"source_files": [], "ast": None, "cfg": None, "label": label}
            else:
                if self.grouped[contract_id]["label"] != label:
                    self.logger.warning(
                        f"{logger_prefix}Contract {contract_id} has conflicting labels: "
                        f"{self.grouped[contract_id]['label']} and {label}"
                    )

            self.grouped[contract_id]["source_files"].append(file_path)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                self.grouped[contract_id][key] = data
            except Exception as e:
                self.logger.error(f"Error loading {file_path}: {e}")

    @staticmethod
    def combine_data(contents: dict, mode: str) -> dict:
        """
        Combine AST and/or CFG data from a contract based on the mode.

        :param contents: Dictionary containing AST and CFG data.
        :param mode: The mode in which data should be combined ("both", "ast", "cfg").
        :return: A dictionary containing the selected data.
        """
        if mode == "both":
            return {k: contents[k] for k in ["ast", "cfg"] if contents[k] is not None}
        elif mode == "ast":
            return {"ast": contents["ast"]} if contents["ast"] is not None else {}
        elif mode == "cfg":
            return {"cfg": contents["cfg"]} if contents["cfg"] is not None else {}
        return {}

    def group_files(self, mode: str = "both", is_input: bool = False) -> List[Tuple[str, List[str], dict, str]]:
        """
        Group JSON files from AST and CFG directories by contract identifier.

        :param mode: "both", "ast", or "cfg" to control which data to include.
        :param is_input: Whether the files being processed are input files (for logging purposes).
        :return: A list of tuples (contract_id, source file paths, combined JSON, label).
        """
        logger_prefix = "Input " if is_input else ""

        if mode in ["both", "ast"] and self.ast_dir:
            self.process_files_in_dir(self.ast_dir, key="ast", progress_desc=f"{logger_prefix}Grouping AST")

        if mode in ["both", "cfg"] and self.cfg_dir:
            self.process_files_in_dir(self.cfg_dir, key="cfg", progress_desc=f"{logger_prefix}Grouping CFG")

        results = [
            (contract_id, contents["source_files"], self.combine_data(contents, mode), contents["label"])
            for contract_id, contents in self.grouped.items()
        ]
        return results

    @staticmethod
    def load_source_code(contract_id, source_dir, label):
        """
        Loads and preprocesses the Solidity source code for a given contract.

        :param contract_id: ID of the contract to load.
        :param source_dir: Directory containing source code.
        :param label: Label (e.g., 'safe', 'reentrant') for classification.
        :return: Cleaned Solidity source code as a string.
        """
        source_path = os.path.join(source_dir, label, f"{contract_id}.sol")

        try:
            with open(source_path, "r", encoding="latin-1") as file:
                raw_code = file.read()
        except FileNotFoundError:
            print(f"Warning: Source code for contract {contract_id} not found at {source_path}.")
            return ""
        except Exception as e:
            print(f"Error loading source code for contract {contract_id}: {e}")
            return ""

        # Preprocess to remove comments and docstrings
        return ContractFileProcessor._clean_solidity_code(raw_code)

    @staticmethod
    def _clean_solidity_code(code: str) -> str:
        """
        Removes all Solidity comments and docstrings.

        :param code: Raw Solidity source code.
        :return: Cleaned Solidity source code.
        """
        # Remove block comments (/* ... */)
        code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

        # Remove single-line comments (// ...)
        code = re.sub(r"//.*", "", code)

        # Remove Solidity NatSpec comments (e.g., @notice, @dev, @param)
        code = re.sub(r"@\w+.*", "", code)

        # Trim extra whitespace and return
        return "\n".join(line.rstrip() for line in code.splitlines() if line.strip())
