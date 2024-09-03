import json
import logging
import os

import pandas as pd
from rich.logging import RichHandler
from tqdm import tqdm

# Setup logging with rich
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger("rich")


class NewCGTGenerator:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.dataset_path = os.path.join(base_path, "consolidated.csv")
        self.new_consolidated_path = os.path.join(base_path, "new_consolidated.csv")
        self.dataset = pd.DataFrame()
        self.new_consolidated_entries = []

    def load_dataset(self, delimiter: str = ";") -> None:
        """
        Load the dataset from a CSV file.
        """
        if not os.path.exists(self.dataset_path):
            logger.error(f"Dataset file not found at {self.dataset_path}")
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")

        logger.info(f'Loading dataset from {self.dataset_path}')
        self.dataset = pd.read_csv(self.dataset_path, delimiter=delimiter)
        logger.info(f'Initial dataset shape: {self.dataset.shape}\n')

    def ensure_fp_sol_strings(self) -> None:
        """
        Ensure 'fp_sol' column contains only strings and the files exist.
        """
        if 'fp_sol' not in self.dataset.columns:
            logger.error("'fp_sol' column not found in the dataset")
            raise KeyError("'fp_sol' column not found in the dataset")

        self.dataset = self.dataset[self.dataset['fp_sol'].apply(lambda x: isinstance(x, str) and x.strip() != "")]
        logger.info(f'Dataset after ensuring valid strings in fp_sol column: {self.dataset.shape}\n')

        # Ensure the 'fp_sol' files actually exist
        self.dataset = self.dataset[
            self.dataset['fp_sol'].apply(lambda x: os.path.exists(os.path.join(self.base_path, x)))]
        logger.info(f'Dataset after ensuring fp_sol files exist: {self.dataset.shape}\n')

    def load_opcodes(self, fp_sol: str) -> str:
        """
        Load opcodes from a text file.
        """
        opcodes_path = os.path.join(self.base_path, 'opcode', f"{os.path.splitext(os.path.basename(fp_sol))[0]}.txt")
        if os.path.exists(opcodes_path):
            try:
                with open(opcodes_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Failed to read opcodes from {opcodes_path}: {e}")
        else:
            logger.warning(f"Opcode file not found for {fp_sol}")
        return ""

    def load_ast(self, fp_sol: str) -> str:
        """
        Load AST from a file.
        """
        ast_path = os.path.join(self.base_path, 'ast', f"{os.path.splitext(os.path.basename(fp_sol))[0]}.ast")
        if os.path.exists(ast_path):
            try:
                with open(ast_path, 'r', encoding='utf-8') as file:
                    return file.read()
            except Exception as e:
                logger.error(f"Failed to read AST from {ast_path}: {e}")
        else:
            logger.warning(f"AST file not found for {fp_sol}")
        return ""

    def load_cfg(self, fp_sol: str) -> dict:
        """
        Load CFG from a JSON file.
        """
        cfg_path = os.path.join(self.base_path, 'cfg', f"{os.path.splitext(os.path.basename(fp_sol))[0]}.json")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from {cfg_path}: {e}")
            except Exception as e:
                logger.error(f"Failed to read CFG from {cfg_path}: {e}")
        else:
            logger.warning(f"CFG file not found for {fp_sol}")
        return {}

    def process_dataset(self) -> None:
        """
        Process the dataset to add the fp_ast, fp_opcode, and fp_cfg columns.
        """
        for _, row in tqdm(self.dataset.iterrows(), desc="Processing rows", leave=False, total=self.dataset.shape[0]):
            fp_sol = row['fp_sol']
            if not isinstance(fp_sol, str) or not fp_sol.strip():
                logger.warning(f"Skipping row with invalid fp_sol: {fp_sol}")
                continue

            fp_ast = self.load_ast(fp_sol)
            fp_opcode = self.load_opcodes(fp_sol)
            fp_cfg = self.load_cfg(fp_sol)

            self.new_consolidated_entries.append({
                **row,
                "fp_ast": fp_ast,
                "fp_opcode": fp_opcode,
                "fp_cfg": json.dumps(fp_cfg)  # Convert the dictionary to a JSON string for storage in CSV
            })

    def save_new_consolidated_dataset(self) -> None:
        """
        Save the new consolidated dataset to a CSV file.
        """
        new_consolidated_dataset = pd.DataFrame(self.new_consolidated_entries)
        if new_consolidated_dataset.empty:
            logger.warning("No valid entries to save.")
        else:
            try:
                new_consolidated_dataset.to_csv(self.new_consolidated_path, index=False)
                logger.info(f'New consolidated dataset saved to {self.new_consolidated_path}')
            except Exception as e:
                logger.error(f"Failed to save new consolidated dataset: {e}")

    def process(self) -> None:
        """
        Main function to orchestrate the overall process.
        """
        self.load_dataset()
        self.ensure_fp_sol_strings()
        self.process_dataset()
        self.save_new_consolidated_dataset()


if __name__ == "__main__":
    base_path = "dataset/cgt"
    processor = NewCGTGenerator(base_path)
    processor.process()
