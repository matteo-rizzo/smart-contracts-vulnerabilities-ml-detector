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
        Ensure 'fp_sol' column contains only valid strings and the files exist.
        """
        if 'fp_sol' not in self.dataset.columns:
            logger.error("'fp_sol' column not found in the dataset")
            raise KeyError("'fp_sol' column not found in the dataset")

        logger.info("Ensuring 'fp_sol' column contains valid strings and files exist...")
        self.dataset = self.dataset[self.dataset['fp_sol'].apply(
            lambda x: isinstance(x, str) and x.strip() != "" and
                      os.path.exists(os.path.join(self.base_path, "source", x + ".sol"))
        )]
        logger.info(f'Dataset after filtering: {self.dataset.shape}\n')

    @staticmethod
    def load_file(file_path: str, file_type: str = "text"):
        """
        Load a file's content based on its type (text or JSON).
        If successful, return the file's name; otherwise, return an empty string.
        """
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    if not content:  # Check for empty file
                        logger.warning(f"File is empty: {file_path}")
                        return ""
                    return os.path.basename(file_path)  # Return only the filename if successful
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON from {file_path}: {e}")
            except Exception as e:
                logger.error(f"Failed to read {file_type} from {file_path}: {e}")
        else:
            logger.warning(f"{file_type.capitalize()} file not found: {file_path}")
        return ""

    def process_dataset(self) -> None:
        """
        Process the dataset to add the fp_ast, fp_opcode, and fp_cfg columns.
        """
        logger.info("Processing dataset rows...")
        for _, row in tqdm(self.dataset.iterrows(), desc="Processing rows", leave=False, total=self.dataset.shape[0]):
            fp_sol = row['fp_sol']
            fp_ast = self.load_file(os.path.join(self.base_path, 'ast', f"{fp_sol}.ast.json"), file_type="json")
            fp_opcode = self.load_file(os.path.join(self.base_path, 'opcode', f"{fp_sol}.opcodes.txt"), file_type="text")
            fp_cfg = self.load_file(os.path.join(self.base_path, 'cfg', f"{fp_sol}-combined.json"), file_type="json")

            self.new_consolidated_entries.append({
                **row,
                "fp_ast": fp_ast,
                "fp_opcode": fp_opcode,
                "fp_cfg": fp_cfg
            })
        logger.info("Finished processing dataset rows.")

    def save_new_consolidated_dataset(self) -> None:
        """
        Save the new consolidated dataset to a CSV file.
        """
        if not self.new_consolidated_entries:
            logger.warning("No valid entries to save.")
            return

        new_consolidated_dataset = pd.DataFrame(self.new_consolidated_entries)
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
        logger.info("Processing completed successfully.")


if __name__ == "__main__":
    base_path = os.path.join("dataset", "cgt")
    processor = NewCGTGenerator(base_path)
    processor.process()
