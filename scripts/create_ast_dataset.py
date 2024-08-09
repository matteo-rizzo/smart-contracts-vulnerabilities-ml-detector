import os

import pandas as pd
from tqdm import tqdm


class ASTDatasetProcessor:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.dataset_path = os.path.join(base_path, "consolidated.csv")
        self.new_consolidated_path = os.path.join(base_path, "new_consolidated.csv")
        self.dataset = pd.DataFrame()
        self.new_consolidated_entries = []

    def load_dataset(self, delimiter: str = ";") -> None:
        """
        Load the dataset from a CSV file.

        :param delimiter: The delimiter used in the CSV file.
        """
        print(f'Loading dataset from {self.dataset_path}')
        self.dataset = pd.read_csv(self.dataset_path, delimiter=delimiter)
        print(f'Initial dataset shape: {self.dataset.shape}\n')

    def ensure_fp_sol_strings(self) -> None:
        """
        Ensure 'fp_sol' column contains only strings.
        """
        self.dataset = self.dataset[self.dataset['fp_sol'].apply(lambda x: isinstance(x, str))]
        print(f'Dataset after ensuring strings in fp_sol column: {self.dataset.shape}\n')

    def process_dataset(self) -> None:
        """
        Process the dataset to add the fp_ast column.
        """
        for _, row in tqdm(self.dataset.iterrows(), desc="Processing rows", leave=False,
                           total=self.dataset.shape[0]):
            fp_sol = row['fp_sol']
            fp_ast = os.path.splitext(os.path.basename(fp_sol))[0]
            self.new_consolidated_entries.append({**row, "fp_ast": fp_ast})

    def save_new_consolidated_dataset(self) -> None:
        """
        Save the new consolidated dataset to a CSV file.
        """
        new_consolidated_dataset = pd.DataFrame(self.new_consolidated_entries)
        new_consolidated_dataset.to_csv(self.new_consolidated_path, index=False)
        print(f'New consolidated dataset saved to {self.new_consolidated_path}')

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
    processor = ASTDatasetProcessor(base_path)
    processor.process()
