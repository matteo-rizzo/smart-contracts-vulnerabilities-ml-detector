import os
from collections import Counter

import pandas as pd
from tqdm import tqdm


class ASTDatasetProcessor:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.source_dir = os.path.join(base_path, "ast_valid")
        self.destination_dir = os.path.join(base_path, "ast_subsets")
        self.dataset_path = os.path.join(base_path, "consolidated.csv")
        self.valid_dataset_path = os.path.join(base_path, "valid_ast.csv")
        self.dataset = pd.DataFrame()

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

    @staticmethod
    def copy_file(source_file_path: str, destination_file_path: str) -> bool:
        """
        Copy a file from the source path to the destination path, modifying it if necessary.

        :param source_file_path: The path to the source file.
        :param destination_file_path: The path to the destination file.
        :return: True if the file was copied successfully, False otherwise.
        """
        try:
            if os.path.getsize(source_file_path) == 0:
                print(f'Skipping empty file: {source_file_path}')
                return False

            with open(source_file_path, 'r') as file:
                lines = file.readlines()

            if lines and lines[0].startswith("JSON AST:"):
                lines = lines[4:]  # Remove the first 4 lines

            with open(destination_file_path, 'w') as file:
                file.writelines(lines)

            print(f'Copied and overwritten {source_file_path} to {destination_file_path} with modification')
            return True

        except Exception as e:
            print(f'Failed to copy {source_file_path} to {destination_file_path}: {e}')
            return False

    def copy_files(self) -> Counter:
        """
        Copy files from the source directory to the appropriate destination directories based on the dataset.

        :return: A Counter object with property counts.
        """
        success_count = 0
        failure_count = 0
        property_counter = Counter()
        valid_entries = []

        for dataset_value in tqdm(self.dataset['dataset'].unique(), desc="Processing datasets"):
            dataset_dir = os.path.join(self.destination_dir, dataset_value)
            os.makedirs(dataset_dir, exist_ok=True)
            print(f'Ensured directory exists: {dataset_dir}')

            subset = self.dataset[self.dataset['dataset'] == dataset_value]

            for index, row in tqdm(subset.iterrows(), desc=f"Copying files for {dataset_value}", leave=False,
                                   total=subset.shape[0]):
                fp_sol = row['fp_sol']
                property_value = row['property']
                property_holds = row['property_holds']
                source_file_path = os.path.join(self.source_dir, os.path.basename(fp_sol) + '.ast.json')
                destination_file_path = os.path.join(dataset_dir, os.path.basename(fp_sol) + '.ast.json')

                if self.copy_file(source_file_path, destination_file_path):
                    success_count += 1
                    property_counter[property_value] += 1
                    valid_entries.append({
                        "dataset": dataset_value,
                        "property": property_value,
                        "property_holds": property_holds,
                        "fp_sol": fp_sol,
                        "fp_ast": os.path.splitext(os.path.basename(fp_sol))[0]
                    })
                else:
                    failure_count += 1

        print(f'Files copied successfully: {success_count} out of {len(self.dataset)}')
        print(f'Files failed to copy: {failure_count}')

        # Generate the valid_ast.csv
        valid_dataset = pd.DataFrame(valid_entries)
        valid_dataset.to_csv(self.valid_dataset_path, index=False)
        print(f'Valid dataset saved to {self.valid_dataset_path}')

        return property_counter

    def process(self) -> None:
        """
        Main function to orchestrate the overall process.
        """
        os.makedirs(self.destination_dir, exist_ok=True)
        print(f'Ensured destination directory exists: {self.destination_dir}')

        self.load_dataset()
        self.ensure_fp_sol_strings()
        property_counter = self.copy_files()

        print("\nSummary of property values after copying:")
        for property_value, count in property_counter.items():
            print(f'{property_value}: {count}')


if __name__ == "__main__":
    base_path = "dataset/cgt"
    processor = ASTDatasetProcessor(base_path)
    processor.process()
