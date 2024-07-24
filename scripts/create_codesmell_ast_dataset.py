import os
from collections import Counter
from typing import Set

import pandas as pd


class DatasetProcessor:
    def __init__(self, base_path: str, filter_most_frequent: bool = False):
        self.base_path = base_path
        self.filter_most_frequent = filter_most_frequent
        self.source_dir = os.path.join(base_path, "ast")
        self.destination_dir = os.path.join(base_path, "ast_valid")
        self.dataset_path = os.path.join(base_path, "consolidated.csv")
        self.dataset = pd.DataFrame()
        self.fp_sol_to_property = {}

    def load_dataset(self, delimiter: str = ";") -> None:
        """
        Load the dataset from a CSV file.

        :param delimiter: The delimiter used in the CSV file.
        """
        print(f'Loading dataset from {self.dataset_path}')
        self.dataset = pd.read_csv(self.dataset_path, delimiter=delimiter)
        print(f'Initial dataset shape: {self.dataset.shape}\n')

    def filter_most_frequent_item(self) -> None:
        """
        Filter the dataset to include only rows with the most frequent item in the 'dataset' column.
        """
        most_frequent_item = self.dataset["dataset"].mode()[0]
        print(f'Most frequent item in the "dataset" column: {most_frequent_item}')
        self.dataset = self.dataset[self.dataset["dataset"] == most_frequent_item]
        print(f'Filtered dataset shape: {self.dataset.shape}')

    def ensure_fp_sol_strings_and_remove_duplicates(self) -> None:
        """
        Ensure 'fp_sol' column contains only strings and remove duplicates.
        """
        self.dataset = self.dataset[self.dataset['fp_sol'].apply(lambda x: isinstance(x, str))].drop_duplicates(
            subset='fp_sol')
        print(f'Dataset after ensuring strings and dropping duplicates shape: {self.dataset.shape}\n')

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
        Copy files from the source directory to the destination directory based on the dataset.

        :return: A Counter object with property counts.
        """
        success_count = 0
        failure_count = 0
        property_counter = Counter()

        self.fp_sol_to_property = self.dataset.set_index('fp_sol')['property'].to_dict()

        for fp_sol, property_value in self.fp_sol_to_property.items():
            source_file_path = os.path.join(self.source_dir, os.path.basename(fp_sol) + '.ast.json')
            destination_file_path = os.path.join(self.destination_dir, os.path.basename(fp_sol) + '.ast.json')

            if self.copy_file(source_file_path, destination_file_path):
                success_count += 1
                property_counter[property_value] += 1
            else:
                failure_count += 1

        print(f'Files copied successfully: {success_count} out of {len(self.fp_sol_to_property)}')
        print(f'Files failed to copy: {failure_count}')

        return property_counter

    def delete_files(self, properties_to_delete: Set[str]) -> None:
        """
        Delete files in the destination directory that correspond to properties with fewer than 5 occurrences.

        :param properties_to_delete: A set of properties to delete.
        """
        for fp_sol, property_value in self.fp_sol_to_property.items():
            if property_value in properties_to_delete:
                destination_file_path = os.path.join(self.destination_dir, os.path.basename(fp_sol) + '.ast.json')
                if os.path.exists(destination_file_path):
                    try:
                        os.remove(destination_file_path)
                        print(f'Deleted file: {destination_file_path} due to insufficient property count')
                    except Exception as e:
                        print(f'Failed to delete {destination_file_path}: {e}')

    def process(self) -> None:
        """
        Main function to orchestrate the overall process.
        """
        os.makedirs(self.destination_dir, exist_ok=True)
        print(f'Ensured destination directory exists: {self.destination_dir}')

        self.load_dataset()

        if self.filter_most_frequent:
            self.filter_most_frequent_item()
        else:
            print(f'No filtering applied based on most frequent item. Number of rows: {self.dataset.shape[0]}')

        self.ensure_fp_sol_strings_and_remove_duplicates()

        property_counter = self.copy_files()

        print("\nSummary of property values after copying:")
        for property_value, count in property_counter.items():
            print(f'{property_value}: {count}')

        properties_to_delete = {property for property, count in property_counter.items() if count < 5}
        print(f'Properties to delete (count < 5): {properties_to_delete}')

        self.delete_files(properties_to_delete)


if __name__ == "__main__":
    base_path = "dataset/cgt"
    processor = DatasetProcessor(base_path, filter_most_frequent=False)
    processor.process()
