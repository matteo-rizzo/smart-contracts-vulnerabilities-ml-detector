import pandas as pd
from itertools import product

# Define the expected modalities and models
expected_modalities = ["source", "runtime", "bytecode", "ast", "opcode", "cfg", "multimodal"]
expected_models = ["bert", "ffnn", "gradient", "knn", "logistic", "lstm", "random", "svm", "xgboost"]

# Define specific rules for ast, cfg, and multimodal
restricted_models_ast_cfg = {"ast": ["bert", "ffnn", "lstm"], "cfg": ["bert", "ffnn", "lstm"]}
restricted_models_multimodal = ["bert", "lstm"]
gcn_modalities = ["ast", "cfg"]

def apply_rules(modality, model):
    """
    Apply the specific rules regarding modality and model combinations.
    """
    # Rule 1: ast and cfg should not have bert, ffnn, or lstm
    if modality in restricted_models_ast_cfg and model in restricted_models_ast_cfg[modality]:
        return False
    # Rule 2: Only ast and cfg can have gcn
    if model == "gcn" and modality not in gcn_modalities:
        return False
    # Rule 3: multimodal should not have bert or lstm
    if modality == "multimodal" and model in restricted_models_multimodal:
        return False
    return True

def find_missing_modalities_models(df):
    """
    Find missing modality-model combinations for each dataset, applying custom rules.
    """
    missing_data = []

    # Group by 'Dataset' and check for missing modalities and models
    for dataset in df['Dataset'].unique():
        # Filter dataset-specific rows
        dataset_df = df[df['Dataset'] == dataset]

        # Create a set of existing (modality, model) pairs for this dataset
        existing_pairs = set(zip(dataset_df['Modality'], dataset_df['Model']))

        # Generate all possible (modality, model) pairs
        all_pairs = set(product(expected_modalities, expected_models))

        # Find the missing pairs by subtracting existing pairs from all pairs
        missing_pairs = all_pairs - existing_pairs

        # Apply the additional rules for ast, cfg, multimodal, and gcn
        for modality, model in missing_pairs:
            if apply_rules(modality, model):
                missing_data.append([dataset, modality, model])

    return missing_data

def print_missing_data_table(missing_data):
    """
    Print the missing modalities and models for each dataset in a table format, ordered by Dataset, Modality, and Model.
    """
    # Convert the missing data to a DataFrame
    missing_df = pd.DataFrame(missing_data, columns=["Dataset", "Missing Modality", "Missing Model"])

    # Sort the DataFrame by Dataset, Missing Modality, and Missing Model
    missing_df = missing_df.sort_values(by=["Dataset", "Missing Modality", "Missing Model"])

    # Display the DataFrame as a table
    print(missing_df.to_string(index=False))

def main(file_path):
    """
    Main function to find and display missing modalities and models for each dataset in a table.
    """
    # Load the dataset
    df = pd.read_csv(file_path)

    # Find missing modalities and models
    missing_data = find_missing_modalities_models(df)

    # Print the missing modalities and models in a table format
    print_missing_data_table(missing_data)

if __name__ == "__main__":
    # Replace with your actual file path
    file_path = "../summary_results.csv"

    # Call the main function
    main(file_path)
