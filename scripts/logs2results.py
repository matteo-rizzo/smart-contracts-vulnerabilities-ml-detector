import os

import pandas as pd
from rich import print
from rich.progress import Progress


def create_directory(path):
    """Creates a directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"[green]Created directory:[/green] {path}")
    else:
        print(f"[yellow]Directory already exists:[/yellow] {path}")


def parse_directory_name(directory_name):
    """Parses the directory name to extract dataset name and experiment type."""
    parts = directory_name.split('_')
    if len(parts) < 3:
        print(f"[yellow]Invalid directory format, skipping:[/yellow] {directory_name}")
        return None, None
    dataset_name = parts[1]
    experiment_type = parts[2]
    return dataset_name, experiment_type


def read_experiment_files(dir_path, experiment_type):
    """Reads all CSV files in a directory and returns a list of dataframes."""
    dataframes = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith('.csv'):
            model_name = file_name.replace('.csv', '')
            df = pd.read_csv(os.path.join(dir_path, file_name))
            df['Experiment Type'] = experiment_type
            df['Model'] = model_name
            dataframes.append(df)
    return dataframes


def process_directories(root_dir):
    """Processes all directories in the root directory to compile dataset data."""
    dataset_data = {}

    with Progress() as progress:
        task = progress.add_task("[cyan]Processing directories...", total=len(os.listdir(root_dir)))

        for dir_name in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                dataset_name, experiment_type = parse_directory_name(dir_name)
                if dataset_name and experiment_type:
                    if dataset_name not in dataset_data:
                        dataset_data[dataset_name] = []
                    dataframes = read_experiment_files(dir_path, experiment_type)
                    dataset_data[dataset_name].extend(dataframes)

            progress.update(task, advance=1)

    return dataset_data


def save_combined_csv(dataset_name, dataframes, output_dir):
    """Saves combined dataframes to a CSV file."""
    combined_df = pd.concat(dataframes, ignore_index=True)
    output_path = os.path.join(output_dir, f'{dataset_name}_combined_experiments.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"[green]Saved combined data for {dataset_name} to[/green] [bold]{output_path}[/bold]")


def save_all_datasets(dataset_data, output_dir):
    """Saves combined data for each dataset to the output directory."""
    with Progress() as progress:
        task = progress.add_task("[cyan]Saving CSV files...", total=len(dataset_data))

        for dataset_name, dataframes in dataset_data.items():
            save_combined_csv(dataset_name, dataframes, output_dir)
            progress.update(task, advance=1)

    print("[bold green]All datasets have been processed and saved successfully![/bold green]")


def main():
    root_dir = 'log'
    output_dir = 'results'

    create_directory(output_dir)
    dataset_data = process_directories(root_dir)
    save_all_datasets(dataset_data, output_dir)


if __name__ == "__main__":
    main()
