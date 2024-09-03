#!/bin/bash

# Base directory containing subfolders with Solidity contracts
BASE_DIR="dataset/cgt/source"

# Base output directory for opcodes
OUTPUT_BASE_DIR="dataset/cgt/opcode"

# Default solc version to use if no pragma is specified or if single compiler mode is selected
DEFAULT_SOLC_VERSION="0.8.6"
USE_SINGLE_COMPILER=false

# Function to display usage information
usage() {
    echo "Usage: $0 [-v solc_version] [-s]"
    echo "  -v solc_version   Specify the solc version to use (default: 0.8.6)"
    echo "  -s                Use a single specified compiler for all files"
    exit 1
}

# Parse command-line options
while getopts "v:s" opt; do
    case $opt in
        v)
            DEFAULT_SOLC_VERSION="$OPTARG"
            ;;
        s)
            USE_SINGLE_COMPILER=true
            ;;
        *)
            usage
            ;;
    esac
done

# Check if solc is installed
if ! command -v solc &> /dev/null; then
    echo "solc could not be found, please install it."
    exit 1
fi

# Check if solc-select is installed
if ! command -v solc-select &> /dev/null; then
    echo "solc-select could not be found, please install it."
    exit 1
fi

# Function to extract the pragma version from a Solidity file
extract_pragma_version() {
    local sol_file=$1
    grep -Eo "^pragma [^;]+( [^;]+)*;" "$sol_file" | grep -Eo "[0-9]+\.[0-9]+\.[0-9]+"
}

# Function to check if a solc version is installed, and install it if not
check_and_install_solc_version() {
    local version=$1
    if ! solc-select versions | grep -q "$version"; then
        echo "solc version $version is not installed. Attempting to install..."
        solc-select install "$version"
        if [ $? -ne 0 ]; then
            echo "[red]Failed to install solc version $version, skipping this file.[/red]"
            return 1  # Indicate failure
        fi
    fi
    return 0  # Indicate success
}

# Function to process each Solidity file
process_file() {
    local sol_file=$1
    local output_dir=$2

    # Extract filename without extension and prepare output filename
    local filename=$(basename -- "$sol_file")
    local contract_name="${filename%.*}"
    local output_file="$output_dir/$contract_name.opcodes.txt"

    local solc_version="$DEFAULT_SOLC_VERSION"

    if [ "$USE_SINGLE_COMPILER" = false ]; then
        # Extract the pragma version from the file, or use the default version
        pragma_version=$(extract_pragma_version "$sol_file")
        if [ -z "$pragma_version" ]; then
            echo "No pragma version found in $sol_file, using default version $DEFAULT_SOLC_VERSION."
        else
            solc_version="$pragma_version"
        fi
    fi

    # Check and install the required solc version if not already installed
    check_and_install_solc_version "$solc_version"
    if [ $? -ne 0 ]; then
        echo "[red]Skipping $sol_file due to solc version installation failure.[/red]"
        return
    fi

    # Compile contract to get opcodes and save it
    echo "Processing $sol_file with solc version $solc_version..."

    solc-select use "$solc_version"
    solc --optimize --opcodes "$sol_file" > "$output_file"
}

# Function to process directories recursively
process_directory() {
    local dir=$1
    local output_base=$2

    # Create corresponding output directory with prefix 'opcodes_'
    local base_name=$(basename -- "$dir")
    local output_dir="$output_base/opcodes_$base_name"
    mkdir -p "$output_dir"

    # Process each Solidity file in the directory
    for file in "$dir"/*.sol; do
        if [[ -f "$file" ]]; then
            process_file "$file" "$output_dir"
        fi
    done

    # Recurse into subdirectories
    for subdir in "$dir"/*; do
        if [[ -d "$subdir" ]]; then
            process_directory "$subdir" "$output_base/opcodes_$base_name"
        fi
    done
}

# Main script execution

# If using a single compiler, ensure the specified solc version is installed and set
if [ "$USE_SINGLE_COMPILER" = true ]; then
    check_and_install_solc_version "$DEFAULT_SOLC_VERSION"
    if [ $? -ne 0 ]; then
        echo "[red]Failed to install default solc version, exiting...[/red]"
        exit 1
    fi
    solc-select use "$DEFAULT_SOLC_VERSION"
else
    # Store the current solc version
    current_version=$(solc --version | grep -oE "[0-9]+\.[0-9]+\.[0-9]+" | head -n 1)
fi

# Start processing from the base directory
process_directory "$BASE_DIR" "$OUTPUT_BASE_DIR"

# Revert to the original solc version if not using a single compiler for all files
if [ "$USE_SINGLE_COMPILER" = false ]; then
    solc-select use "$current_version"
fi

echo "[green]All contracts processed. Opcodes are stored in $OUTPUT_BASE_DIR[/green]"
