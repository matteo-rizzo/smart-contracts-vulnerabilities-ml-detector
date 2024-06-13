#!/bin/bash

# Base directory containing subfolders with Solidity contracts
BASE_DIR="/path/to/solidity/contracts"

# Base output directory for ASTs
OUTPUT_BASE_DIR="/path/to/ast_output"

# Check if solc is installed
if ! command -v solc &> /dev/null
then
    echo "solc could not be found, please install it."
    exit 1
fi

# Function to process each Solidity file
process_file() {
    local sol_file=$1
    local output_dir=$2

    # Extract filename without extension and prepare output filename
    local filename=$(basename -- "$sol_file")
    local contract_name="${filename%.*}"
    local output_file="$output_dir/$contract_name.ast.json"

    # Compile contract to get AST and save it
    echo "Processing $sol_file..."
    solc --ast-json "$sol_file" > "$output_file"
}

# Function to process directories recursively
process_directory() {
    local dir=$1
    local output_base=$2

    # Create corresponding output directory with prefix 'ast_'
    local base_name=$(basename -- "$dir")
    local output_dir="$output_base/ast_$base_name"
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
            process_directory "$subdir" "$output_base/ast_$base_name"
        fi
    done
}

# Start processing from the base directory
process_directory "$BASE_DIR" "$OUTPUT_BASE_DIR"

echo "All contracts processed. ASTs are stored in $OUTPUT_BASE_DIR"
