#!/bin/bash

# Define paths
SOL_FILE="dataset/aisc/source/unchecked_low_level_calls/sc_unchecked_low_level_calls_65.sol"
AST_DIR="dataset/aisc/ast/ast_unchecked_low_level_calls"
AST_FILE="${AST_DIR}/sc_unchecked_low_level_calls_65.ast.json"

# Check if solc and jq are installed
if ! command -v solc &> /dev/null; then
    echo "solc could not be found, please install it."
    exit 1
fi

if ! command -v jq &> /dev/null; then
    echo "jq could not be found, please install it."
    exit 1
fi

# Ensure the output directory exists
mkdir -p "$AST_DIR"

# Generate the AST in compact JSON format
solc --ast-compact-json $SOL_FILE | sed '1,4d' > $AST_FILE

# Check if the AST file was created successfully
if [ ! -s "$AST_FILE" ]; then
    echo "Failed to generate AST. Please check the solidity file."
    exit 1
fi

# Print the AST using jq to format it
echo "Printing AST:"
jq . $AST_FILE
