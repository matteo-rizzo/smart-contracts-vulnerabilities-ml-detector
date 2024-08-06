"""
Installation Steps:

1. Install Rust and Cargo:
   Some dependencies might require the Rust toolchain. Install Rust and Cargo globally.

   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

   After installation, add Rust to your PATH:

   source $HOME/.cargo/env

2. Verify Rust and Cargo Installation:
   Ensure Rust and Cargo are installed correctly by running:

   rustc --version
   cargo --version

3. Create a New Virtual Environment:
   If you're facing issues with your current environment, create a new virtual environment:

   python -m venv new_venv
   source new_venv/bin/activate

4. Install Dependencies:
   Install the required dependencies in the new virtual environment:

   pip install eth-utils eth-hash evmdasm tqdm

5. Run the Script:
   You can run the script with or without specifying directories. If directories are not specified, the script uses default values.

   - Using default directories:
     python script.py

   - Using custom directories:
     python script.py custom_base_directory custom_output_directory

This script processes bytecode files in the specified directory, disassembles them to opcodes using `evmdasm`, and saves the results in the output directory.
"""

import os
import sys

from eth_utils import decode_hex
from evmdasm import EvmBytecode
from tqdm import tqdm

# Default directories
DEFAULT_BASE_DIR = "dataset/cgt/bytecode"
DEFAULT_OUTPUT_DIR = "dataset/cgt/opcodes"


# Function to display usage information
def usage():
    print("Usage: python script.py [base_directory] [output_directory]")
    print(f"Default base_directory: {DEFAULT_BASE_DIR}")
    print(f"Default output_directory: {DEFAULT_OUTPUT_DIR}")
    sys.exit(1)


# Function to disassemble bytecode to opcodes and save it
def disassemble_bytecode(bytecode, output_file):
    bytecode_obj = EvmBytecode(decode_hex(bytecode))
    instructions = bytecode_obj.disassemble()

    with open(output_file, 'w') as f:
        for instruction in instructions:
            f.write(f"{instruction.name}\n")


# Function to process each bytecode file
def process_file(bytecode_file, output_dir):
    # Extract filename without extension and prepare output filename
    filename = os.path.basename(bytecode_file)
    contract_name = os.path.splitext(filename)[0]
    output_file = os.path.join(output_dir, f"{contract_name}.opcodes.txt")

    # Read bytecode from file
    with open(bytecode_file, 'r') as file:
        bytecode = file.read().strip()

    # Disassemble bytecode to opcodes and save it
    try:
        disassemble_bytecode(bytecode, output_file)
    except Exception as e:
        print(f"Failed to disassemble {bytecode_file}: {e}")


# Function to process directories recursively
def process_directory(base_dir, output_base):
    for root, dirs, files in os.walk(base_dir):
        # Create corresponding output directory
        relative_path = os.path.relpath(root, base_dir)
        output_dir = os.path.join(output_base, relative_path)
        os.makedirs(output_dir, exist_ok=True)

        hex_files = [file for file in files if file.endswith(".hex")]

        # Process each hex file in the directory with a progress bar
        for file in tqdm(hex_files, desc=f"Processing files in {root}"):
            process_file(os.path.join(root, file), output_dir)


# Main script execution
if __name__ == "__main__":
    if len(sys.argv) > 3:
        usage()

    BASE_DIR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_BASE_DIR
    OUTPUT_BASE_DIR = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR

    if not os.path.exists(BASE_DIR):
        print(f"Base directory {BASE_DIR} does not exist.")
        sys.exit(1)

    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

    # Start processing from the base directory with a progress bar
    process_directory(BASE_DIR, OUTPUT_BASE_DIR)

    print(f"All contracts processed. Opcodes are stored in {OUTPUT_BASE_DIR}")
