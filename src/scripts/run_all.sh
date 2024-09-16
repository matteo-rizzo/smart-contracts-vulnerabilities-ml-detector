#!/bin/bash

source .venv/bin/activate

# Define the root directory of your project
PROJECT_ROOT="src/scripts"

# Define the file types and subsets to iterate over
# file_types=("source" "runtime" "bytecode" "ast" "cfg" "opcode")
file_types=("opcode")
# subsets=("CodeSmells" "Zeus" "eThor" "ContractFuzzer" "SolidiFI" "EverEvolvingG" "Doublade" "NPChecker" "JiuZhou" "SBcurated" "SWCregistry" "EthRacer" "NotSoSmartC")
subsets=("CodeSmells")

# Define the scripts to run
#scripts=("ffnn.py" "lstm.py" "codebert.py" "ml_classifiers.py", "gcn.py")
scripts=("ml_classifiers.py")

# Export PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Multimodal
#for subset in "${subsets[@]}"; do
#    # Iterate over each script
#    for script in "${scripts[@]}"; do
#        echo "Running $script for subset: $subset"
#        python "$PROJECT_ROOT/$script" --file_type "source" --subset "$subset" --multimodal True
#    done
#done

# Iterate over each file type
for file_type in "${file_types[@]}"; do
    # Iterate over each subset
    for subset in "${subsets[@]}"; do
        # Iterate over each script
        for script in "${scripts[@]}"; do
            echo "Running $script for file type: $file_type and subset: $subset"
            python "$PROJECT_ROOT/$script" --file_type "$file_type" --subset "$subset"
        done
    done
done

