#!/bin/bash

source venv/bin/activate

#!/bin/bash

# Define the root directory of your project
PROJECT_ROOT="src/scripts"

# Define the file types and subsets to iterate over
file_types=("source" "runtime" "bytecode")
subsets=("CodeSmells" "Zeus" "eThor" "ContractFuzzer" "SolidiFI" "EverEvolvingG" "Doublade" "NPChecker" "JiuZhou" "SBcurated" "SWCregistry" "EthRacer" "NotSoSmartC")

# Define the scripts to run
scripts=("ffnn.py" "lstm.py" "codebert.py" "ml_classifier.py")

# Export PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

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

