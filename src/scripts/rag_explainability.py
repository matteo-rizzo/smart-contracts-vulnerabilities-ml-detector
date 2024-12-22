import json
import os
import time

from llama_index.core import Settings

from src.classes.rag.ModelManager import ModelManager
from src.classes.rag.VectorRAG import VectorRAG
from src.classes.utils.DebugLogger import DebugLogger
from src.classes.utils.EnvLoader import EnvLoader
from src.functions.utils import extract_and_parse_json
from src.settings import LLM_MODE, EMBEDDING_MODE

EnvLoader(env_dir="src/config").load_env_files()

Settings.llm = ModelManager().get_llm(LLM_MODE)
Settings.embed_model = ModelManager().get_embedding_model(EMBEDDING_MODE)

logger = DebugLogger()

EVAL_PROMPT = """

You must follow these steps:

1. **Retrieve Examples**:
   Search your knowledge base for relevant examples of Solidity smart contracts labeled as **reentrant** or **non-reentrant**. Focus on:
   - Contracts with **reentrancy vulnerabilities**, such as making external calls (`call`, `delegatecall`, `transfer`) before updating state variables.
   - Contracts that use **mitigations** like the *checks-effects-interactions* pattern, `ReentrancyGuard` modifiers, or mutex locks.

   Provide **contract snippets** and **explanations** of why these examples were labeled as reentrant or non-reentrant.

2. **Analyze the Target Contract**:
   Carefully analyze the **input Solidity contract** to identify:
   - Use of external calls (`msg.sender.call`, `delegatecall`, `send`, etc.).
   - Whether state variables are updated **before** or **after** the external call.
   - Reentrancy mitigations like `ReentrancyGuard` modifiers or the *checks-effects-interactions* pattern.

3. **Classify**:
   Based on the retrieved examples and your analysis, classify the target contract as:
   - **Reentrant**: If it contains vulnerabilities that allow external calls before updating state variables.
   - **Non-Reentrant**: If it uses proper safeguards or patterns to prevent reentrancy.

4. **Justify the Classification**:
   Explain your reasoning in detail. Compare the patterns you observed in the target contract with the retrieved examples. Highlight specific lines or functions that led to your conclusion.

5. **Output**:
   Return the result in the following structured JSON format:

---

### Output Format

```json
{
  "classification": "Reentrant / Non-Reentrant",
  "justification": "Provide a detailed explanation of your reasoning.",
  "analysis": "Key observations about the target contract, including function behaviors, external calls, and state updates."
}
```

Important: The output must be the JSON only.

---

### Input

"""

EXAMPLE_CONTRACT = """

// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract VulnerableBank {
    mapping(address => uint256) public balances;

    // Deposit ether into the bank
    function deposit() public payable {
        balances[msg.sender] += msg.value;
    }

    // Withdraw ether from the bank
    function withdraw(uint256 _amount) public {
        require(balances[msg.sender] >= _amount, "Insufficient balance");
        
        // Send the amount to the caller
        (bool sent, ) = msg.sender.call{value: _amount}("");
        require(sent, "Failed to send Ether");
        
        // Update the balance
        balances[msg.sender] -= _amount;
    }

    // Get the contract's balance
    function getBalance() public view returns (uint256) {
        return address(this).balance;
    }
}

"""


def evaluate(path_to_contracts, rag):
    """ Tests the RAG model's ability to classify Solidity contracts in a directory. """
    correct = 0

    # Extract ground truth category from the path
    gt_category = os.path.basename(path_to_contracts)

    # Create a unique log directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path_to_log = os.path.join("log", f"test_{gt_category}_{timestamp}")
    os.makedirs(path_to_log, exist_ok=True)

    # Cache list of files to avoid multiple os.listdir calls
    files = [f for f in os.listdir(path_to_contracts)
             if f.endswith(".sol") and os.path.isfile(os.path.join(path_to_contracts, f))]
    total_files = len(files)

    logger.info(f"Testing RAG model on {total_files} files from category: {gt_category}")
    logger.info(f"Results will be logged in: {path_to_log}")

    for index, filename in enumerate(files, start=48):
        try:
            # Build the full path to the file
            path_to_file = os.path.join(path_to_contracts, filename)

            # Read contract content
            with open(path_to_file, 'r', encoding='latin-1') as file:
                contract_content = file.read()

            # Query the RAG system
            answer = rag.query(EVAL_PROMPT + contract_content)
            sources = rag.fetch_sources(answer.source_nodes)

            logger.debug(f"[{index}/{total_files}] Processing file: {filename}")
            logger.debug(f"*** ANSWER ***:\n{answer}\n --> SOURCES: {sources}")

            # Extract and parse JSON from the answer
            json_answer = extract_and_parse_json(str(answer))

            # Write the JSON output to a file
            output_path = os.path.join(path_to_log, f"{filename}.json")
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(json_answer, output_file, indent=4)

            # Check classification accuracy
            if json_answer.get("classification") == gt_category:
                correct += 1

        except FileNotFoundError:
            logger.error(f"File not found: {path_to_file}")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for file {filename}: {e}")
        except Exception as e:
            logger.error(f"An error occurred processing {filename}: {e}")

    # Calculate accuracy
    accuracy = correct / total_files if total_files > 0 else 0
    logger.info(f"Classification Accuracy for '{gt_category}': {accuracy:.2%}")

    # Summary log
    print(f"Processed {total_files} files. Accuracy: {accuracy:.2%}")


def main():
    rag = VectorRAG()

    path_to_data_train = os.path.join("dataset", "manually-verified-preprocessed-train", "source")

    path_to_train_reentrant = os.path.join(path_to_data_train, "reentrant")
    rag.load_and_index_documents(path_to_train_reentrant, reload_index=True)

    path_to_train_safe = os.path.join(path_to_data_train, "safe")
    rag.load_and_index_documents(path_to_train_safe, reload_index=True)

    # path_to_data_test = os.path.join("dataset", "manually-verified-preprocessed-test", "source")
    #
    # path_to_test_reentrant = os.path.join(path_to_data_test, "reentrant")
    # evaluate(path_to_test_reentrant, rag)
    #
    # path_to_test_safe = os.path.join(path_to_data_test, "safe")
    # evaluate(path_to_test_safe, rag)

    answer = rag.query(EVAL_PROMPT + EXAMPLE_CONTRACT)
    sources = rag.fetch_sources(answer.source_nodes)
    logger.info(f"\n\n *** ANSWER *** \n\n {answer} \n\n SOURCES: {sources} \n\n")


if __name__ == "__main__":
    main()
