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
You must follow these steps precisely to evaluate the target Solidity contract, which is provided in AST (Abstract Syntax Tree) format:

2. **Classify the Contract**:
   Based on the retrieved context:
   - Classify the contract as **Reentrant** if it contains patterns or vulnerabilities matching reentrant behavior.
   - Classify the contract as **Safe** if it implements proper safeguards or mitigations that align with non-reentrant behavior.

3. **Justify the Classification with Cited Evidence**:
   - Provide a detailed explanation for the classification, comparing the patterns observed in the input contract's AST with those in the retrieved context.
   - Cite sources explicitly, referring to the relevant knowledge base entries used to form the reasoning.
   - Highlight specific AST nodes, lines, or functions in the target contract that led to the classification decision.

4. **Structured Output**:
   Return the classification result in the following strict JSON format:

---

### Output Format

```json
{
  "classification": "Reentrant / Safe",
  "analysis": "Key observations about the input contract's AST, including specific nodes, functions, external calls, and state update sequences."
}
```

### Rules:
- Only use the information from the context for classification and justification.
- Do not make assumptions beyond the retrieved context.
- Ensure all cited sources are detailed and verifiable.

### Input Contract (in AST format)

"""


def evaluate(path_to_contracts: str, rag: "VectorRAG") -> None:
    """
    Tests the RAG model's ability to classify Solidity contracts in a directory.

    Parameters:
        path_to_contracts (str): Directory containing Solidity (.sol) contract files.
        rag (VectorRAG): An instance of the RAG model used for querying and fetching sources.
    """
    if not os.path.isdir(path_to_contracts):
        logger.error(f"Directory not found: {path_to_contracts}")
        return

    # Extract the ground truth category from the directory name (handles trailing slashes)
    gt_category = os.path.basename(os.path.normpath(path_to_contracts))

    # Create a unique log directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_dir = os.path.join("log", f"test_{gt_category}_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)

    # Get list of Solidity files in the directory (sorted for consistency)
    files = sorted(
        f for f in os.listdir(path_to_contracts)
        if f.endswith(".ast.json") and os.path.isfile(os.path.join(path_to_contracts, f))
    )
    total_files = len(files)

    if total_files == 0:
        logger.warning(f"No Solidity (.sol) files found in {path_to_contracts}.")
        return

    logger.info(f"Testing RAG model on {total_files} files from category: {gt_category}")
    logger.info(f"Results will be logged in: {log_dir}")

    # Initialize ModelManager and LLM once outside the loop
    model_manager = ModelManager()
    llm = model_manager.get_llm(LLM_MODE)

    correct = 0
    for index, filename in enumerate(files, start=1):
        path_to_file = os.path.join(path_to_contracts, filename)
        try:
            with open(path_to_file, 'r', encoding='latin-1') as file:
                contract_content = file.read()
        except Exception as e:
            logger.error(f"Error reading file {path_to_file}: {e}")
            continue

        logger.debug(f"[{index}/{total_files}] Processing file: {filename}")

        try:
            # Query the RAG system by combining the evaluation prompt with the contract content
            answer = rag.query(EVAL_PROMPT + contract_content)
            sources = rag.fetch_sources(answer.source_nodes)
            source_nodes = rag.fetch_source_nodes(answer.source_nodes)
        except Exception as e:
            logger.error(f"Error querying RAG system for file {filename}: {e}")
            continue

        logger.debug(f"*** ANSWER ***:\n{answer}")
        logger.debug(f"*** SOURCES ***: {sources}")

        try:
            json_answer = extract_and_parse_json(str(answer))
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for file {filename}: {e}")
            continue

        label = json_answer.get("classification", "").strip()
        if not label:
            logger.error(f"No classification label found in JSON answer for file {filename}")
            continue

        # Build the explainability prompt
        retrieved_contracts = [
            (node.text, node.metadata.get('category', 'unknown')) for node in source_nodes
        ]
        explainability_prompt = (
            "Given:\n"
            "  1. An input smart contract provided in AST (Abstract Syntax Tree) format\n"
            "  2. Some similar contracts retrieved by a RAG system and their labels\n"
            "  3. The classification of the contract as reentrant or safe\n"
            "Provide a detailed explanation of why the smart contract was classified as such.\n"
            "When formulating your explanation, refer explicitly to the AST nodes, functions, and other relevant structural elements that led to the classification.\n"
            f"Input contract (AST format) classified as {label}:\n{contract_content}\n"
            f"Retrieved contracts: {retrieved_contracts}\n"
        )

        try:
            explanation = str(llm.complete(explainability_prompt))
        except Exception as e:
            logger.error(f"Error generating explanation for file {filename}: {e}")
            explanation = "Explanation generation failed."

        logger.debug(f"*** EXPLANATION ***:{explanation}")

        # Add the sources and explanation to the JSON answer
        json_answer["sources"] = sources
        json_answer["explanation"] = explanation

        # Write the JSON output to a file in the log directory
        output_path = os.path.join(log_dir, f"{filename}.json")
        try:
            with open(output_path, 'w', encoding='utf-8') as output_file:
                json.dump(json_answer, output_file, indent=4)
        except Exception as e:
            logger.error(f"Error writing output file {output_path}: {e}")

        # Check classification accuracy (using case-insensitive comparison)
        if label.lower() == gt_category.lower():
            correct += 1

        running_accuracy = correct / index
        logger.info(f"Processed {index}/{total_files} files. Running accuracy: {running_accuracy:.2%}")

    accuracy = correct / total_files
    logger.info(f"Final classification accuracy for '{gt_category}': {accuracy:.2%}")
    logger.debug(f"Processed {total_files} files. Final Accuracy: {accuracy:.2%}")


def main() -> None:
    """
    Main function to initialize the RAG system, load training documents,
    and evaluate the test datasets.
    """
    try:
        rag = VectorRAG()
    except Exception as e:
        logger.error(f"Error initializing VectorRAG: {e}")
        return

    # Load training documents for the 'reentrant' and 'safe' categories
    path_to_data_train = os.path.join("dataset", "manually-verified-train", "ast")
    path_to_train_reentrant = os.path.join(path_to_data_train, "reentrant")
    path_to_train_safe = os.path.join(path_to_data_train, "safe")

    try:
        rag.load_and_index_documents(path_to_train_reentrant, reload_index=True)
        rag.load_and_index_documents(path_to_train_safe, reload_index=True)
    except Exception as e:
        logger.error(f"Error loading and indexing training documents: {e}")
        return

    # Evaluate the model on the test datasets for both categories
    path_to_data_test = os.path.join("dataset", "manually-verified-test", "ast")
    path_to_test_reentrant = os.path.join(path_to_data_test, "reentrant")
    path_to_test_safe = os.path.join(path_to_data_test, "safe")

    evaluate(path_to_test_reentrant, rag)
    evaluate(path_to_test_safe, rag)


if __name__ == "__main__":
    main()
