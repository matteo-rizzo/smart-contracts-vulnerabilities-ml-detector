import json
import multiprocessing
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

from src.classes.utils.DebugLogger import DebugLogger
from src.classes.xrag.ContractFileProcessor import ContractFileProcessor
from src.classes.xrag.Document import Document
from src.classes.xrag.KnowledgeBase import KnowledgeBase
from src.classes.xrag.Retriever import Retriever
from src.functions.xrag import process_input_contract_worker


class ContractAnalyzer:
    def __init__(self, dataset_base: str, mode: str = "both", use_multiprocessing: bool = False) -> None:
        """
        Initializes the ContractAnalyzer class.

        :param dataset_base: Base path for the dataset, expects a format placeholder.
        :param mode: The mode of analysis (e.g., 'both', 'train', 'test').
        :param use_multiprocessing: If True, enables multiprocessing; otherwise, runs sequentially.
        """
        self.mode = mode
        self.dataset_base = dataset_base
        self.use_multiprocessing = use_multiprocessing
        self.logger = DebugLogger()
        self.logger.info(f"Initializing ContractAnalyzer with mode: {self.mode}")

        # Initialize knowledge base and LLM processor
        self.kb = KnowledgeBase("knowledge_base.db")
        self.kb.clear_contracts()

        # Load directories
        self.candidate_dirs = self._get_directories("train")
        self.input_dirs = self._get_directories("test")
        self.logger.info("Candidate and input directories loaded.")

        # Prepare candidate documents and retriever
        candidate_documents = self._prepare_documents("candidate")
        self.logger.info(f"Candidate documents prepared. Total: {len(candidate_documents)}")
        self.retriever = Retriever(candidate_documents)

        # Create log directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(f"log/contracts_analysis_{self.mode}_{timestamp}")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"Log directory created at {self.log_dir}")

    def store_contracts(self, contracts: List[Tuple[str, List[str], Dict, str]]) -> None:
        """
        Stores contract information in the knowledge base.

        :param contracts: A list of tuples (contract_id, source_files, combined_json, label).
        """
        for contract_id, source_files, combined_json, label in contracts:
            self.kb.store_combined_contract(contract_id, source_files, combined_json, label)

    def load_contracts(self) -> List[Document]:
        """Loads all stored contracts from the knowledge base."""
        return self.kb.load_all_contracts()

    def _get_directories(self, dataset_type: str) -> Dict[str, str]:
        """
        Returns the directory paths for a given dataset type.

        :param dataset_type: Type of dataset ('train' or 'test').
        :return: Dictionary containing paths for source, AST, and CFG files.
        """
        base_path = self.dataset_base.format(dataset_type)
        return {
            "src": os.path.join(base_path, "source"),
            "ast": os.path.join(base_path, "ast"),
            "cfg": os.path.join(base_path, "cfg"),
        }

    def analyze_contracts(self):
        """Analyzes input contracts using multiprocessing (if enabled) or sequential processing."""
        input_documents = self._prepare_documents("input")
        self.logger.info(f"Input documents prepared. Total: {len(input_documents)}")

        if not input_documents:
            self.logger.warning("No input documents found. Skipping analysis.")
            return

        # Convert retriever to a serializable format
        candidate_docs_data = [doc.to_dict() for doc in self.retriever.documents]

        # Prepare tasks with only serializable data
        tasks = [
            (
                json.dumps(input_doc.text),  # Store text as JSON
                input_doc.metadata,  # Metadata should be simple dicts
                self.input_dirs,
                self.candidate_dirs,
                self.log_dir,
                candidate_docs_data,  # Pass candidate documents in a serializable format
            )
            for input_doc in input_documents
        ]

        if self.use_multiprocessing:
            self.logger.info("Using multiprocessing for contract analysis.")
            with multiprocessing.Pool() as pool:
                results = pool.map(process_input_contract_worker, tasks)
        else:
            self.logger.info("Running contract analysis sequentially (no multiprocessing).")
            results = [process_input_contract_worker(task) for task in tasks]

        self.logger.info(f"Completed processing all input contracts. Processed: {len(results)}")

    def _prepare_documents(self, contract_type: str) -> List[Document]:
        """
        Prepares documents for a given contract type ('candidate' or 'input').

        :param contract_type: 'input' for input contracts, 'candidate' for retriever candidates.
        :return: List of Document objects.
        """
        is_input = contract_type == "input"
        directories = self.input_dirs if is_input else self.candidate_dirs
        processor = ContractFileProcessor(ast_dir=directories["ast"], cfg_dir=directories["cfg"])
        self.logger.info(f"Preparing {contract_type} documents using ContractFileProcessor.")

        return self._process_contracts(processor, is_input)

    def _process_contracts(self, processor: ContractFileProcessor, is_input: bool) -> List[Document]:
        """
        Processes contracts using the given processor.

        :param processor: Instance of ContractFileProcessor.
        :param is_input: True for input contracts, False for candidates.
        :return: List of Document objects.
        """
        contracts = processor.group_files(mode=self.mode, is_input=is_input)

        if not contracts:
            self.logger.warning(f"No {'input' if is_input else 'candidate'} contracts found.")
            return []

        if is_input:
            docs = [
                Document(
                    text=json.dumps(contract[2], indent=2),
                    metadata={"contract_id": contract[0], "label": contract[3]},
                )
                for contract in contracts
            ]
            self.logger.info(f"Processed {len(docs)} input contracts.")
            return docs
        else:
            self.store_contracts(contracts)
            stored_contracts = self.load_contracts()
            self.logger.info(f"Stored and loaded {len(stored_contracts)} candidate contracts.")
            return stored_contracts
