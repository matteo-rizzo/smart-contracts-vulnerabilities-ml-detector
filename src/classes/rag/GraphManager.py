from typing import List

from llama_index.core import StorageContext, Document, PropertyGraphIndex
from llama_index.core.graph_stores import PropertyGraphStore
from llama_index.core.indices.property_graph import SchemaLLMPathExtractor

from src.classes.rag.KnowledgeManager import KnowledgeManager
from src.classes.rag.ModelManager import ModelManager
from src.classes.rag.SchemaHandler import SchemaHandler
from src.classes.rag.UnstructuredTransform import UnstructuredTransform
from src.settings import LLM_MODE


class GraphManager(KnowledgeManager):
    """
    Manages graph store operations, including cleaning, indexing, and query execution.
    Extends KnowledgeManager to support configurations specific to graph-based indexing.
    """

    def __init__(self, store: PropertyGraphStore, storage_context: StorageContext) -> None:
        """
        Initialize the GraphManager with a graph-based store and storage context.

        :param store: The graph data storage backend.
        :param storage_context: Context or configuration settings for storage management.
        """
        super().__init__(store, storage_context)
        self.persist_dir = "graph_index"
        self.logger.info("GraphManager initialized with a graph store.")

    def create_index(self, documents: List[Document]) -> None:
        """
        Index a list of documents into the knowledge graph with specified configurations.

        :param documents: List of Document objects to be indexed.
        """
        self.logger.info("Starting document indexing into the graph store. This may take some time.")

        try:
            self.index = PropertyGraphIndex.from_documents(
                documents=documents,
                kg_extractors=[
                    # SimpleLLMPathExtractor(),
                    SchemaLLMPathExtractor(
                        llm=ModelManager().get_llm(LLM_MODE),
                        possible_entities=SchemaHandler.get_entities(),
                        possible_relations=SchemaHandler.get_relations(),
                        kg_validation_schema=SchemaHandler.get_validation_schema(),
                        strict=False,
                        max_triplets_per_chunk=3
                    ),
                ],
                property_graph_store=self.store,
                storage_context=self.storage_context,
                embed_kg_nodes=True,
                show_progress=True,
                transformations=[UnstructuredTransform()],
            )
            self.logger.success("Document indexing completed successfully.")
        except Exception as e:
            self.logger.error("Error during document indexing:", exc_info=True)
            raise RuntimeError("Graph indexing failed.") from e

    def load_index(self) -> bool:
        """
        Load the index from the graph store if available.

        :return: True if the index was successfully loaded, False otherwise.
        """
        if not self.store:
            self.logger.warning("No graph store is available. Unable to load index.")
            return False

        try:
            self.logger.info("Attempting to load the index from the graph store.")
            self.index = PropertyGraphIndex.from_existing(
                property_graph_store=self.store, embed_kg_nodes=True
            )
            self.logger.success("Index loaded successfully from the graph store.")
            return True
        except Exception as e:
            self.logger.error("Error while loading the index:", exc_info=True)
            return False
