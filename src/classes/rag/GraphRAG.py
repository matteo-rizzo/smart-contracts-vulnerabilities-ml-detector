from llama_index.core import StorageContext

from src.classes.rag.GraphManager import GraphManager
from src.classes.rag.RAG import RAG


class GraphRAG(RAG):
    """
    A specialized implementation of the RAG (Retrieval-Augmented Generation) pipeline
    that leverages Neo4j for both vector-based document retrieval and knowledge graph storage.

    This class supports structured and unstructured query handling by integrating graph-based
    storage and retrieval mechanisms with the RAG pipeline.
    """

    def __init__(self) -> None:
        """
        Initializes the GraphRAG pipeline with components for graph-based and vector-based retrieval.
        """
        super().__init__()

    def _initialize_managers(self) -> None:
        """
        Configures the graph manager and its associated storage contexts.

        This method sets up the graph store and initializes the graph manager, enabling
        efficient graph-based storage and retrieval operations. It ensures that the pipeline
        can handle both graph structures and their integration with vector-based retrieval.

        :raises Exception:
            If the initialization of the graph manager fails, an error is logged with details.
        """
        try:
            self.logger.info("Initializing graph manager...")

            # Create the graph store and its associated storage context
            graph_store = self.db_manager.create_graph_store()
            graph_storage_context = StorageContext.from_defaults(graph_store=graph_store)

            # Initialize the GraphManager
            self.knowledge_manager = GraphManager(graph_store, graph_storage_context)

            self.logger.success("Graph manager initialized successfully.")
        except Exception as e:
            self.logger.error(f"Failed to initialize graph manager: {e}", exc_info=True)
