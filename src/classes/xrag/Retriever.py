from typing import List, Any, Dict

from tqdm import tqdm

from src.classes.utils.DebugLogger import DebugLogger
from src.classes.xrag.SimilarityCalculator import SimilarityCalculator


class Retriever:
    def __init__(self, documents: List[Any], similarity_mode: str = "aggregated", weights: Dict[str, float] = None):
        """
        Initializes the Retriever with a list of documents and a similarity calculator.

        :param documents: List of documents to retrieve from.
        :param similarity_mode: The mode used for similarity calculation.
        :param weights: Optional weights for similarity computation.
        """
        self.logger = DebugLogger()
        self.documents = documents
        self.similarity_calculator = SimilarityCalculator(similarity_mode, weights)

        self.logger.info(f"Retriever initialized with {len(documents)} documents, similarity_mode='{similarity_mode}'")

    def retrieve(self, input_doc: Any, k: int = 3) -> List[Any]:
        """
        Retrieves the top-k most similar documents to the input document.

        :param input_doc: The document used as the query.
        :param k: Number of documents to retrieve.
        :return: List of top-k most similar documents.
        """
        input_json = input_doc.metadata.get("json", {})
        scored_docs = []

        for idx, doc in tqdm(enumerate(self.documents), desc="Retrieving documents"):
            candidate_json = doc.metadata.get("json", {})
            score = self.similarity_calculator.compute_similarity(input_json, candidate_json)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_k_docs = [doc for _, doc in scored_docs[:k]]

        self.logger.info(f"Retrieved top {k} documents. Scores: {[score for score, _ in scored_docs[:k]]}")
        return top_k_docs
