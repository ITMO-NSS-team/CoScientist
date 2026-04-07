from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """
    Base abstraction for embedding models.
    Must be usable both in ETL and Retrieval.
    """

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        raise NotImplementedError


class BatchedEmbeddingModel(EmbeddingModel):

    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self._embed_batch([text])[0]

    @abstractmethod
    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError