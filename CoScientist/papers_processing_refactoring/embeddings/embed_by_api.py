import requests
from typing import List

from .base_embedder import BatchedEmbeddingModel


class APIEmbeddingModel(BatchedEmbeddingModel):
    
    def __init__(
        self,
        url: str,
        timeout: int = 1000,
        batch_size: int = 16,
        headers: dict | None = None,
    ):
        super().__init__(batch_size=batch_size)
        self.url = url
        self.timeout = timeout
        self.headers = headers or {}

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        
        response = requests.post(
            self.url,
            json=texts,
            headers=self.headers,
            timeout=self.timeout,
        )

        response.raise_for_status()
        data = response.json()

        # Expected format:
        # { "embeddings": [[...], [...], ...] }
        return data["embeddings"]