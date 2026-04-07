from typing import List

from .base_embedder import BatchedEmbeddingModel


class LocalEmbeddingModel(BatchedEmbeddingModel):

    def __init__(self, model_name: str, batch_size: int = 32):
        super().__init__(batch_size=batch_size)

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers must be installed for LocalEmbeddingModel"
            ) from e

        self.model = SentenceTransformer(model_name)

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,
        )

        return embeddings.tolist()