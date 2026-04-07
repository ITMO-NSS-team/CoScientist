# from qdrant_client import QdrantClient
# from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchValue
#
# from .base import VectorStore
# from ...domain.entities import Chunk
#
#
# class QdrantVectorStore(VectorStore):
#     def __init__(self, url: str, api_key: str, collection_name: str = "papers"):
#         self.client = QdrantClient(url=url, api_key=api_key)
#         self.collection_name = collection_name
#         # Убедитесь, что коллекция создана с поддержкой нужных векторов (dense/sparse)
#
#     def upsert_chunks(self, chunks: list[Chunk], embeddings: list[dict | list[float]]) -> None:
#         if not chunks:
#             return
#
#         points = []
#         for chunk, emb in zip(chunks, embeddings):
#             payload = chunk.model_dump()
#             # Qdrant принимает id либо как int, либо как UUID str.
#             # Если ваши chunk.id это хеши, они могут не подойти.
#             # Лучше генерировать UUID5 на основе chunk.id, если он не в формате UUID.
#
#             points.append(
#                 PointStruct(
#                     id=chunk.id,  # Убедитесь, что это валидный UUID
#                     vector=emb,  # Может быть list (dense) или dict (named vectors)
#                     payload=payload
#                 )
#             )
#
#         self.client.upsert(
#             collection_name=self.collection_name,
#             points=points
#         )
#
#     def search(self, query_vector: dict | list[float], limit: int = 5, filters: dict = None) -> list[Chunk]:
#         from qdrant_client.http.models import Filter, FieldCondition, MatchValue
#
#         qdrant_filter = None
#         if filters:
#             must_conditions = [
#                 FieldCondition(key=k, match=MatchValue(value=v))
#                 for k, v in filters.items()
#             ]
#             qdrant_filter = Filter(must=must_conditions)
#
#         search_result = self.client.search(
#             collection_name=self.collection_name,
#             query_vector=query_vector,
#             limit=limit,
#             query_filter=qdrant_filter,
#             with_payload=True
#         )
#
#         return [Chunk(**hit.payload) for hit in search_result]
#
#     def delete_by_article_id(self, article_id: str) -> None:
#         self.client.delete(
#             collection_name=self.collection_name,
#             points_selector=Filter(
#                 must=[FieldCondition(key="article_id", match=MatchValue(value=article_id))]
#             )
#         )
