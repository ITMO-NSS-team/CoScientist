from typing import Optional, Dict, Any, List, Union

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ConfigDict

from ..domain.entities import Article, Chunk
from ..embeddings import *
from ..storage.artifacts.domain_s3 import S3DomainArtifactStore
from ..storage.artifacts.etl_s3 import S3ETLArtifactStore, MockArtifactStore
from ..storage.state.state_db import SQLiteStateManager
from ..storage.vector.base import VectorStore


class ETLContext(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    article: Article

    raw_data: Optional[bytes] = None
    parsed_representation: Optional[Any] = None
    chunks: Dict[str, List[Chunk]] = Field(default_factory=dict)
    embeddings: Dict[str, Dict[str, List[List[float]] | List[str]]] = Field(default_factory=dict)
    
    artifacts: Dict[str, Any] = Field(default_factory=dict)
    
    state_manager: SQLiteStateManager
    artifact_store: Union[S3ETLArtifactStore, MockArtifactStore]
    public_store: S3DomainArtifactStore
    vector_store: VectorStore
    
    llm: ChatOpenAI
    embedding_model: EmbeddingModel
