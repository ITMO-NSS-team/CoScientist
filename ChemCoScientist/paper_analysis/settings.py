import os

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

from definitions import CONFIG_PATH

load_dotenv(CONFIG_PATH)

allowed_providers = ["google-vertex"]


class ChromaSettings(BaseSettings):
    """
    Manages settings for Chroma database and related components.
    
        This class encapsulates configuration details for connecting to and interacting with
        Chroma, an embedding database, as well as related embedding and reranking services.
    
        Class Attributes:
        - chroma_host
        - chroma_port
        - allow_reset
        - embedding_host
        - embedding_port
        - embedding_endpoint
        - reranker_host
        - reranker_port
        - reranker_endpoint
    """

    # Chroma DB settings
    chroma_host: str = os.getenv("CHROMA_HOST")
    chroma_port: int = os.getenv("CHROMA_PORT")
    allow_reset: bool = False
    
    # Documents collection's settings
    embedding_host: str = os.getenv("EMBEDDING_HOST")
    embedding_port: int = os.getenv("EMBEDDING_PORT")
    embedding_endpoint: str = "/embed"
    
    # Reranker settings
    reranker_host: str = os.getenv("RERANKER_HOST")
    reranker_port: int = os.getenv("RERANKER_PORT")
    reranker_endpoint: str = "/rerank"

settings = ChromaSettings()