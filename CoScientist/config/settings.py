"""
Application configuration using Pydantic Settings.
"""
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag_tools.config import Settings as ToolRAGSettings


ROOT_DIR = Path(__file__).parent.parent.absolute()


# =========================
# LLM CONFIG
# =========================
class LLMSettings(BaseModel):
    allowed_providers: List[str] = ["google-vertex", "azure"]

    service_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    main_url: Optional[str] = None
    scenario_url: Optional[str] = None
    main_model: Optional[str] = None
    scenario_model: Optional[str] = None

    service_url: Optional[str] = None
    service_cc_url: Optional[str] = None

    vision_url: Optional[str] = None
    summary_url: Optional[str] = None
    marker_model: Optional[str] = None
    datasets_url: Optional[str] = None
    deepeval_url: Optional[str] = None


# =========================
# LLM CONFIG
# =========================
class ServicesSettings(BaseModel):
    tavily_api_key: Optional[str] = None
    openalex_api_key: Optional[str] = None


# =========================
# MED LLM CONFIG
# =========================
class MedLLMSettings(BaseModel):
    task_url: Optional[str] = None
    result_url: Optional[str] = None

    login: Optional[str] = None
    password: Optional[str] = None

    poll_interval: int = 10
    max_polls: int = 60

# =========================
# STORAGE
# =========================
class StorageSettings(BaseModel):
    root_dir: Path = ROOT_DIR

    parse_results: Optional[str] = None
    chroma_storage: Optional[str] = None
    papers_storage: Optional[str] = None
    ds_storage: Optional[str] = None
    img_storage: Optional[str] = None
    another_storage: Optional[str] = None
    memory_db: Optional[str] = None

    path_to_data: Optional[str] = None
    path_to_cvae_checkpoint: Optional[str] = None
    path_to_results: Optional[str] = None
    path_to_temp_files: Optional[str] = None
    my_papers: Optional[str] = None

    logging_path: Optional[str] = 'logs/'


# =========================
# HOSTS & PORTS
# =========================
class HostsPortsSettings(BaseModel):
    chroma_host: Optional[str] = None
    embedding_host: Optional[str] = None
    reranker_host: Optional[str] = None
    opencChemie_host: Optional[str] = None
    chem_services_host: Optional[str] = None
    retrosynthesis_services_host: Optional[str] = None

    chroma_port: Optional[str] = None
    embedding_port: Optional[str] = None
    reranker_port: Optional[str] = None
    opencChemie_port: Optional[str] = None
    chem_services_port: Optional[str] = None
    retrosynthesis_services_port: Optional[str] = None
  

# =========================
# COLLECTIONS
# =========================
class CollectionsSettings(BaseModel):
    summaries: Optional[str] = None
    texts: Optional[str] = None
    images: Optional[str] = None


# =========================
# S3
# =========================
class S3Settings(BaseModel):
    use_s3: bool = False
    endpoint_url: Optional[str] = None
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    bucket_name: Optional[str] = None


# =========================
# OPIK
# =========================
class OpikSettings(BaseModel):
    api_key: Optional[str] = None
    url_override: Optional[str] = None
    opik_project_name: Optional[str] = None


# =========================
# HITL (Human-in-the-Loop)
# =========================
class HITLSettings(BaseModel):
    enabled: bool = True

# =========================
# MAIN SETTINGS
# =========================
class Settings(BaseSettings):
    """Main application settings."""

    llm: LLMSettings = LLMSettings()
    services: ServicesSettings = ServicesSettings()
    med_llm: MedLLMSettings = MedLLMSettings()
    storage: StorageSettings = StorageSettings()
    hosts_ports: HostsPortsSettings = HostsPortsSettings()
    collections: CollectionsSettings = CollectionsSettings()
    s3: S3Settings = S3Settings()
    opik: OpikSettings = OpikSettings()
    hitl: HITLSettings = HITLSettings()
    tool_rag: ToolRAGSettings = ToolRAGSettings()

    model_config = SettingsConfigDict(
        env_file=".env",          
        env_nested_delimiter="__",     # IMPORTANT for nesting
        extra="ignore"
    )


# Global instance
settings = Settings()


def get_settings() -> Settings:
    return settings