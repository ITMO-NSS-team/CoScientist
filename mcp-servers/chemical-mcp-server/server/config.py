from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PACKAGE_ROOT / ".env"


class Settings(BaseSettings):
    """All environment-driven settings for the chemical MCP server."""

    model_config = SettingsConfigDict(
        env_file=str(DEFAULT_ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Chemical service
    chem_services_host: str = Field(default="localhost")
    chem_services_port: int = Field(default=8005)
    chem_services_timeout: int = Field(default=60)
    # Docking is a much heavier computation than the other endpoints on this
    # client (PDF/figure extraction, SMILES conversion) — 60s is routinely too
    # short for it, especially with several PDB ids docked at once. Separate
    # knob so raising it doesn't also make every lightweight call wait longer
    # to fail when the service is actually down.
    chem_services_docking_timeout: int = Field(default=300)

    # Retrosynthesis
    retrosynthesis_services_host: str = Field(default="localhost")
    retrosynthesis_services_port: int = Field(default=8001)
    retrosynthesis_request_timeout: int = Field(default=60)

    # S3-compatible storage
    s3_endpoint_url: str | None = None
    s3_bucket_name: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None

    chem_mcp_host: str = Field(default="0.0.0.0")
    chem_mcp_port: int = Field(default=7331)
    chem_mcp_path: str = Field(default="/mcp")


def get_settings() -> Settings:
    """Load and validate settings (cached). Call from `main()` for fail-fast startup."""
    return Settings()
