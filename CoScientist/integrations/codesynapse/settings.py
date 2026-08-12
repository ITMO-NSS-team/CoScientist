"""Lightweight façade settings that do not import the full RAG configuration."""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class CodesynapseIntegrationSettings(BaseSettings):
    enabled: bool = False
    mongo_uri: str | None = None
    mongo_database: str = "coscientist"
    jwks_url: str | None = None
    a2a_public_url: str | None = None
    callback_url: str | None = None
    control_token_grace_seconds: int = 300
    hitl_default_timeout_seconds: float = 900.0
    inline_artifact_limit_bytes: int = 512 * 1024
    inline_trace_payload_limit_bytes: int = 128 * 1024

    model_config = SettingsConfigDict(env_prefix="CODESYNAPSE_", extra="ignore")

    def missing_readiness_requirements(self) -> list[str]:
        if not self.enabled:
            return []
        return [name for name in ("mongo_uri", "jwks_url", "a2a_public_url") if not getattr(self, name)]
