"""JWT verification for primary Codesynapse A2A start requests."""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any

import jwt
from pydantic import BaseModel, Field, model_validator


class IntegrationClaims(BaseModel):
    issuer: str = Field(alias="iss", min_length=1)
    audience: str = Field(alias="aud", min_length=1)
    tenant_id: str = Field(min_length=1)
    project_id: str = Field(min_length=1)
    external_run_id: str = Field(min_length=1)
    research_request_sha256: str = Field(min_length=64, max_length=64)
    context_sha256: str = Field(min_length=64, max_length=64)
    trace_callback_url: str = Field(min_length=1)
    trace_capability_token_hash: str = Field(min_length=64, max_length=64)
    control_token_hash: str = Field(min_length=64, max_length=64)

    @model_validator(mode="after")
    def reject_blank_scope(self) -> "IntegrationClaims":
        for field in ("issuer", "audience", "tenant_id", "project_id", "external_run_id", "trace_callback_url"):
            if not getattr(self, field).strip():
                raise ValueError(f"{field} must not be blank")
        return self


def claims_from_payload(payload: dict[str, Any]) -> IntegrationClaims:
    """Validate decoded JWT payload independently from the cryptographic transport."""

    return IntegrationClaims.model_validate(payload)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: dict[str, Any]) -> str:
    """Hash JSON in the canonical representation used by the signed A2A context."""

    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return sha256_text(payload)


class CodesynapseJWTVerifier:
    """Verify short-lived asymmetric JWTs using the Codesynapse internal JWKS."""

    def __init__(self, *, jwks_url: str, issuer: str, audience: str) -> None:
        self._client = jwt.PyJWKClient(jwks_url)
        self._issuer = issuer
        self._audience = audience

    async def verify(self, encoded_token: str) -> IntegrationClaims:
        signing_key = await asyncio.to_thread(self._client.get_signing_key_from_jwt, encoded_token)
        payload = jwt.decode(
            encoded_token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            issuer=self._issuer,
            audience=self._audience,
            options={"require": ["exp", "iss", "aud"]},
        )
        return claims_from_payload(payload)
