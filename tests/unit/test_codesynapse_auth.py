import pytest
from pydantic import ValidationError

from CoScientist.integrations.codesynapse.auth import IntegrationClaims, claims_from_payload, sha256_json, sha256_text


def test_claims_require_scoped_external_identity():
    claims = claims_from_payload({
        "iss": "codesynapse", "aud": "coscientist", "tenant_id": "tenant-1",
        "project_id": "project-1", "external_run_id": "external-1",
        "research_request_sha256": sha256_text("Research request"),
        "context_sha256": sha256_json({"project": "example"}),
        "trace_callback_url": "http://codesynapse/internal/trace",
        "trace_capability_token_hash": sha256_text("trace-capability"),
        "control_token_hash": sha256_text("control-capability"),
    })

    assert claims.external_run_id == "external-1"

    with pytest.raises(ValidationError):
        IntegrationClaims(
            issuer="codesynapse", audience="coscientist", tenant_id="tenant-1", project_id="project-1", external_run_id=" ",
            research_request_sha256=sha256_text("Research request"), context_sha256=sha256_json({}),
            trace_callback_url="http://codesynapse/internal/trace", trace_capability_token_hash=sha256_text("trace"),
            control_token_hash=sha256_text("control"),
        )
