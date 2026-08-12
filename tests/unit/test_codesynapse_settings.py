from CoScientist.integrations.codesynapse.settings import CodesynapseIntegrationSettings


def test_codesynapse_settings_have_safe_disabled_defaults():
    settings = CodesynapseIntegrationSettings()

    assert not settings.enabled
    assert settings.inline_artifact_limit_bytes == 512 * 1024
    assert settings.inline_trace_payload_limit_bytes == 128 * 1024


def test_codesynapse_settings_require_mongo_and_jwks_when_enabled():
    settings = CodesynapseIntegrationSettings(enabled=True)

    missing = settings.missing_readiness_requirements()
    assert "mongo_uri" in missing
    assert "jwks_url" in missing
