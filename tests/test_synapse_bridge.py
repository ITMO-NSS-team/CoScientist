"""Unit tests for the Synapse v1 adapter bridge (CoScientist/checkpoints/synapse.py)."""
import asyncio
import tempfile
from types import SimpleNamespace


# ── Task 1: SynapseSettings ──────────────────────────────────────────────────

def test_synapse_settings_default_off(monkeypatch):
    monkeypatch.delenv("SYNAPSE__ENABLED", raising=False)
    from CoScientist.config.settings import Settings
    s = Settings()
    assert s.synapse.enabled is False
    assert s.synapse.callback_url is None
    assert s.synapse.bundle_base_url is None
    assert s.synapse.otlp_endpoint is None


def test_synapse_settings_from_env(monkeypatch):
    monkeypatch.setenv("SYNAPSE__ENABLED", "true")
    monkeypatch.setenv("SYNAPSE__CALLBACK_URL", "http://localhost:9999")
    from CoScientist.config.settings import Settings
    s = Settings()
    assert s.synapse.enabled is True
    assert s.synapse.callback_url == "http://localhost:9999"
