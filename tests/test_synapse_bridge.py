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


# ── Task 2: run registry + snapshot_ref ──────────────────────────────────────

def test_run_registry_roundtrip():
    from CoScientist.checkpoints import synapse
    synapse.clear_runs()
    synapse.register_run("ctx-1", "run-1a2b", "00-abc-def-01")
    assert synapse.run_id_for("ctx-1") == "run-1a2b"
    assert synapse.traceparent_for("ctx-1") == "00-abc-def-01"
    assert synapse.run_id_for("unknown") is None


def test_snapshot_ref_build(monkeypatch):
    from CoScientist.checkpoints import synapse
    monkeypatch.setattr(synapse, "_bundle_base_url", lambda: "http://host:8100")
    assert synapse.snapshot_ref_for("ckpt_X") == "http://host:8100/api/checkpoints/ckpt_X/bundle"


def test_snapshot_ref_none_when_unconfigured(monkeypatch):
    from CoScientist.checkpoints import synapse
    monkeypatch.setattr(synapse, "_bundle_base_url", lambda: None)
    assert synapse.snapshot_ref_for("ckpt_X") is None
