"""Keep unit tests off the real web-state directory.

The session registry now persists to disk (WEB_STATE_DIR, default
graph_runs/web_state). Without isolation a test that creates a user would both
pollute the developer's real state and inherit users from earlier runs — which
made nickname-uniqueness tests fail with 409 on the second run.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_web_state(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web_state"))
