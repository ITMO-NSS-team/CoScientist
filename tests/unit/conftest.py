"""Keep unit tests off state the developer actually uses.

Two stores now persist to disk: the session registry (WEB_STATE_DIR, default
graph_runs/web_state) and the sandbox bindings (SANDBOX_BINDINGS_FILE, default
graph_runs/sandbox_bindings.json). Without isolation a test both pollutes real
state and inherits from earlier runs — nickname-uniqueness tests failed with 409
on a second run, and binding tests wrote fake container ids into the file the
running system reads to decide which sandbox to continue in.
"""
import pytest


@pytest.fixture(autouse=True)
def _isolated_web_state(tmp_path, monkeypatch):
    monkeypatch.setenv("WEB_STATE_DIR", str(tmp_path / "web_state"))
    monkeypatch.setenv("SANDBOX_BINDINGS_FILE",
                       str(tmp_path / "sandbox_bindings.json"))
