"""`read_saved_events` — the event log of a saved bundle, on its own.

Reading a whole bundle to get at its event log means unpacking the ADK session,
both graphs and possibly a sandbox trajectory: tens of megabytes for a list the
status indicator's `?demo=` mode replays in seconds. This reads one member.
"""

import json
import zipfile

import pytest

from CoScientist.web import session_bundle


@pytest.fixture
def snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(session_bundle, "_snapshots_dir", lambda: tmp_path)
    return tmp_path


def _bundle(path, members):
    with zipfile.ZipFile(path, "w") as zf:
        for name, payload in members.items():
            zf.writestr(name, json.dumps(payload))


def test_reads_only_the_event_log(snapshots):
    events = [{"type": "user_message", "message": "hi"},
              {"type": "tool_activity", "phase": "call", "tool": "tavily_search"}]
    _bundle(snapshots / "run.cossession.zip", {
        "manifest.json": {"bundle_version": 1},
        "agent_events.json": events,
        "adk_session.json": {"huge": "x" * 10_000},
    })
    assert session_bundle.read_saved_events("run.cossession.zip") == events


def test_a_bundle_without_an_event_log_is_empty_not_an_error(snapshots):
    _bundle(snapshots / "bare.cossession.zip", {"manifest.json": {"bundle_version": 1}})
    assert session_bundle.read_saved_events("bare.cossession.zip") == []


def test_a_corrupt_event_log_is_empty_not_an_error(snapshots):
    with zipfile.ZipFile(snapshots / "broken.cossession.zip", "w") as zf:
        zf.writestr("manifest.json", "{}")
        zf.writestr("agent_events.json", "{not json")
    assert session_bundle.read_saved_events("broken.cossession.zip") == []


def test_a_non_list_event_log_is_rejected(snapshots):
    _bundle(snapshots / "odd.cossession.zip", {
        "manifest.json": {}, "agent_events.json": {"events": []},
    })
    assert session_bundle.read_saved_events("odd.cossession.zip") == []


def test_missing_bundle_raises(snapshots):
    with pytest.raises(FileNotFoundError):
        session_bundle.read_saved_events("nope.cossession.zip")


def test_path_traversal_is_stripped(snapshots):
    _bundle(snapshots / "run.cossession.zip", {
        "manifest.json": {}, "agent_events.json": [{"type": "user_message"}],
    })
    assert session_bundle.read_saved_events("../../run.cossession.zip") == [
        {"type": "user_message"}
    ]
