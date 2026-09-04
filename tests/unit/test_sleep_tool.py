"""``sleep_tool`` lets an agent explicitly pause between status/log checks on a
long-running job, instead of re-checking every turn and burning LLM tokens.

The repo has no pytest-asyncio, so each async call is driven with asyncio.run.
"""

import asyncio

import CoScientist.tools.sleep_tool as sleep_tool_module
from CoScientist.tools.sleep_tool import MAX_SLEEP_MINUTES, sleep_tool


def _run_with_fake_sleep(monkeypatch, minutes):
    captured = []

    async def fake_sleep(seconds):
        captured.append(seconds)

    monkeypatch.setattr(sleep_tool_module.asyncio, "sleep", fake_sleep)
    result = asyncio.run(sleep_tool(minutes))
    return result, captured[0]


def test_a_normal_request_sleeps_for_exactly_that_long(monkeypatch):
    result, slept_seconds = _run_with_fake_sleep(monkeypatch, 3)

    assert result == {"slept_minutes": 3.0}
    assert slept_seconds == 180.0


def test_a_request_over_the_cap_is_silently_clamped(monkeypatch):
    result, slept_seconds = _run_with_fake_sleep(monkeypatch, 999)

    assert result == {"slept_minutes": MAX_SLEEP_MINUTES}
    assert slept_seconds == MAX_SLEEP_MINUTES * 60


def test_a_negative_request_never_sleeps_negative_time(monkeypatch):
    result, slept_seconds = _run_with_fake_sleep(monkeypatch, -5)

    assert result == {"slept_minutes": 0.0}
    assert slept_seconds == 0.0


def test_exactly_the_cap_is_left_alone(monkeypatch):
    result, slept_seconds = _run_with_fake_sleep(monkeypatch, MAX_SLEEP_MINUTES)

    assert result == {"slept_minutes": MAX_SLEEP_MINUTES}
    assert slept_seconds == MAX_SLEEP_MINUTES * 60
