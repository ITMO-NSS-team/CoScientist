import asyncio
import re
from unittest.mock import patch

from google.adk.sessions import InMemorySessionService

from CoScientist.main import CoScientistManager


class _Runner:
    def __init__(self, *, agent, app_name, session_service):
        self.agent = agent
        self.app_name = app_name
        self.session_service = session_service


def test_manager_generates_random_local_ids():
    manager = CoScientistManager()
    assert re.fullmatch(r"user_[0-9a-f]{32}", manager.user_id)
    assert re.fullmatch(r"session_[0-9a-f]{32}", manager.session_id)


def test_managers_share_service_but_keep_sessions_separate():
    async def scenario():
        service = InMemorySessionService()
        with patch("CoScientist.main.Runner", _Runner):
            first = CoScientistManager(
                user_id="user_a",
                session_id="session_a",
                session_service=service,
            )
            second = CoScientistManager(
                user_id="user_a",
                session_id="session_b",
                session_service=service,
            )
            await first.initialize()
            await second.initialize()
            await first.initialize()  # idempotent

        first_session = await service.get_session(
            app_name="coscientist_app", user_id="user_a", session_id="session_a"
        )
        second_session = await service.get_session(
            app_name="coscientist_app", user_id="user_a", session_id="session_b"
        )
        assert first.session_service is second.session_service is service
        assert first.runner.session_service is second.runner.session_service is service
        assert first_session.state["active_tasks"] == []
        assert second_session.state["active_tasks"] == []

    asyncio.run(scenario())

