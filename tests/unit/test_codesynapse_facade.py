import asyncio

from CoScientist.hitl.models import HITLAction, HITLRequest
from CoScientist.integrations.codesynapse.facade import CodesynapseFacade, StartRequest
from CoScientist.integrations.codesynapse.models import RunState
from CoScientist.integrations.codesynapse.store import InMemoryIntegrationStore


class _Executor:
    async def execute(self, request, hitl_handler):
        return "# Scientific report"


def test_facade_start_is_idempotent_and_returns_working_task():
    async def scenario():
        facade = CodesynapseFacade(store=InMemoryIntegrationStore(), executor=_Executor())
        request = StartRequest(
            external_run_id="external-1", tenant_id="tenant-1", project_id="project-1",
            research_request="Find a hypothesis",
        )
        first = await facade.start(request, run_in_background=False)
        second = await facade.start(request, run_in_background=False)

        assert first.a2a_task_id == second.a2a_task_id
        task = await facade.get_task(first.a2a_task_id)
        assert task.state is RunState.COMPLETED
        assert task.artifacts.final_report.text == "# Scientific report"

    asyncio.run(scenario())


def test_facade_idempotency_survives_two_process_local_locks():
    async def scenario():
        store = InMemoryIntegrationStore()
        first_facade = CodesynapseFacade(store=store, executor=_Executor())
        second_facade = CodesynapseFacade(store=store, executor=_Executor())
        request = StartRequest(
            external_run_id="external-1", tenant_id="tenant-1", project_id="project-1",
            research_request="Find a hypothesis",
        )

        first, second = await asyncio.gather(
            first_facade.start(request, run_in_background=True),
            second_facade.start(request, run_in_background=True),
        )

        assert first.a2a_task_id == second.a2a_task_id
        await first_facade.interrupt_non_terminal_tasks()

    asyncio.run(scenario())


def test_facade_marks_non_terminal_tasks_interrupted_on_restart():
    async def scenario():
        gate = asyncio.Event()

        class BlockingExecutor:
            async def execute(self, request, hitl_handler):
                await gate.wait()
                return "unreachable"

        facade = CodesynapseFacade(store=InMemoryIntegrationStore(), executor=BlockingExecutor())
        started = await facade.start(
            StartRequest(
                external_run_id="external-1", tenant_id="tenant-1", project_id="project-1",
                research_request="Find a hypothesis",
            )
        )
        await asyncio.sleep(0)
        interrupted = await facade.interrupt_non_terminal_tasks()

        task = await facade.get_task(started.a2a_task_id)
        assert interrupted == 1
        assert task.state is RunState.INTERRUPTED
        assert task.artifacts.error.data["error_code"] == "interrupted"

    asyncio.run(scenario())


def test_facade_cancel_returns_terminal_cancelled_task():
    async def scenario():
        gate = asyncio.Event()

        class BlockingExecutor:
            async def execute(self, request, hitl_handler):
                await gate.wait()
                return "unreachable"

        facade = CodesynapseFacade(store=InMemoryIntegrationStore(), executor=BlockingExecutor())
        started = await facade.start(
            StartRequest(
                external_run_id="external-1", tenant_id="tenant-1", project_id="project-1",
                research_request="Find a hypothesis",
            )
        )
        await asyncio.sleep(0)
        assert await facade.cancel(started.a2a_task_id)

        task = await facade.get_task(started.a2a_task_id)
        assert task.state is RunState.CANCELLED
        assert task.artifacts.error.data["error_code"] == "cancelled"
        events = await facade._store.replay_events(task.coscientist_run_id)
        assert events[-1].type == "run.cancelled"

    asyncio.run(scenario())


def test_facade_flushes_trace_outbox_after_each_recorded_event():
    async def scenario():
        flushed = []

        class Dispatcher:
            async def flush_run(self, run_id):
                flushed.append(run_id)
                return 1

        facade = CodesynapseFacade(
            store=InMemoryIntegrationStore(),
            executor=_Executor(),
            delivery_factory=lambda request: Dispatcher(),
        )
        await facade.start(
            StartRequest(
                external_run_id="external-1", tenant_id="tenant-1", project_id="project-1",
                research_request="Find a hypothesis",
            ),
            run_in_background=False,
        )

        assert len(flushed) >= 2

    asyncio.run(scenario())


def test_facade_exposes_hitl_timeout_as_a_stable_terminal_error_code():
    async def scenario():
        class TimeoutExecutor:
            async def execute(self, request, hitl_handler):
                await hitl_handler.handle_request(HITLRequest(
                    agent_name="planner", action_type=HITLAction.APPROVE, message="approve", timeout_seconds=0.001,
                ))
                return "unreachable"

        facade = CodesynapseFacade(store=InMemoryIntegrationStore(), executor=TimeoutExecutor())
        task = await facade.start(
            StartRequest(
                external_run_id="external-1", tenant_id="tenant-1", project_id="project-1",
                research_request="Find a hypothesis",
            ),
            run_in_background=False,
        )
        assert task.state is RunState.FAILED
        assert task.artifacts.error.data["error_code"] == "hitl_timeout"

    asyncio.run(scenario())
