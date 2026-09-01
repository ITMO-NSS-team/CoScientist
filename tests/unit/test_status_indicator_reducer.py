"""The status indicator's phrase reducer, driven headlessly through node.

The reducer lives in ``CoScientist/web/static/status_indicator.js`` because it
is pure UI, but what it decides is not cosmetic: it is the only thing a
non-developer reads while a run is in flight. These tests pin the three ways it
was wrong when first written against a recorded session —

* ``research_*`` tools were announced as web search (``research`` *contains*
  "search", and the catch-all search rule matched first);
* a truncated JSON argument dump was printed as the "detail" under the phrase;
* a failed run was relabelled "Готово" by the idle ``status`` frame that
  follows it;
* painting before ``mount()`` threw, and the exception surfaced inside the
  page's own ``bootstrap()`` — which then asked the user to create an account
  by hand instead of picking up ``COSCIENTIST_USERNAME``.

Skipped when node is not installed; nothing else in the suite needs it.
"""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

MODULE = (
    Path(__file__).resolve().parents[2]
    / "CoScientist" / "web" / "static" / "status_indicator.js"
)

# A DOM stub plus a virtual clock. The reducer holds each phrase for 1.2 s and
# throttles painting, so a synchronous replay has to advance time between
# events or every transition would still be pending when the harness exits.
HARNESS = r"""
let clock = 1e12;
Date.now = () => clock;
const store = {};
global.localStorage = { getItem: k => (k in store ? store[k] : null),
                        setItem: (k, v) => { store[k] = String(v); } };
function fakeEl() {
  return {
    innerHTML: '',
    set textContent(v) {
      this.innerHTML = String(v == null ? '' : v)
        .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    },
    classList: { add() {}, remove() {}, contains() { return false; } },
    querySelector() { return { addEventListener() {} }; },
  };
}
const rootEl = fakeEl();
global.document = { createElement: () => fakeEl() };
global.location = { search: '' };
global.window = global;
global.fetch = () => Promise.reject(new Error('offline'));

require(process.argv[2]);
StatusIndicator.mount(rootEl);
StatusIndicator.setLang(process.argv[4] || 'ru');

const lines = [];
for (const event of JSON.parse(process.argv[3])) {
  clock += 2000;
  StatusIndicator.feed(event);
  StatusIndicator.setConnected(true);   // forces a paint
  const line = rootEl.innerHTML.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
  if (line && lines[lines.length - 1] !== line) lines.push(line);
}
console.log(JSON.stringify(lines));
process.exit(0);
"""


# The same stubs, but nothing is ever mounted — the app's `bootstrap()` runs
# while the page is still parsing, so the reducer really is called this early.
HARNESS_UNMOUNTED = r"""
let clock = 1e12;
Date.now = () => clock;
const store = {};
global.localStorage = { getItem: k => (k in store ? store[k] : null),
                        setItem: (k, v) => { store[k] = String(v); } };
function fakeEl() {
  return {
    innerHTML: '',
    set textContent(v) { this.innerHTML = String(v == null ? '' : v); },
    classList: { add() {}, remove() {}, contains() { return false; } },
    querySelector() { return { addEventListener() {} }; },
  };
}
global.document = { createElement: () => fakeEl() };
global.location = { search: '' };
global.window = global;
global.fetch = () => Promise.reject(new Error('offline'));

require(process.argv[2]);

// Everything the page can call before the container exists.
StatusIndicator.reset();
StatusIndicator.setLang('ru');
StatusIndicator.setConnected(false);
StatusIndicator.feed({ type: 'user_message' });
clock += 2000;
StatusIndicator.feed({ type: 'tool_activity', phase: 'call', tool: 'tavily_search',
                       call_id: 'a', author: 'ResearchAgent', args: { query: 'solubility' } });
StatusIndicator.markStopped();

// A container that breaks on contact must not break the caller either.
const hostile = { get classList() { throw new Error('detached node'); } };
StatusIndicator.mount(hostile);
StatusIndicator.feed({ type: 'user_message' });

// Now a real one: the state gathered while unmounted has to still be there.
const root = fakeEl();
StatusIndicator.mount(root);
StatusIndicator.setConnected(true);
clock += 2000;
StatusIndicator.feed({ type: 'tool_activity', phase: 'call', tool: 'run_sandbox_task',
                       call_id: 'b', author: 'CoderAgent', args: { task: 'train' } });
console.log(JSON.stringify(
  root.innerHTML.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim()));
process.exit(0);
"""


@pytest.fixture(scope="module")
def replay(tmp_path_factory):
    """Return ``replay(events, lang='ru') -> [str]``: the distinct rendered lines."""
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    harness = tmp_path_factory.mktemp("si") / "harness.js"
    harness.write_text(HARNESS, encoding="utf-8")

    def run(events, lang="ru"):
        out = subprocess.run(
            ["node", str(harness), str(MODULE), json.dumps(events), lang],
            capture_output=True, text=True, timeout=60, check=True,
        )
        return json.loads(out.stdout.strip().splitlines()[-1])

    return run


def _call(tool, call_id, author="OrchestratorAgent", args=None):
    return {"type": "tool_activity", "phase": "call", "tool": tool,
            "call_id": call_id, "author": author, "args": args if args is not None else {}}


def _result(tool, call_id, author="OrchestratorAgent", result=None):
    return {"type": "tool_activity", "phase": "result", "tool": tool,
            "call_id": call_id, "author": author, "result": result}


def test_research_graph_tools_are_not_web_search(replay):
    lines = replay([
        {"type": "user_message"},
        _call("research_commit", "c1", "ResearchAgent", {"nodes": []}),
        _result("research_commit", "c1", "ResearchAgent"),
        _call("read_research_graph", "c2", "ResearchAgent"),
    ])
    joined = " | ".join(lines)
    assert "граф знаний" in joined
    assert "карт" in joined            # «Сверяюсь с картой исследования»
    assert "в интернете" not in joined


def test_truncated_json_arguments_never_reach_the_phrase(replay):
    # `tool_activity` degrades an oversized argument dict to a truncated string;
    # printing it would caption the run with `{"tasks": [{"id": "TASK-1"…`.
    lines = replay([
        {"type": "user_message"},
        _call("create_plan", "p1", "PlannerAgent",
              '{"tasks": [{"id": "TASK-1", "title": "Research"}]} …'),
    ])
    joined = " | ".join(lines)
    assert "Планирую шаги" in joined
    assert "{" not in joined and "TASK-1" not in joined


def test_plan_progress_is_read_off_the_task_tracker(replay):
    lines = replay([
        {"type": "user_message"},
        _call("create_plan", "p1", "PlannerAgent", "{...}"),
        _result("create_plan", "p1", "PlannerAgent", {"plan": [1, 2, 3, 4, 5]}),
    ])
    assert any("Шаг 1 из 5" in line for line in lines)


def test_a_real_query_is_shown_as_the_detail(replay):
    lines = replay([
        {"type": "user_message"},
        _call("tavily_search", "s1", "ResearchAgent", {"query": "ibuprofen solubility"}),
    ])
    joined = " | ".join(lines)
    assert "Ищу информацию в интернете" in joined
    assert "ibuprofen solubility" in joined


def test_knowledge_memory_is_the_knowledge_base_not_the_graph(replay):
    lines = replay([
        {"type": "user_message"},
        _call("search_knowledge_memory", "k1", "ResearchAgent", {"query": "CVAE"}),
    ])
    assert any("в базе знаний" in line for line in lines)


def test_delegation_names_the_agent_a_user_would_recognise(replay):
    lines = replay([
        {"type": "user_message"},
        _call("CoderAgent", "d1", "TaskExecutorAgent", {"request": "build the dataset"}),
    ])
    assert any("Передаю задачу: Инженер" in line for line in lines)


def test_one_failing_tool_does_not_fail_the_run(replay):
    lines = replay([
        {"type": "user_message"},
        _call("run_sandbox_task", "x1", "CoderAgent", {"task": "install rdkit"}),
        {"type": "tool_activity", "phase": "error", "tool": "run_sandbox_task",
         "call_id": "x1", "author": "CoderAgent", "error": "boom"},
    ])
    joined = " | ".join(lines)
    assert "пробую иначе" in joined
    assert "Что-то пошло не так" not in joined


def test_a_failed_run_is_never_relabelled_done(replay):
    # The runtime broadcasts an idle `status` after the error; it must not turn
    # the red indicator green.
    lines = replay([
        {"type": "user_message"},
        {"type": "agent_event", "author": "CoderAgent"},
        {"type": "error", "message": "Error processing query: TaskGroup"},
        {"type": "status", "status": "idle"},
        {"type": "final_response"},
    ])
    assert "Что-то пошло не так" in lines[-1]
    joined = " | ".join(lines)
    assert "Готово" not in joined
    # The raw traceback summary belongs in the chat, not in the status line.
    assert "TaskGroup" not in joined


def test_the_user_is_not_shown_as_an_agent(replay):
    lines = replay([
        {"type": "user_message"},
        {"type": "agent_event", "author": "user"},
        {"type": "agent_event", "author": "OrchestratorAgent"},
    ])
    joined = " | ".join(lines)
    assert "Координатор" in joined
    assert "user" not in joined


def test_an_unknown_tool_is_classified_by_its_description(replay):
    # An MCP server built at runtime names its tools whatever its source
    # repository did. `tool_activity` carries each tool's own description for
    # exactly this case.
    lines = replay([
        {"type": "user_message"},
        {"type": "tool_activity", "phase": "call", "tool": "qsr_lookup_v2",
         "call_id": "u1", "author": "TaskExecutorAgent", "args": {"q": "aspirin"},
         "description": "Search PubMed for clinical trials matching a query."},
    ])
    assert any("научные статьи" in line for line in lines)


def test_a_tool_with_no_description_falls_back_to_its_name(replay):
    lines = replay([
        {"type": "user_message"},
        {"type": "tool_activity", "phase": "call", "tool": "zx_frobnicate",
         "call_id": "u1", "author": "TaskExecutorAgent", "args": {}},
    ])
    joined = " | ".join(lines)
    assert "Работаю с инструментом" in joined
    # Underscores are a machine's punctuation, not a reader's.
    assert "zx frobnicate" in joined and "zx_frobnicate" not in joined


def test_the_in_progress_task_is_named_not_just_counted(replay):
    lines = replay([
        {"type": "user_message"},
        _call("create_plan", "p1", "PlannerAgent", "{...}"),
        _result("create_plan", "p1", "PlannerAgent", {"plan": [
            {"id": "TASK-1", "title": "Собрать датасет"},
            {"id": "TASK-2", "title": "Обучить модель"},
        ]}),
        _call("update_task_status", "u1", "OrchestratorAgent",
              {"task_id": "TASK-1", "status": "DONE"}),
        _result("update_task_status", "u1", "OrchestratorAgent",
                {"task": {"id": "TASK-1", "status": "DONE"}}),
        _call("update_task_status", "u2", "OrchestratorAgent",
              {"task_id": "TASK-2", "status": "IN_PROGRESS"}),
        _result("update_task_status", "u2", "OrchestratorAgent",
                {"task": {"id": "TASK-2", "status": "IN_PROGRESS"}}),
    ])
    assert any("Шаг 2 из 2: Обучить модель" in line for line in lines)


def test_the_sandbox_agents_own_plan_reaches_the_status_line(replay):
    # A sandbox run is one tool call that can last hours and reports nothing
    # until it ends; its agent's plan is the only account of what is happening.
    lines = replay([
        {"type": "user_message"},
        _call("run_sandbox_task", "s1", "CoderAgent", {"task": "train a CVAE"}),
        {"type": "sandbox_plan", "agent": "CoderAgent", "plan": {
            "revision": 3,
            "current": "Обучить модель",
            "progress": {"total": 3, "done": 1},
            "items": [
                {"title": "Скачать датасет", "status": "done"},
                {"title": "Обучить модель", "status": "in_progress"},
                {"title": "Выгрузить результаты", "status": "todo"},
            ],
        }},
    ])
    assert any("Шаг 2 из 3: Обучить модель" in line for line in lines)


def test_the_sandbox_plan_outranks_the_orchestrators(replay):
    """The nested plan describes the step actually running right now."""
    lines = replay([
        {"type": "user_message"},
        _call("create_plan", "p1", "PlannerAgent", "{...}"),
        _result("create_plan", "p1", "PlannerAgent", {"plan": [
            {"id": "TASK-1", "title": "Исследование"},
            {"id": "TASK-2", "title": "Эксперимент"},
        ]}),
        _call("run_sandbox_task", "s1", "CoderAgent", {"task": "train"}),
        {"type": "sandbox_plan", "plan": {
            "revision": 1, "current": "Обучить модель",
            "progress": {"total": 3, "done": 1}, "items": [],
        }},
    ])
    assert "Шаг 2 из 3: Обучить модель" in lines[-1]
    assert "Исследование" not in lines[-1]


def test_the_sandbox_plan_is_dropped_when_its_task_ends(replay):
    lines = replay([
        {"type": "user_message"},
        _call("create_plan", "p1", "PlannerAgent", "{...}"),
        _result("create_plan", "p1", "PlannerAgent", {"plan": [
            {"id": "TASK-1", "title": "Исследование"},
        ]}),
        _call("run_sandbox_task", "s1", "CoderAgent", {"task": "train"}),
        {"type": "sandbox_plan", "plan": {
            "revision": 1, "current": "Обучить модель",
            "progress": {"total": 3, "done": 1}, "items": [],
        }},
        _result("run_sandbox_task", "s1", "CoderAgent", {"status": "success"}),
    ])
    # The container is done with that task; its plan describes nothing now, and
    # the orchestrator's own plan takes the line back.
    assert "Обучить модель" not in lines[-1]
    assert "Шаг 1 из 1: Исследование" in lines[-1]


def test_a_plan_with_no_progress_block_still_reads(replay):
    """`progress` is the service's convenience; `items` is the source of truth."""
    lines = replay([
        {"type": "user_message"},
        _call("run_sandbox_task", "s1", "CoderAgent", {"task": "train"}),
        {"type": "sandbox_plan", "plan": {"revision": 1, "items": [
            {"title": "Скачать датасет", "status": "done"},
            {"title": "Обучить модель", "status": "in_progress"},
        ]}},
    ])
    assert any("Шаг 2 из 2: Обучить модель" in line for line in lines)


def test_english_is_a_complete_translation(replay):
    lines = replay([
        {"type": "user_message"},
        _call("tavily_search", "s1", "ResearchAgent", {"query": "solubility"}),
        _call("CoderAgent", "d1", "TaskExecutorAgent", {"request": "run it"}),
    ], lang="en")
    joined = " | ".join(lines)
    assert "Searching the web" in joined
    assert "Handing over to: Engineer" in joined
    # No Russian may leak through the English dictionary.
    assert not any("Ѐ" <= ch <= "ӿ" for ch in joined)


def test_the_indicator_survives_being_used_before_it_is_mounted(tmp_path):
    """The page's `bootstrap()` runs before the container exists.

    A throw here does not just lose a status line: it lands in bootstrap's own
    `catch`, which then treats the session as unrestorable and opens the
    identity modal — so `COSCIENTIST_USERNAME` silently stops working.
    """
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    harness = tmp_path / "unmounted.js"
    harness.write_text(HARNESS_UNMOUNTED, encoding="utf-8")

    out = subprocess.run(
        ["node", str(harness), str(MODULE)],
        capture_output=True, text=True, timeout=60,
    )
    assert out.returncode == 0, out.stderr
    line = json.loads(out.stdout.strip().splitlines()[-1])
    # Mounted at last, and showing the state it accumulated while it could not
    # draw — not a blank line.
    assert "Пишу и запускаю код" in line
