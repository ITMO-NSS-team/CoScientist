"""The execution graph regrouped as a chronological trace.

The call graph shows what is connected to what; these pin down what the trace
adds — one entry per user prompt, its calls in the order they ran, and the
input, output and duration of each.
"""
from CoScientist.graph.projection import turns


def _full():
    return {"nodes": [
        {"id": "goal:a", "kind": "goal", "turn_id": "a", "label": "first ask",
         "status": "success", "t_start": 100.0, "t_end": 140.0},
        {"id": "tool:2", "kind": "tool_call", "turn_id": "a", "label": "execute_bash",
         "executor_agent": "CoderAgent", "status": "failed", "input": "ls",
         "output": "boom", "t_start": 110.0, "t_end": 110.4},
        {"id": "tool:1", "kind": "tool_call", "turn_id": "a", "label": "tavily_search",
         "executor_agent": "ResearchAgent", "status": "success", "input": "{}",
         "output": "12 papers", "t_start": 101.0, "t_end": 103.5},
        {"id": "result:a", "kind": "result", "turn_id": "a", "output": "done",
         "t_start": 140.0, "t_end": 140.0},
        {"id": "goal:b", "kind": "goal", "turn_id": "b", "label": "second ask",
         "status": "running", "t_start": 200.0},
        {"id": "agent:CoderAgent", "kind": "agent", "status": "success"},
    ]}


def test_turns_are_separated_and_ordered_by_time():
    result = turns(_full())
    assert [t["turn_id"] for t in result["turns"]] == ["a", "b"]
    assert result["turns"][0]["prompt"] == "first ask"
    # Calls come back in the order they ran, not the order they were recorded.
    assert [c["tool"] for c in result["turns"][0]["calls"]] == [
        "tavily_search", "execute_bash"]


def test_a_call_carries_its_agent_arguments_result_and_duration():
    call = turns(_full())["turns"][0]["calls"][0]
    assert call["agent"] == "ResearchAgent"
    assert call["input"] == "{}" and call["output"] == "12 papers"
    assert call["status"] == "success" and call["duration"] == 2.5


def test_agent_nodes_are_not_turns():
    # One agent node serves the whole session, so it belongs to no single turn
    # and must not become one.
    assert all(t["turn_id"] != "agent:CoderAgent" for t in turns(_full())["turns"])


def test_snapshots_written_before_turn_id_still_group():
    legacy = {"nodes": [
        {"id": "goal:old", "kind": "goal", "label": "before the field existed",
         "status": "success", "t_start": 10.0, "t_end": 20.0},
        {"id": "result:old", "kind": "result", "output": "answer", "t_start": 20.0},
    ]}
    result = turns(legacy)
    assert result["count"] == 1
    assert result["turns"][0]["turn_id"] == "old"
    assert result["turns"][0]["answer"] == "answer"


def test_a_call_with_no_turn_is_kept_not_dropped():
    orphan = {"nodes": [{"id": "tool:x", "kind": "tool_call", "label": "t",
                         "status": "success", "t_start": 1.0}]}
    result = turns(orphan)
    assert result["count"] == 1 and result["turns"][0]["turn_id"] == "untagged"
