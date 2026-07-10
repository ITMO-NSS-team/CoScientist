"""Standalone regression checks for the upgraded deterministic gates (R1-R8).

No pytest dependency in the harness itself — run inside the base image where
`alembic` imports cleanly:

    docker run --rm -e PYTHONPATH=/app -v "$PWD/CoScientist/alembic-remaster/tests":/t \
        --entrypoint python alembic-base:latest /t/gate_checks.py

Exits non-zero on any failure.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp()
os.environ["ALEMBIC_WORKDIR"] = _TMP          # must precede alembic imports
os.chdir(_TMP)


def check(name, cond):
    print(f"{'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        check.failed = True


check.failed = False


def main() -> int:
    from alembic.contract import (
        Plan, EnvSpec, ToolSpec, ToolReport, Validation,
        load_plan, parse_json_block, render_validation_md, save_plan,
        update_stage_status,
    )
    from alembic.tools.paths import set_current_repo, reports_dir, output_dir
    from alembic.tools.fs import _norm_out_rel
    from alembic.tools.invoke import (
        _TEST_LINE, _missing_input_files, _parse_result,
        _invoke_tool_function_sync, check_tool_artefacts, _run_tool_tests_sync,
        find_undefined_names,
    )
    from alembic.tools.analysis import (
        decide_layout, symbol_table, target_top_modules, verify_target,
    )
    from alembic.tools.codegen import (
        render_code_py, render_server, render_setup_sh, tool_signature,
    )
    from alembic.tools.fs import _looks_like_sha

    set_current_repo("https://github.com/x/demo")
    F = "```"

    # ── path normalization ────────────────────────────────────────────────────
    check("norm strips leading output/", _norm_out_rel("output/server.py") == "server.py")
    check("norm strips doubled output/", _norm_out_rel("output/output/tests/t.py") == "tests/t.py")
    check("norm leaves clean path", _norm_out_rel("tools/run.py") == "tools/run.py")

    # ── plan-block parsing (missing fence / trailing commas) ─────────────────
    check("json: proper fenced",
          parse_json_block("x\n" + F + 'json\n{"tools":[]}\n' + F) == {"tools": []})
    check("json: missing closing fence recovered",
          (parse_json_block(F + 'json\n{"tools":[{"name":"a","target":"m:f"}]}\n')
           or {}).get("tools") == [{"name": "a", "target": "m:f"}])
    check("json: trailing commas tolerated",
          parse_json_block(F + 'json\n{"tools":[1,2,],}\n') == {"tools": [1, 2]})

    # ── ToolReport semantics (R6 / Q&A) ───────────────────────────────────────
    t = ToolReport("a", tests_passed=3, tests_total=3, exec_ok=True)
    check("passed: tests green + no crash", t.passed and not t.perfect)
    t = ToolReport("a", tests_passed=3, tests_total=3, exec_ok=True,
                   invoc_passed=2, invoc_total=2)
    check("perfect: + all invocation tests", t.perfect and t.status == "perfect")
    t = ToolReport("a", tests_passed=2, tests_total=3, exec_ok=True)
    check("failed: a red test blocks passed", not t.passed)
    t = ToolReport("a", tests_passed=3, tests_total=3, exec_ok=False)
    check("failed: a crash blocks passed", not t.passed)
    t = ToolReport("a", exec_ok=True, exec_note="still running after 120s")
    check("timeout stays a runtime success", t.passed)   # R6
    t = ToolReport("a")
    check("nothing measured => untested", t.status == "untested")
    v = Validation(tools=[ToolReport("a", tests_passed=1, tests_total=1, exec_ok=True,
                                     invoc_passed=1, invoc_total=1),
                          ToolReport("b", exec_ok=False)])
    c = v.counts()
    check("counts roll up", c["tools_passed"] == 1 and c["tools_perfect"] == 1
          and c["tools_total"] == 2 and c["exec_ok"] == 1 and c["exec_attempted"] == 2)
    check("validation.md renders", "1/2 passed" in render_validation_md("demo", v))

    # ── plan round-trip with new fields ───────────────────────────────────────
    plan = Plan(repo_url="https://github.com/x/demo", env=EnvSpec(),
                tools=[ToolSpec(name="f", target="m:f", sample_args={"x": 1},
                                evidence="README says 0.97", verified=True)],
                tasks=[{"name": "task_f", "arguments": {}}])
    save_plan(plan)
    p2 = load_plan("https://github.com/x/demo")
    check("plan round-trips sample_args/evidence/tasks",
          p2.tools[0].sample_args == {"x": 1} and p2.tools[0].evidence
          and p2.tasks[0]["name"] == "task_f")

    # ── stage_status incremental writer (R2) ─────────────────────────────────
    update_stage_status("explorer", status="passed", resets=1, gate={"verified": 2})
    update_stage_status("explorer", extra=True)
    ss = json.loads((reports_dir() / "stage_status.json").read_text())
    check("stage_status merges fields",
          ss["explorer"]["status"] == "passed" and ss["explorer"]["resets"] == 1
          and ss["explorer"]["extra"] is True)

    # ── sentinel parse + missing-input detection ─────────────────────────────
    check("sentinel parse",
          _parse_result("noise\n<<<ALEMBIC_RESULT>>>\n{\"ok\": true, \"result\": 1}")
          == {"ok": True, "result": 1})
    rd, od = Path("/no_repo"), Path("/no_out")
    check("missing input flagged", bool(_missing_input_files({"f": "d/x.csv"}, rd, od)))
    check("HF id ignored", not _missing_input_files({"m": "org/model"}, rd, od))
    check("device ignored", not _missing_input_files({"device": "cuda:0"}, rd, od))
    # TM-Bench: a /mount/input dir that isn't present ⇒ missing data (runtime
    # success), even without a file extension; /mount/output paths are fine.
    check("missing mount input flagged",
          bool(_missing_input_files({"slide_dir": "/mount/input/SLIDES"}, rd, od)))
    check("mount output not flagged",
          not _missing_input_files({"out": "/mount/output/features"}, rd, od))

    # ── static checks ─────────────────────────────────────────────────────────
    check("undefined name caught",
          find_undefined_names("import os\nx=os.x\ny=torch.z\n") == ["torch"])
    check("clean file ok", find_undefined_names("import torch\nx=torch.z\n") is None)

    # ── pytest -v line parsing (smoke/invoc split) ────────────────────────────
    out = ("tests/test_f.py::test_smoke_import PASSED [ 25%]\n"
           "tests/test_f.py::test_smoke_bad_input FAILED [ 50%]\n"
           "tests/test_f.py::test_invoc_reference[case0] PASSED [ 75%]\n"
           "tests/test_f.py::test_invoc_shape ERROR [100%]\n")
    hits = [(m.group(1), m.group(2)) for m in _TEST_LINE.finditer(out)]
    check("pytest -v lines parsed (incl. parametrized)", len(hits) == 4
          and hits[2] == ("test_invoc_reference", "PASSED"))

    # ── AST symbol verification / layout / top modules ────────────────────────
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        (repo / "pkg").mkdir()
        (repo / "pkg" / "__init__.py").write_text("")
        (repo / "pkg" / "mod.py").write_text(
            "def real_fn(a, b):\n    return a\n"
            "class Thing:\n    def do_it(self, x):\n        return x\n")
        (repo / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.10"\n')
        tbl = symbol_table(repo)
        check("real function verified", verify_target("pkg.mod:real_fn", tbl, repo)["ok"])
        check("params extracted", verify_target("pkg.mod:real_fn", tbl, repo)["params"] == ["a", "b"])
        check("method verified", verify_target("pkg.mod:Thing.do_it", tbl, repo)["ok"])
        check("hallucination dropped", not verify_target("pkg.mod:no_such", tbl, repo)["ok"])
        (repo / "run_it.py").write_text("print('hi')\n")
        check("bare .py path verified", verify_target("run_it.py", tbl, repo)["ok"])
        check("one-venv from >=3.10", decide_layout(repo)["layout"] == "one-venv")
        (repo / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.7,<3.9"\n')
        check("two-venv from <3.10", decide_layout(repo)["layout"] == "two-venv")
    check("target top modules",
          target_top_modules(["pkg.mod:f", "other:g", "script:run.py", "bare"])
          == ["other", "pkg"])

    # ── functions-first artefacts: G3 + runner + pytest end-to-end (R3) ──────
    out_dir = output_dir()
    (out_dir / "tools").mkdir(parents=True, exist_ok=True)
    (out_dir / "tests").mkdir(parents=True, exist_ok=True)
    (out_dir / "tools" / "echo.py").write_text(
        'def echo(x: int, tag: str = "t") -> dict:\n'
        '    """Echo x back.\n\n    Args:\n        x: value.\n        tag: label.\n\n'
        '    Returns:\n        {"x": int, "tag": str}\n\n    Example:\n        echo(1)\n    """\n'
        '    import json\n    return {"x": x, "tag": tag}\n')
    (out_dir / "tests" / "test_echo.py").write_text(
        "from tools.echo import echo\n\n"
        "def test_smoke_returns_dict():\n    assert echo(1)[\"x\"] == 1\n\n"
        "def test_invoc_reference():\n    assert echo(2, tag=\"a\") == {\"x\": 2, \"tag\": \"a\"}\n")
    g3 = check_tool_artefacts(["echo"])
    check("G3 passes on a good tool", g3["passed"])
    g3b = check_tool_artefacts(["ghost"])
    check("G3 fails on a missing tool", not g3b["passed"] and "ghost" in g3b["errors"])
    r = _invoke_tool_function_sync("echo", {"x": 5})
    check("runner round-trip", r.get("ok") and r["result"] == {"x": 5, "tag": "t"})
    tr = _run_tool_tests_sync("echo")
    check("per-tool pytest split", tr["smoke_passed"] == 1 and tr["smoke_total"] == 1
          and tr["invoc_passed"] == 1 and tr["invoc_total"] == 1 and not tr["failures"])

    # ── codegen: signature extraction + server render + code.py (Q&A) ────────
    sig = tool_signature("echo", out_dir)
    check("signature extracted", sig and sig["params"][0] == ("x", "int", None)
          and sig["params"][1] == ("tag", "str", "'t'"))
    src = render_server("demo", [sig])
    compile(src, "server.py", "exec")
    check("server renders + compiles", "@mcp.tool()" in src
          and 'def echo(x: int, tag: str = \'t\') -> dict:' in src
          and '_call("echo", {"x": x, "tag": tag})' in src)
    code = render_code_py("echo", out_dir)
    check("code.py is the verbatim function", code.startswith("def echo(")
          and "import json" in code)

    # ── function_param_names: filter sample args against the WRITTEN signature,
    # not the repo target's params (task tools rename them) ───────────────────
    from alembic.tools.codegen import function_param_names
    (out_dir / "tools" / "task_tool.py").write_text(
        "def task_tool(*, slide_dir: str, output_dir: str) -> dict:\n"
        "    return {'n': 1}\n")
    names, has_kw = function_param_names("task_tool", out_dir)
    check("param names read from written fn (kwonly)",
          names == {"slide_dir", "output_dir"} and not has_kw)
    # the task example args (slide_dir/output_dir) must all survive filtering,
    # even though a repo symbol might call them wsi_dir/output_dir.
    sample = {"slide_dir": "/mount/input/S", "output_dir": "/mount/output/F"}
    kept = {k: v for k, v in sample.items() if k in (names or set())}
    check("task-renamed args survive the signature filter", kept == sample)
    (out_dir / "tools" / "kw.py").write_text(
        "def kw(a, **kwargs) -> dict:\n    return {}\n")
    _, kw_has = function_param_names("kw", out_dir)
    check("**kwargs detected (skip filtering)", kw_has)
    sh = render_setup_sh(["uv venv .venv", "uv pip install numpy"])
    check("setup.sh renders", sh.startswith("#!") and "uv pip install numpy" in sh)

    # ── clone ref detection (TM-Bench commit vs branch pinning) ───────────────
    check("sha ref detected", _looks_like_sha("1fdf48c"))
    check("branch ref not a sha", not _looks_like_sha("v1"))

    # ── src-layout repo-import smoke path (functions-first, src/ repos) ───────
    # current repo is https://github.com/x/demo → repos/ and output/ under _TMP.
    from alembic.tools.invoke import check_repo_imports
    from alembic.tools.paths import repo_path
    src_mod = repo_path() / "src" / "mymod"
    src_mod.mkdir(parents=True, exist_ok=True)
    (src_mod / "__init__.py").write_text("VALUE = 1\n")
    venv_bin = output_dir() / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    py = venv_bin / "python"
    if not py.exists():
        py.symlink_to(sys.executable)
    ri = check_repo_imports(["mymod"])
    check("src-layout module imports via repo/src on path", ri["passed"])
    ri2 = check_repo_imports(["definitely_absent_mod"])
    check("truly-missing module still fails smoke", not ri2["passed"])

    print("\n" + ("SOME CHECKS FAILED" if check.failed else "ALL GATE CHECKS PASSED"))
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
