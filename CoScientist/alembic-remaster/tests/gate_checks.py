"""Standalone regression checks for the remaster's deterministic gates.

No pytest dependency — run inside the base image where `alembic` imports
cleanly:

    docker run --rm -e PYTHONPATH=/app -v "$PWD/CoScientist/alembic-remaster/tests":/t \
        alembic-base:latest --entrypoint python /t/gate_checks.py

or from a clean /app-style copy on the host. Exits non-zero on any failure.
"""
import sys
import tempfile
from pathlib import Path


def check(name, cond):
    print(f"{'ok  ' if cond else 'FAIL'} {name}")
    if not cond:
        check.failed = True


check.failed = False


def main() -> int:
    from alembic.contract import (render_validation_md, Validation, ToolVerdict,
                                  parse_json_block, parse_yaml_block, parse_samples)
    from alembic.tools.invoke import (_parse_result, _bad_sample_reason,
                                      _find_undefined_names, _import_safe_prefix)
    from alembic.tools.analysis import symbol_table, verify_target, decide_layout
    from alembic.tools.fs import _norm_out_rel

    F = "```"  # code fence — kept out of the literals so it reads clearly

    # -1. output-dir path normalization: a redundant leading "output/" (the
    # doubling mistake that hid server.py from the artefact check) is stripped.
    check("norm strips leading output/", _norm_out_rel("output/server.py") == "server.py")
    check("norm strips doubled output/", _norm_out_rel("output/output/tests/t.py") == "tests/t.py")
    check("norm leaves clean path", _norm_out_rel("helpers/run.py") == "helpers/run.py")
    check("norm strips ./ and /", _norm_out_rel("/./server.py") == "server.py")

    # 0. contract-block parsing must survive the common LLM slips: a missing
    # closing fence, trailing commas, and unrelated ```python blocks before it.
    check("json: proper fenced",
          parse_json_block("x\n" + F + 'json\n{"tools":[]}\n' + F) == {"tools": []})
    check("json: MISSING closing fence recovered",
          (parse_json_block(F + 'json\n{"tools":[{"name":"a","target":"m:f"}]}\n')
           or {}).get("tools") == [{"name": "a", "target": "m:f"}])
    check("json: trailing commas tolerated",
          parse_json_block(F + 'json\n{"tools":[1,2,],}\n') == {"tools": [1, 2]})
    check("yaml: missing closing fence recovered",
          isinstance(parse_yaml_block(F + "yaml\nsamples:\n  f:\n    a: 1\n"), dict))

    # 1. validation.md renders in the exact section shape the harness parses.
    v = Validation(syntax_ok=True, tests_ran=True, tests_passed=3, tests_failed=1,
                   tools=[ToolVerdict("a", "PASSED"), ToolVerdict("b", "SKIPPED", "no data"),
                          ToolVerdict("c", "FAILED", "boom")])
    md = render_validation_md("demo", v)
    check("validation.md has all 4 harness sections",
          all(h in md for h in ("## Syntax & Imports", "## Tests",
                                "## Tool Invocations", "## Overall")))
    check("tool line format matches harness regex",
          "- **a** — PASSED" in md and "- **b** — SKIPPED (no data)" in md)
    check("tests line carries counts", "3 passed, 1 failed" in md)

    # 2. sentinel result parsing survives banners before the marker.
    check("sentinel parse",
          _parse_result("noise\n<<<ALEMBIC_RESULT>>>\n{\"ok\": true, \"result\": 1}")
          == {"ok": True, "result": 1})
    check("last-json fallback",
          _parse_result('warn\n{"ok": true, "result": 2}') == {"ok": True, "result": 2})

    # 3. bad-sample gate: flags nonexistent path, ignores HF id / device.
    rd, od = Path("/no_repo"), Path("/no_out")
    check("bad path flagged", bool(_bad_sample_reason({"f": "d/x.csv"}, rd, od)))
    check("HF id ignored", not _bad_sample_reason({"m": "org/model"}, rd, od))
    check("device ignored", not _bad_sample_reason({"device": "cuda:0"}, rd, od))

    # 4. static checks.
    check("undefined name caught",
          _find_undefined_names("import os\nx=os.x\ny=torch.z\n") == ["torch"])
    check("clean file ok", _find_undefined_names("import torch\nx=torch.z\n") is None)

    # 5. AST symbol verification + method indexing on a synthetic repo.
    with tempfile.TemporaryDirectory() as d:
        repo = Path(d)
        (repo / "pkg").mkdir()
        (repo / "pkg" / "__init__.py").write_text("")
        (repo / "pkg" / "mod.py").write_text(
            "def real_fn(a, b):\n    return a\n"
            "class Thing:\n    def do_it(self, x):\n        return x\n")
        (repo / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.10"\n')
        t = symbol_table(repo)
        check("real function verified", verify_target("pkg.mod:real_fn", t, repo)["ok"])
        check("real function params extracted",
              verify_target("pkg.mod:real_fn", t, repo)["params"] == ["a", "b"])
        check("real method verified", verify_target("pkg.mod:Thing.do_it", t, repo)["ok"])
        check("hallucinated symbol dropped",
              not verify_target("pkg.mod:no_such", t, repo)["ok"])
        # script targets: explicit "script:" and a bare ".py" path the Explorer
        # forgot to prefix both resolve when the file exists.
        (repo / "run_it.py").write_text("print('hi')\n")
        check("script: prefix verified", verify_target("script:run_it.py", t, repo)["ok"])
        check("bare .py path verified", verify_target("run_it.py", t, repo)["ok"])
        check("missing script dropped", not verify_target("nope.py", t, repo)["ok"])
        check("one-venv layout from >=3.10", decide_layout(repo)["layout"] == "one-venv")
        (repo / "pyproject.toml").write_text('[project]\nrequires-python = ">=3.7,<3.9"\n')
        lay = decide_layout(repo)
        check("two-venv layout from <3.10",
              lay["layout"] == "two-venv" and lay["repo_python"] in ("3.8", "3.7"))

    print("\n" + ("SOME CHECKS FAILED" if check.failed else "ALL GATE CHECKS PASSED"))
    return 1 if check.failed else 0


if __name__ == "__main__":
    sys.exit(main())
