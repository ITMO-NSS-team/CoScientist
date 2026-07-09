"""The structured inter-stage contract + deterministic report rendering.

Machine-critical data (tool list, signatures, sample args, SKIP set, expected
returns, venv layout) travels between stages as structured data parsed and
validated by code — not as free-text markdown the next LLM must re-read
correctly. Reports stay human-readable, but every field a *gate* depends on is
extracted here, and ``validation.md`` is rendered by code (never hand-formatted
by an LLM) so the benchmark harness can always parse it.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict

import yaml

from alembic.tools.paths import reports_dir

# ── Fenced-block extraction ──────────────────────────────────────────────────
# Models frequently forget the closing ``` fence, so we do NOT require it: grab
# everything after the opening fence, cut a closing fence only if present, and
# recover the object structurally (balanced braces / forgiving YAML). This makes
# the whole inter-stage contract resilient to the single most common LLM slip.
def _after_fence(text: str, lang: str) -> str | None:
    """Text following the first ```<lang> fence, up to a closing ``` if any."""
    m = re.search(rf"```[ \t]*{lang}[ \t]*\n?(.*)", text or "", re.DOTALL | re.IGNORECASE)
    if not m:
        return None
    body = m.group(1)
    return body.split("```", 1)[0]


def _brace_slice(s: str) -> str | None:
    """The first balanced {...} object in ``s`` (quote/escape aware), or None."""
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = esc = False
    for i in range(start, len(s)):
        c = s[i]
        if in_str:
            esc = (c == "\\") and not esc
            if c == '"' and not esc:
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    return None


def parse_json_block(text: str) -> dict | None:
    """First ```json object in ``text`` as a dict — tolerant of a missing
    closing fence and trailing commas; falls back to any balanced object."""
    text = text or ""
    for chunk in (_after_fence(text, "json"), text):
        if not chunk:
            continue
        blob = _brace_slice(chunk)
        if not blob:
            continue
        for attempt in (blob, re.sub(r",(\s*[}\]])", r"\1", blob)):
            try:
                val = json.loads(attempt)
            except json.JSONDecodeError:
                continue
            if isinstance(val, dict):
                return val
    return None


def parse_yaml_block(text: str) -> dict | None:
    """First ```yaml block in ``text`` as a dict — tolerant of a missing
    closing fence. No whole-document fallback (arbitrary markdown must not be
    coerced into a YAML mapping)."""
    body = _after_fence(text, "ya?ml")
    if not body:
        return None
    try:
        val = yaml.safe_load(body)
    except yaml.YAMLError:
        return None
    return val if isinstance(val, dict) else None


# ── Plan (Explorer proposal → verified by the Plan gate) ─────────────────────
@dataclass
class ToolSpec:
    name: str
    target: str                       # "module.path:symbol" or "script:relpath"
    purpose: str = ""
    params: list[str] = field(default_factory=list)   # real names (Plan gate)
    verified: bool = False            # AST-verified against the clone
    note: str = ""                    # why unverified / demoted


@dataclass
class EnvSpec:
    layout: str = "one-venv"          # "one-venv" | "two-venv"
    server_python: str = "3.11"
    repo_python: str | None = None
    requirements_files: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    system_libs: list[str] = field(default_factory=list)
    weights: list[dict] = field(default_factory=list)


@dataclass
class Plan:
    repo_url: str
    env: EnvSpec
    tools: list[ToolSpec]

    def to_json(self) -> str:
        return json.dumps(
            {"repo_url": self.repo_url, "env": asdict(self.env),
             "tools": [asdict(t) for t in self.tools]},
            indent=2, ensure_ascii=False,
        )


def save_plan(plan: Plan) -> None:
    d = reports_dir(plan.repo_url)
    d.mkdir(parents=True, exist_ok=True)
    (d / "plan.json").write_text(plan.to_json(), encoding="utf-8")


def load_plan(repo_url: str) -> Plan | None:
    p = reports_dir(repo_url) / "plan.json"
    if not p.exists():
        return None
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return Plan(
        repo_url=raw.get("repo_url", repo_url),
        env=EnvSpec(**raw.get("env", {})),
        tools=[ToolSpec(**t) for t in raw.get("tools", [])],
    )


# ── Coder samples block (server.md → drives the validator loop) ──────────────
@dataclass
class SampleSpec:
    name: str
    sample_args: dict | None          # None => SKIP
    holdout_args: dict | None = None
    returns: dict | None = None       # {key: type} expected output shape (F2)
    skip: bool = False
    skip_reason: str = ""


def parse_samples(repo_url: str) -> list[SampleSpec]:
    """Parse server.md's ``samples:`` YAML block into SampleSpecs.

    Each entry is either ``SKIP`` or a dict. A dict with a ``sample_args`` key
    uses the rich shape (sample_args/holdout_args/returns/skip_reason); a plain
    dict of args is accepted as sample_args directly (lenient back-compat).
    Returns [] on any parse failure — the caller treats that as "no guidance."
    """
    server_md = reports_dir(repo_url) / "server.md"
    if not server_md.exists():
        return []
    block = parse_yaml_block(server_md.read_text(encoding="utf-8", errors="replace"))
    samples = (block or {}).get("samples")
    if not isinstance(samples, dict):
        return []

    specs: list[SampleSpec] = []
    for name, v in samples.items():
        if isinstance(v, str) and v.strip().upper() == "SKIP":
            specs.append(SampleSpec(name=name, sample_args=None, skip=True,
                                    skip_reason="marked SKIP by coder"))
        elif isinstance(v, dict) and "sample_args" in v:
            sa = v.get("sample_args")
            is_skip = (isinstance(sa, str) and sa.strip().upper() == "SKIP") or sa is None
            specs.append(SampleSpec(
                name=name,
                sample_args=None if is_skip else sa,
                holdout_args=v.get("holdout_args"),
                returns=v.get("returns"),
                skip=is_skip,
                skip_reason=v.get("skip_reason", "") if is_skip else "",
            ))
        elif isinstance(v, dict):
            specs.append(SampleSpec(name=name, sample_args=v))
        # anything else: ignore silently
    return specs


# ── Validation result (produced by the code-driven validator loop) ───────────
@dataclass
class ToolVerdict:
    name: str
    status: str                       # PASSED | FAILED | SKIPPED
    reason: str = ""


@dataclass
class Validation:
    syntax_ok: bool = False
    syntax_error: str = ""
    tests_ran: bool = False
    tests_passed: int | None = None
    tests_failed: int | None = None
    tests_error: str = ""
    tools: list[ToolVerdict] = field(default_factory=list)
    debugger_actions: list[str] = field(default_factory=list)

    @property
    def overall_ok(self) -> bool:
        if not self.syntax_ok:
            return False
        if any(t.status == "FAILED" for t in self.tools):
            return False
        # at least one tool must have actually been invoked & passed
        return any(t.status == "PASSED" for t in self.tools)


def render_validation_md(repo_name: str, v: Validation) -> str:
    """Render validation.md in the EXACT format run_benchmark.parse_validation
    expects: ## Syntax & Imports / ## Tests / ## Tool Invocations / ## Overall.
    """
    L = [f"# {repo_name} Validation Report", ""]

    L += ["## Syntax & Imports", "PASSED" if v.syntax_ok else "FAILED"]
    if not v.syntax_ok and v.syntax_error:
        L.append(v.syntax_error.strip()[:1000])
    L.append("")

    L.append("## Tests")
    if v.tests_ran and v.tests_passed is not None:
        head = "PASSED" if (v.tests_failed or 0) == 0 else "FAILED"
        L.append(f"{head} — {v.tests_passed} passed, {v.tests_failed or 0} failed")
    else:
        L.append("FAILED — 0 passed, 0 failed")
        if v.tests_error:
            L.append(v.tests_error.strip()[:1000])
    L.append("")

    L.append("## Tool Invocations")
    if v.tools:
        for t in v.tools:
            reason = f" ({t.reason})" if t.reason else ""
            L.append(f"- **{t.name}** — {t.status}{reason}")
    else:
        L.append("- (no tools were declared)")
    L.append("")

    L.append("## Debugger Actions")
    if v.debugger_actions:
        L += [f"- {a}" for a in v.debugger_actions]
    else:
        L.append("None required.")
    L.append("")

    L.append("## Overall")
    if v.overall_ok:
        L.append("PASSED (all invoked tools succeeded; SKIPPED tools do not count as failure)")
    else:
        failed = [t.name for t in v.tools if t.status == "FAILED"]
        bits = []
        if not v.syntax_ok:
            bits.append("syntax/imports failed")
        if failed:
            bits.append("failing tools: " + ", ".join(failed))
        if not any(t.status == "PASSED" for t in v.tools):
            bits.append("no tool passed a live invocation")
        L.append("FAILED (" + "; ".join(bits) + ")" if bits else "FAILED")
    return "\n".join(L) + "\n"


def write_validation(repo_url: str, repo_name: str, v: Validation) -> None:
    d = reports_dir(repo_url)
    d.mkdir(parents=True, exist_ok=True)
    (d / "validation.md").write_text(render_validation_md(repo_name, v), encoding="utf-8")
    (d / "validation.json").write_text(
        json.dumps({
            "syntax_ok": v.syntax_ok, "tests_passed": v.tests_passed,
            "tests_failed": v.tests_failed,
            "tools": [asdict(t) for t in v.tools],
            "debugger_actions": v.debugger_actions,
            "overall_ok": v.overall_ok,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
