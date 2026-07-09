"""Single source of truth for models, timeouts, and loop caps.

Everything tunable lives here so there is exactly one place to look. This
module imports nothing from the rest of the package, so any module may import
it at top level without risking a circular import (which is why the old
deferred-import-from-main.py dance for timeouts is gone).
"""
from __future__ import annotations

import os

# ── Model ───────────────────────────────────────────────────────────────────
# One knob for every agent. Default is the qwen dev model; the top-quality
# benchmark run sets MODEL=z-ai/glm-5. Sampling params are passed to the
# provider ONLY when explicitly set (leaving them unset = provider default,
# which avoids the temperature-0 repetition loops observed on qwen — see
# docs/audit/02-stability.md N1). Determinism comes from the deterministic
# gates, not from pinning the sampler.
MODEL             = os.environ.get("MODEL", "openrouter/qwen/qwen3-235b-a22b-2507")
MODEL_TEMPERATURE = os.environ.get("MODEL_TEMPERATURE")  # None => unset
MODEL_TOP_P       = os.environ.get("MODEL_TOP_P")

# Optional target-task spec (TM-Bench mode). A JSON string, or a path to a
# JSON/YAML file, describing {description, arguments, returns, example}. When
# set, the Explorer/Coder must produce a tool matching it and the Validator
# checks against its declared returns. Unset => native autonomous mode.
TARGET_TASK = os.environ.get("ALEMBIC_TARGET_TASK")

APP_NAME = "alembic_app"
USER_ID  = "user_1"

STAGES = ("explorer", "environment", "coder", "validator")

# ── Per-stage wall-clock budgets (seconds) ────────────────────────────────────
STAGE_TIMEOUT = {
    "explorer":    900,    # read + propose
    "environment": 2400,   # venv + heavy ML deps (biggest cost)
    "coder":       1500,   # server.py + helpers + tests
    "validator":   2400,   # syntax + pytest + per-tool invocation + repair
}

# Reserve the last 15% of a stage's budget to force a "write what you have"
# report before the hard timeout cancels everything (F32).
REPORT_GRACE_FRACTION = 0.85

# ── Other wall-clock timeouts ─────────────────────────────────────────────────
BASH_TIMEOUT                = 15    # quick reads (ls/grep/head)
BASH_ENV_TIMEOUT            = 900   # slow installs / weight downloads
VENV_SETUP_TIMEOUT          = 600   # a single setup_venv subprocess (N10)
VENV_COMPAT_TIMEOUT         = 240   # compat-check script
SERVER_IMPORT_CHECK_TIMEOUT = 60    # server.py import-exec (heavy ML imports are slow)
HELPER_IMPORT_CHECK_TIMEOUT = 60    # per-helper import check (F28)
PYTEST_TIMEOUT              = 120
INVOKE_TIMEOUT              = 120    # a single live tool invocation; slower => SKIPPED (F37)
DEBUGGER_CALL_TIMEOUT       = 600    # one debugger round-trip (F16)
REPORTER_TIMEOUT            = 300    # F35 fallback reporter cap

# ── Loop breakers ─────────────────────────────────────────────────────────────
MAX_STEPS         = 120   # hard ceiling on events per agent turn
MAX_TOOL_REPEATS  = 3     # abort on N identical consecutive tool calls
MAX_TOOL_CYCLE    = 3     # abort on N identical NON-consecutive calls (set-cycling)
MAX_GUARD_RETRIES = 3     # re-nudge an agent that missed write_report / venv
MAX_TRANSIENT_FAULT_RETRIES = 2  # retry a silent provider fault (F22), off-budget
MAX_STATIC_GATE_RETRIES     = 2  # coder re-tries after a failed static gate

# Tools whose own instructions require repeated identical-arg calls — exempt
# from the non-consecutive cycle breaker but not the consecutive one (F36).
TOOL_CYCLE_EXEMPT = frozenset({"validate_syntax", "run_tests", "check_venv_compat"})

# ── Tool-selection caps (Plan gate) ───────────────────────────────────────────
MAX_TOOLS = 12   # hard cap on tools exposed per repo (matches ToolRosella)

# ── Output size caps ──────────────────────────────────────────────────────────
MAX_BYTES              = 40_000   # stdout/stderr text shown to the LLM
RESULT_MAX_LIST_ITEMS  = 20       # cap list length in a successful tool result (F30)
RESULT_MAX_STR_LEN     = 2_000    # cap string length in a successful tool result (F30)

# Sentinel that separates a helper's real stdout from its JSON result (N5).
RESULT_SENTINEL = "<<<ALEMBIC_RESULT>>>"
