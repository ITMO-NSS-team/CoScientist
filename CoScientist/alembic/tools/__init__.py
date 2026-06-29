"""Agent-facing tools for the alembic pipeline.

This package keeps the original flat import surface — ``from alembic.tools
import clone_repo, setup_venv, ...`` — while splitting the implementation by
concern across submodules:

    _paths   workdir layout, constants, per-repo path helpers
    shell    bash / bash_env
    fs       clone/read/search repo, read/write output and reports
    venv     setup_venv, check_venv_compat
    invoke   validate_syntax, run_tests, invoke_mcp_tool
"""
from alembic.common import get_repo_name
from alembic.tools.paths import WORKDIR
from alembic.tools.shell import bash, bash_env
from alembic.tools.fs import (
    clone_repo, read_file, search, read_report,
    write_file, read_output_file, update_file, write_report,
)
from alembic.tools.venv import setup_venv, check_venv_compat
from alembic.tools.invoke import validate_syntax, run_tests, invoke_mcp_tool

__all__ = [
    "WORKDIR", "get_repo_name",
    "bash", "bash_env",
    "clone_repo", "read_file", "search", "read_report",
    "write_file", "read_output_file", "update_file", "write_report",
    "setup_venv", "check_venv_compat",
    "validate_syntax", "run_tests", "invoke_mcp_tool",
]
