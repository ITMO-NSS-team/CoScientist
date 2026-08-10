"""Promote CoderAgent sandbox files into experiment artifact lineage."""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, MutableMapping

from CoScientist.config import get_settings
from CoScientist.experiments.runtime.shared import artifact_name_key

_WORKSPACE_STATE_KEY = "coder_workspace_id"


def _find_candidate(root: Path, expected_name: str) -> Path | None:
    """Locate a sandbox file by exact basename, then by normalized stem."""
    if not root.is_dir():
        return None
    target = Path(expected_name).name
    if (exact := root / target).is_file():
        return exact
    if matches := [p for p in root.rglob(target) if p.is_file()]:
        return min(matches, key=lambda p: len(p.parts))
    if not (want := artifact_name_key(target)):
        return None
    soft = [p for p in root.rglob("*") if p.is_file() and artifact_name_key(p.name) == want]
    return min(soft, key=lambda p: len(p.parts)) if soft else None


def promote_coder_workspace_artifacts(
    state: MutableMapping[str, Any],
) -> list[dict[str, Any]]:
    """Copy expected coder outputs into experiment_artifacts and capture them."""
    runtime = state.get("experiment_runtime")
    if not isinstance(runtime, dict):
        return []
    task_id, attempt_id = runtime.get("active_task_id"), runtime.get("active_attempt_id")
    if not task_id or not attempt_id:
        return []
    expected = (((runtime.get("tasks") or {}).get(task_id) or {}).get("task") or {}).get("expected_artifacts") or []
    if not isinstance(expected, list) or not expected:
        return []

    workspace_id = state.get(_WORKSPACE_STATE_KEY) or ""
    root = Path(get_settings().code_exec.workspace_root)
    sandbox = root / str(workspace_id) if workspace_id else None
    if sandbox is None or not sandbox.is_dir():
        return []

    dest_dir = root / "experiment_artifacts" / str(task_id) / str(attempt_id)
    dest_dir.mkdir(parents=True, exist_ok=True)

    bucket = state.setdefault("coder_artifacts", [])
    if not isinstance(bucket, list):
        state["coder_artifacts"] = []
        bucket = state["coder_artifacts"]
    existing = {str(item.get("workspace_path") or "") for item in bucket if isinstance(item, dict)}

    promoted: list[dict[str, Any]] = []
    for item in expected:
        if not isinstance(item, dict) or not (name := str(item.get("name") or "").strip()):
            continue
        if (source := _find_candidate(sandbox, name)) is None:
            continue
        destination = dest_dir / Path(name).name
        if destination.resolve() != source.resolve():
            shutil.copy2(source, destination)
        payload = destination.read_bytes()
        record = {
            "name": name,
            "role": item.get("role") or "data",
            "media_type": item.get("media_type"),
            "workspace_path": str(destination),
            "size_bytes": len(payload),
            "checksum_sha256": hashlib.sha256(payload).hexdigest(),
            "producer_tool": "CoderAgent",
            "source": "coder_workspace",
        }
        if record["workspace_path"] not in existing:
            bucket.append(record)
            existing.add(record["workspace_path"])
            promoted.append(record)
    return promoted


__all__ = ["promote_coder_workspace_artifacts"]
