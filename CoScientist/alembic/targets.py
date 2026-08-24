"""Where a build runs: a Docker daemon, and what it needs to be addressed.

A build does not have to run on the machine that started it. It may go to a
remote Docker context — the box with the GPU, or the one with the cores — and
each such daemon needs slightly different handling: the ``--context`` argument,
sometimes a pinned API version, sometimes ``--gpus``.

Everything here is pure or injectable (the one probe, :func:`detect_gpu`, takes
a ``runner``), so target selection is testable without Docker or a GPU. Which
hosts exist is deployment configuration, not code: build the targets you have.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ExecutionTarget:
    """A Docker daemon to build on.

    ``docker_context`` is ``None`` for the local daemon, otherwise the name of a
    ``docker context``. ``gpus`` is forwarded as ``--gpus`` when set.
    ``api_version`` pins ``DOCKER_API_VERSION`` for a daemon old enough to
    reject the client's default — which is why the pin belongs to a target and
    is never global: a pin one host needs makes a newer host unreachable.
    """

    name: str
    docker_context: str | None = None
    gpus: str | None = None
    api_version: str | None = None

    @property
    def is_gpu(self) -> bool:
        return bool(self.gpus)

    @property
    def is_local(self) -> bool:
        return self.docker_context is None


LOCAL = ExecutionTarget(name="local")


def detect_gpu(runner=subprocess.run) -> bool:
    """True if this host exposes an NVIDIA GPU to Docker.

    Probes ``docker info`` for the nvidia runtime — that is what actually gates
    ``docker run --gpus`` — then falls back to ``nvidia-smi``. Any probe failure
    (no Docker, no driver, timeout) reads as no GPU, so ``--gpus`` is forwarded
    only when it will work.
    """
    for cmd, needle in (
        (["docker", "info", "--format", "{{.Runtimes}}"], "nvidia"),
        (["nvidia-smi", "-L"], "gpu"),
    ):
        try:
            result = runner(cmd, capture_output=True, text=True, check=False, timeout=20)
        except (OSError, subprocess.SubprocessError):
            continue
        if getattr(result, "returncode", 1) == 0 and needle in (result.stdout or "").lower():
            return True
    return False


def docker_cli(
    target: ExecutionTarget | None = None, *, context: str | None = None
) -> list[str]:
    """The ``docker`` argv prefix for a target: ``docker [--context X]``.

    An explicit ``context`` wins over the target's; neither means the local
    daemon and a bare ``docker``.
    """
    ctx = context if context is not None else (target.docker_context if target else None)
    return ["docker", "--context", ctx] if ctx else ["docker"]


def docker_env(
    base_env: dict[str, str] | None = None,
    *,
    target: ExecutionTarget | None = None,
    api_version: str | None = None,
) -> dict[str, str]:
    """The environment for a docker call, with ``DOCKER_API_VERSION`` pinned only
    where the target needs it. With nothing to pin the environment is returned
    unchanged — a spurious pin is what makes a newer daemon look unreachable.
    """
    env = dict(base_env if base_env is not None else os.environ)
    version = api_version if api_version is not None else (target.api_version if target else None)
    if version:
        env["DOCKER_API_VERSION"] = version
    return env


def with_gpus(target: ExecutionTarget, gpus: str | None) -> ExecutionTarget:
    """``target`` with its ``--gpus`` replaced (e.g. by what was detected)."""
    return replace(target, gpus=gpus)
