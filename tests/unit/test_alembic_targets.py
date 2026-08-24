"""A build can run on another machine, and each daemon needs its own handling.

No Docker and no GPU here: the one probe takes an injectable runner.
"""

import subprocess

from CoScientist.alembic.targets import (
    ExecutionTarget,
    LOCAL,
    detect_gpu,
    docker_cli,
    docker_env,
    with_gpus,
)


class _Result:
    def __init__(self, stdout="", returncode=0):
        self.stdout = stdout
        self.returncode = returncode


def _runner(answers):
    def run(cmd, **kw):
        return answers.get(cmd[0], _Result(returncode=1))

    return run


def test_a_gpu_is_detected_from_dockers_own_runtimes():
    """That is what actually gates `docker run --gpus`."""
    assert detect_gpu(_runner({"docker": _Result("[nvidia runc]")}))


def test_nvidia_smi_is_the_fallback():
    assert detect_gpu(_runner({"nvidia-smi": _Result("GPU 0: NVIDIA A100")}))


def test_no_gpu_when_neither_probe_says_so():
    assert not detect_gpu(_runner({"docker": _Result("[runc]")}))


def test_a_broken_probe_reads_as_no_gpu():
    """Better to omit --gpus than to pass it where it will fail."""

    def boom(cmd, **kw):
        raise subprocess.SubprocessError("docker missing")

    assert not detect_gpu(boom)


def test_the_local_daemon_takes_a_bare_docker():
    assert docker_cli(LOCAL) == ["docker"]


def test_a_remote_target_is_addressed_by_context():
    target = ExecutionTarget(name="gpu-box", docker_context="gpu")

    assert docker_cli(target) == ["docker", "--context", "gpu"]


def test_an_explicit_context_wins():
    assert docker_cli(LOCAL, context="other") == ["docker", "--context", "other"]


def test_an_old_daemon_gets_its_api_version_pinned():
    target = ExecutionTarget(name="old", docker_context="old", api_version="1.43")

    assert docker_env({}, target=target)["DOCKER_API_VERSION"] == "1.43"


def test_a_target_without_a_pin_gets_none():
    """A pin one host needs makes a newer host look unreachable."""
    assert "DOCKER_API_VERSION" not in docker_env({}, target=LOCAL)


def test_gpu_access_can_be_set_from_what_was_detected():
    target = with_gpus(LOCAL, "all")

    assert target.is_gpu and target.is_local
