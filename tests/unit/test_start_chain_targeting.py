"""start_chain builds where it is told and advertises an address that works.

start_chain imports the alembic package under its container layout, so the
module is loaded standalone with those dependencies stubbed — the same reason
tests/unit/_codegen_loader.py exists.
"""

import argparse
import importlib.util
import sys
import types
from pathlib import Path

_START_CHAIN = (
    Path(__file__).resolve().parents[2] / "CoScientist" / "alembic" / "start_chain.py"
)


def _load():
    for name in ("alembic", "alembic.common", "alembic.remote", "alembic.targets"):
        sys.modules.pop(name, None)
    pkg = types.ModuleType("alembic")
    pkg.__path__ = [str(_START_CHAIN.parent)]
    sys.modules["alembic"] = pkg
    spec = importlib.util.spec_from_file_location("alembic_start_chain_under_test", _START_CHAIN)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


sc = _load()


def _ns(**kw):
    base = dict(mount_dir=None, context=None, stage_volume=None, advertise_host=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_no_data_directory_means_no_mount(monkeypatch):
    assert sc._mount_args("repo", _ns()) == []


def test_a_local_build_binds_the_host_path(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd))

    args = sc._mount_args("repo", _ns(mount_dir=str(tmp_path)))

    assert args == ["-v", f"{tmp_path}:/mount/data:ro"]
    assert ran == []  # nothing to stage


def test_a_remote_build_stages_the_data_first(tmp_path, monkeypatch):
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd) or _Ok())
    monkeypatch.setattr(sc, "context_endpoint", lambda ctx: "ssh://user@gpu-box:22")

    args = sc._mount_args("repo", _ns(mount_dir=str(tmp_path), context="gpu"))

    assert any("cp" in cmd for cmd in ran)  # the data was copied over
    assert args[0] == "-v" and args[1].endswith(":/mount/data:ro")
    assert str(tmp_path) not in args[1]  # a local path would not exist there


def test_a_named_volume_that_already_exists_is_reused(tmp_path, monkeypatch):
    """Staging a large dataset again on every build is the cost this avoids."""
    ran = []
    monkeypatch.setattr(sc, "_run", lambda cmd, **kw: ran.append(cmd) or _Ok())
    monkeypatch.setattr(sc, "context_endpoint", lambda ctx: "ssh://user@gpu-box:22")
    monkeypatch.setattr(sc, "_volume_exists", lambda ctx, vol: True)

    args = sc._mount_args("repo", _ns(mount_dir=str(tmp_path), context="gpu", stage_volume="ct"))

    assert ran == []
    assert args == ["-v", "ct:/mount/data:ro"]


class _Ok:
    returncode = 0
