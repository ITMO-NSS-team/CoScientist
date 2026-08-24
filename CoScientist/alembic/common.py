from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

BASE_IMAGE = "alembic-base:latest"

# A digest of the sources baked into the base image, stamped on it as a label.
# Without it a base image built weeks ago is reused as if it were current, and
# the pipeline silently runs old code — the failure mode is invisible, because
# everything works, just not with the code in the checkout. The paths below must
# track what the Dockerfile COPYs in.
_LABEL_KEY = "alembic.source_sha"
_BAKED_PATHS = (
    "CoScientist/alembic",
    "docker/alembic/requirements.txt",
    "docker/alembic/entrypoint.py",
    "docker/alembic/serve.py",
)


def get_repo_name(repo_url: str) -> str:
    """Last path segment of a repo URL, without a trailing ``.git``."""
    return re.sub(r"\.git$", "", repo_url.rstrip("/").split("/")[-1])


def _image_exists(name: str) -> bool:
    """True if a local docker image with this name/tag exists."""
    return subprocess.run(
        ["docker", "image", "inspect", name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def _iter_baked_files(project_root: Path) -> Iterator[Path]:
    """Every file baked into the base image, in a deterministic order.

    ``__pycache__``/``.pyc`` are skipped so a stray bytecode file cannot
    perturb the digest.
    """
    for rel in _BAKED_PATHS:
        base = project_root / rel
        if base.is_file():
            yield base
        elif base.is_dir():
            for path in sorted(base.rglob("*")):
                if (
                    path.is_file()
                    and "__pycache__" not in path.parts
                    and path.suffix != ".pyc"
                ):
                    yield path


def _source_hash(dockerfile: Path, project_root: Path) -> str:
    """Digest of the Dockerfile plus everything it copies into the base image.

    The same checkout hashes identically on any machine, so the value doubles as
    a build identifier.
    """
    digest = hashlib.sha256()
    digest.update(b"Dockerfile\0")
    try:
        digest.update(dockerfile.read_bytes())
    except OSError:
        pass
    for path in _iter_baked_files(project_root):
        digest.update(str(path.relative_to(project_root)).encode() + b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _image_label(name: str, key: str) -> str | None:
    """Value of image label ``key``, or ``None`` if the image or label is absent."""
    result = subprocess.run(
        ["docker", "image", "inspect", "--format",
         f'{{{{ index .Config.Labels "{key}" }}}}', name],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return None
    value = result.stdout.strip()
    # docker renders a missing label as "<no value>".
    return value if value and value != "<no value>" else None


def ensure_base_image(dockerfile: Path, project_root: Path,
                platform: str | None = None, rebuild: bool = False) -> None:
    """Build ``alembic-base:latest`` when it is missing, forced, or out of date.

    Out of date means the sources baked into the present image differ from the
    checkout — detected through the ``alembic.source_sha`` label. Rebuilding on
    that difference is what stops a cached image from quietly running old
    pipeline code.
    """
    want = _source_hash(dockerfile, project_root)
    if not rebuild and _image_exists(BASE_IMAGE):
        have = _image_label(BASE_IMAGE, _LABEL_KEY)
        if have == want:
            print(f"[alembic] base image {BASE_IMAGE} up to date ({want}) — reusing.")
            return
        print(
            f"[alembic] base image {BASE_IMAGE} is stale "
            f"(built from {have or 'unlabelled sources'}, checkout is {want}) "
            "— rebuilding.",
            flush=True,
        )
    cmd = ["docker", "build"]
    if platform:
        cmd += ["--platform", platform]
    cmd += ["--label", f"{_LABEL_KEY}={want}",
            "-t", BASE_IMAGE, "-f", str(dockerfile), str(project_root)]
    print(f"[alembic] building base image: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
