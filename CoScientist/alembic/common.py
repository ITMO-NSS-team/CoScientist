from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

BASE_IMAGE = "alembic-base:latest"


def get_repo_name(repo_url: str) -> str:
    """Last path segment of a repo URL, without a trailing ``.git``."""
    return re.sub(r"\.git$", "", repo_url.rstrip("/").split("/")[-1])


def _image_exists(name: str) -> bool:
    """True if a local docker image with this name/tag exists."""
    return subprocess.run(
        ["docker", "image", "inspect", name],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def ensure_base_image(dockerfile: Path, project_root: Path,
                platform: str | None = None, rebuild: bool = False) -> None:
    """Build ``alembic-base:latest`` once if it is missing (or forced)."""
    if not rebuild and _image_exists(BASE_IMAGE):
        print(f"[alembic] base image {BASE_IMAGE} present — reusing.")
        return
    cmd = ["docker", "build"]
    if platform:
        cmd += ["--platform", platform]
    cmd += ["-t", BASE_IMAGE, "-f", str(dockerfile), str(project_root)]
    print(f"[alembic] building base image: {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(r.returncode)
