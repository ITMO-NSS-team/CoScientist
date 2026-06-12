"""Standalone Opik credential resolution + client.

Reads credentials without importing the heavy CoScientist package, in priority
order: process env (OPIK_API_KEY / OPIK__API_KEY) > ~/.opik.config > repo .env.
"""
from __future__ import annotations

import configparser
import os
import pathlib

DEFAULTS = {
    "workspace": "itmo-nss",
    "host": "https://www.comet.com/opik/api/",
    "project": "adk-coscientist",
}

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _from_opik_config() -> dict:
    cfg = pathlib.Path.home() / ".opik.config"
    if not cfg.exists():
        return {}
    p = configparser.ConfigParser()
    p.read(cfg)
    if not p.has_section("opik"):
        return {}
    s = p["opik"]
    out = {
        "api_key": s.get("api_key"),
        "workspace": s.get("workspace"),
        "host": s.get("url_override"),
        "project": s.get("project_name"),
    }
    return {k: v for k, v in out.items() if v}


def _from_dotenv() -> dict:
    env = _REPO_ROOT / ".env"
    out: dict = {}
    if not env.exists():
        return out
    for line in env.read_text().splitlines():
        line = line.strip()
        if line.startswith("OPIK__API_KEY="):
            out["api_key"] = line.split("=", 1)[1].strip()
        elif line.startswith("OPIK__URL_OVERRIDE="):
            out["host"] = line.split("=", 1)[1].strip()
    return out


def resolve_config() -> dict:
    cfg = dict(DEFAULTS)
    cfg.update(_from_dotenv())          # lowest priority
    cfg.update(_from_opik_config())     # mid
    env_key = os.environ.get("OPIK_API_KEY") or os.environ.get("OPIK__API_KEY")
    if env_key:                          # highest
        cfg["api_key"] = env_key
    for env_name, key in (("OPIK_WORKSPACE", "workspace"),
                          ("OPIK_PROJECT", "project"),
                          ("OPIK_URL_OVERRIDE", "host")):
        if os.environ.get(env_name):
            cfg[key] = os.environ[env_name]
    return cfg


def get_client():
    """Return (opik.Opik client, project_name)."""
    import opik

    cfg = resolve_config()
    if not cfg.get("api_key"):
        raise SystemExit(
            "No Opik api_key found. Set OPIK_API_KEY, or ~/.opik.config, or .env OPIK__API_KEY."
        )
    client = opik.Opik(workspace=cfg["workspace"], api_key=cfg["api_key"], host=cfg["host"])
    return client, cfg["project"]
