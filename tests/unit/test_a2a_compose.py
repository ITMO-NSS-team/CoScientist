"""Structural checks for the A2A Docker Compose stack."""

from pathlib import Path

import yaml


def _compose() -> dict:
    path = Path(__file__).resolve().parents[2] / "docker" / "docker-compose.a2a.yml"
    return yaml.safe_load(path.read_text())


def test_a2a_image_is_built_once_and_reused_by_all_services():
    services = _compose()["services"]
    a2a_services = {
        name: service
        for name, service in services.items()
        if name.startswith("a2a-")
    }

    assert {service["image"] for service in a2a_services.values()} == {
        "coscientist-a2a:latest"
    }
    assert [
        name for name, service in a2a_services.items()
        if "build" in service
    ] == ["a2a-orchestrator"]
