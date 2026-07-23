"""Tests for keeping package import light enough for module entry points."""

import importlib
import sys


def test_package_import_does_not_build_agents():
    for name in list(sys.modules):
        if name == "CoScientist" or name.startswith("CoScientist."):
            del sys.modules[name]

    package = importlib.import_module("CoScientist")

    assert package.__version__ == "1.0.0"
    assert "CoScientist.agents" not in sys.modules
    assert "CoScientist.main" not in sys.modules
