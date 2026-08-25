"""MCP binding repair."""
from __future__ import annotations

from .helpers import (
    _repo_fit_payload,
)

def test_repair_plan_mcp_bindings_fills_or_demotes():
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    inventory = [
        {
            "tool": "chemical_space_clustering",
            "server_id": "bfc62a287aaf7b5a",
            "description": "Cluster metabolites in chemical space",
        },
        {
            "tool": "predict_ld50",
            "server_id": "bfc62a287aaf7b5a",
            "description": "Predict LD50",
        },
    ]
    payload = {
        "source_request": "Cluster metabolites and report chemical space structure.",
        "tasks": [
            {
                "id": "EXP-1",
                "name": "Clustering metabolites",
                "description": "Cluster by molecular similarity",
                "route": "fedot_mas",
                "design": {
                    "analysis_artifacts": [
                        {
                            "name": "clustering_report.pdf",
                            "prepare_via": "mcp",
                            "path_or_tool": "clustering_report_generator",
                        }
                    ]
                },
                "mcp_servers": [],
            },
            {
                "id": "EXP-2",
                "name": "Obscure step",
                "description": "Something with no matching inventory tool",
                "route": "fedot_mas",
                "design": {
                    "analysis_artifacts": [
                        {"name": "x.json", "prepare_via": "mcp", "path_or_tool": "totally_fake_tool"}
                    ]
                },
                "mcp_servers": [],
            },
        ]
    }
    repaired = repair_plan_mcp_bindings(payload, inventory)
    assert repaired["tasks"][0]["mcp_servers"][0]["tools"][0]["name"] == "chemical_space_clustering"
    assert repaired["tasks"][0]["design"]["analysis_artifacts"][0]["path_or_tool"] == "chemical_space_clustering"
    # Unrelated leftover task is not bound by retrieval score.
    assert repaired["tasks"][1]["route"] == "coder"
    assert not repaired["tasks"][1].get("mcp_servers")


def test_repair_binds_generate_case_mols_not_docking_or_demote():
    """MADD-L shaped: empty mcp + affinity wording + Generate molecules → generation tool."""
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    inventory = [
        {
            "tool": "calculate_docking",
            "server_id": "srv-dock",
            "score": 0.31,
            "description": "Calculate docking score / affinity for a molecule.",
        },
        {
            "tool": "generate_case_mols",
            "server_id": "srv-gen",
            "score": 0.92,
            "description": "GAN molecule generation for disease cases.",
        },
    ]
    payload = {
        "source_request": (
            "Suggest several molecules that have high docking affinity with KRAS G12C protein. "
            "Molecules should possess common drug-like properties. "
            "Generate highly potent non-covalent BTK inhibitors."
        ),
        "tasks": [
            {
                "id": "EXP-1",
                "name": "Molecule Generation with High Docking Affinity",
                "description": (
                    "Identify and generate molecules that exhibit high docking affinity "
                    "for KRAS G12C and have drug-like properties."
                ),
                "route": "fedot_mas",
                "design": {
                    "experiment_question": (
                        "What molecules possess high docking affinity for KRAS G12C "
                        "protein and meet drug-like properties?"
                    ),
                    "analysis_artifacts": [
                        {
                            "name": "generated_molecules.json",
                            "prepare_via": "mcp",
                            "path_or_tool": "generated_molecules.json",
                        }
                    ],
                },
                "mcp_servers": [],
            }
        ],
    }
    repaired = repair_plan_mcp_bindings(payload, inventory)
    task = repaired["tasks"][0]
    assert task["route"] == "fedot_mas"
    assert task["mcp_servers"][0]["tools"][0]["name"] == "generate_case_mols"
    assert task["design"]["analysis_artifacts"][0]["path_or_tool"] == "generate_case_mols"
    assert not any("demoted_to_coder" in str(w) for w in (task.get("warnings") or []))


def test_repair_never_demotes_when_inventory_covers_request():
    """Live inventory covering the ask must bind, never demote to coder."""
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    inventory = [
        {
            "tool": "generate_case_mols",
            "server_id": "srv-gen",
            "description": "Generate case molecules",
        }
    ]
    payload = {
        "source_request": "Generate small molecules for alzheimer anti-inflammatory case.",
        "tasks": [
            {
                "id": "EXP-1",
                "name": "Generate candidates",
                "description": "Generate molecules for the case",
                "route": "fedot_mas",
                "design": {
                    "experiment_question": "Which generated molecules look drug-like?",
                    "analysis_artifacts": [
                        {
                            "name": "out.csv",
                            "prepare_via": "mcp",
                            "path_or_tool": "candidates.csv",
                        }
                    ],
                },
                "mcp_servers": [],
            }
        ],
    }
    repaired = repair_plan_mcp_bindings(payload, inventory)
    assert repaired["tasks"][0]["route"] == "fedot_mas"
    assert repaired["tasks"][0]["mcp_servers"][0]["tools"][0]["name"] == "generate_case_mols"
    assert not any(
        "demoted_to_coder" in str(w) for w in (repaired["tasks"][0].get("warnings") or [])
    )


def test_repair_demotes_only_when_inventory_empty():
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    payload = {
        "source_request": "Generate molecules with high docking affinity.",
        "tasks": [
            {
                "id": "EXP-1",
                "name": "Generate",
                "description": "Generate molecules",
                "route": "fedot_mas",
                "design": {
                    "analysis_artifacts": [
                        {"name": "out.json", "prepare_via": "mcp", "path_or_tool": "x"}
                    ]
                },
                "mcp_servers": [],
            }
        ],
    }
    repaired = repair_plan_mcp_bindings(payload, [])
    assert repaired["tasks"][0]["route"] == "coder"
    assert any("empty_inventory" in w for w in repaired["tasks"][0]["warnings"])


def test_repair_rewrites_coder_generate_inhibitors_via_family():
    """Unnamed generate-inhibitors slot + generate MCP → Fedot (glue asks)."""
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    payload = {
        "source_request": (
            "Generate GSK-3beta inhibitors with high activity. "
            "Suggest some small molecules that inhibit KRAS G12C."
        ),
        "tasks": [{
            "id": "EXP-1",
            "name": "GSK-3beta inhibitors",
            "description": "Generate GSK-3beta inhibitors with high activity",
            "operation_ref": "OP-1",
            "route": "coder",
            "design": {},
            "mcp_servers": [],
        }],
    }
    covering = [{
        "server_id": "srv-gen",
        "tool": "generate_case_mols",
        "description": "Generate molecules for a hardcoded disease case.",
    }]
    repaired = repair_plan_mcp_bindings(
        payload, covering,
        operations=[{"operation_id": "OP-1", "statement": "Generate GSK-3beta inhibitors with high activity"}],
    )
    assert repaired["tasks"][0]["route"] == "fedot_mas"
    assert repaired["tasks"][0]["mcp_servers"][0]["tools"][0]["name"] == "generate_case_mols"


def test_repair_rewrites_coder_when_inventory_covers_task():
    """Coder that names an inventory tool → Fedot, not Alembic and not a gap."""
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    payload = {
        "source_request": "Estimate synthetic accessibility SA score for SMILES",
        "tasks": [{
            "id": "EXP-1",
            "name": "SA score",
            "description": "Call smiles2prop to compute SA score",
            "route": "coder",
            "design": {},
            "mcp_servers": [],
        }],
    }
    covering = [{
        "server_id": "srv",
        "tool": "smiles2prop",
        "description": "molecular properties and SA score / synthesizability",
    }]
    repaired = repair_plan_mcp_bindings(
        payload, covering,
        repo_candidates=[{"url": "https://github.com/whitead/synspace"}],
        route_alembic=True,
    )
    assert repaired["tasks"][0]["route"] == "fedot_mas"
    assert repaired["tasks"][0]["mcp_servers"][0]["tools"][0]["name"] == "smiles2prop"
    assert "auto_rewrote_coder_to_ready_mcp" in repaired["tasks"][0]["warnings"]


def test_repair_keeps_unnamed_coder_required_when_leftover_inventory_has_no_repo():
    """Unnamed coder + leftover MCP for a different op stays required coder."""
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    payload = {
        "source_request": "Curate a metabolite dataset from literature, then cluster.",
        "tasks": [{
            "id": "EXP-1",
            "name": "Curate Metabolite Dataset",
            "description": "Compile SMILES from literature review",
            "route": "coder",
            "design": {},
            "mcp_servers": [],
        }],
    }
    leftover = [{
        "server_id": "srv-tox",
        "tool": "predict_general_toxicity",
        "description": "Predict general toxicity LD50",
        "score": 0.9,
    }]
    repaired = repair_plan_mcp_bindings(payload, leftover, route_alembic=True)
    task = repaired["tasks"][0]
    assert task["route"] == "coder"
    assert task.get("optional") is not True
    assert not any("capability_gap" in str(w) for w in (task.get("warnings") or []))
    assert not task.get("mcp_servers")

def test_repair_routes_to_alembic_when_inventory_empty_and_repo_fits():
    """No inventory, route_alembic on, a repo fits → auto-route instead of coder."""
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    repaired = repair_plan_mcp_bindings(
        _repo_fit_payload("fedot_mas"), [],
        repo_candidates=[{"url": "https://github.com/whitead/synspace"}],
        route_alembic=True,
    )
    task = repaired["tasks"][0]
    assert task["route"] == "alembic_build"
    assert task["repo_url"] == "https://github.com/whitead/synspace"
    assert task["post_build_route"] == "fedot_mas"
    assert task["mcp_servers"] == []
    assert any("auto_routed_alembic" in w for w in task["warnings"])


def test_repair_ignores_repo_candidates_when_route_alembic_disabled():
    """route_alembic off → still demote to coder even with a fitting repo."""
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    repaired = repair_plan_mcp_bindings(
        _repo_fit_payload("fedot_mas"), [],
        repo_candidates=[{"url": "https://github.com/whitead/synspace"}],
        route_alembic=False,
    )
    assert repaired["tasks"][0]["route"] == "coder"


def test_repair_routes_unnamed_coder_to_alembic_when_repo_fits_despite_leftover_inventory():
    """S5: leftover MCP for another op is not coverage; a fitting repo → alembic."""
    from CoScientist.experiments.critique.mcp_repair import repair_plan_mcp_bindings

    payload = {
        "source_request": "10 ps molecular dynamics of KRAS G12C using OpenMM",
        "tasks": [{
            "id": "EXP-1",
            "name": "OpenMM MD",
            "description": "Run explicit-solvent MD",
            "route": "coder",
            "design": {},
            "mcp_servers": [],
        }],
    }
    leftover = [{
        "server_id": "srv",
        "tool": "smiles2prop",
        "description": "molecular properties and SA score / synthesizability",
        "score": 0.4,
    }]
    repaired = repair_plan_mcp_bindings(
        payload, leftover,
        repo_candidates=[{"url": "https://github.com/openmm/openmm"}],
        route_alembic=True,
    )
    task = repaired["tasks"][0]
    assert task["route"] == "alembic_build"
    assert task.get("optional") is not True
    assert not task.get("mcp_servers")

