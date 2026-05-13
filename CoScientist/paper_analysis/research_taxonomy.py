from typing import Literal, TypeAlias


ResearchDomain: TypeAlias = Literal[
    "Chemistry"
]


DOMAIN_TO_SUBDOMAINS: dict[str, list[str]] = {
    "Chemistry": [
        "Polymer Chemistry",
        "Organic Chemistry",
        "Nanomaterials",
        "Molecular Dynamics",
        "Membrane Chemistry",
        "Electrochemistry",
        "DFT",
        "Biological Macromolecules",
        "Biological Chemistry",
        "Analytical Chemistry"
    ]
}


def get_research_domains() -> list[str]:
    return list(DOMAIN_TO_SUBDOMAINS.keys())


def get_sub_domains_for_domain(domain: str | None) -> list[str]:
    if not domain:
        return []
    return DOMAIN_TO_SUBDOMAINS.get(domain, [])


def format_domain_subdomain_mapping_for_prompt() -> str:
    lines: list[str] = []
    for domain, subdomains in DOMAIN_TO_SUBDOMAINS.items():
        if subdomains:
            lines.append(f"- {domain}: {', '.join(subdomains)}")
        else:
            lines.append(f"- {domain}: no predefined sub-domains")
    return "\\n".join(lines)
