import base64
import json
import logging
import os
import uuid
from io import StringIO
from typing import Annotated, Dict, List, Optional

import aiohttp
import pandas as pd
import pubchempy as pcp
import py3Dmol
import requests
from fastmcp import FastMCP
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import CalcMolDescriptors

from .clients.affinity_db import (
    VALID_AFFINITY_TYPES,
    fetch_affinity_bindingdb,
    fetch_chembl_data,
    fetch_uniprot_id,
)
from .clients.chemical_client import ChemServiceError
from .config import get_settings
from .ocr_pipeline import (
    extract_molecules_from_image_urls,
    extract_reactions_from_image_urls,
)
from .service_resources import chem_service, retrosynthesis_service, s3_service
from .utils.drawing_utils import draw_molecules_grid, draw_reactions_strip, draw_route_image


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

mcp = FastMCP("ChemTools")


@mcp.tool()
async def fetch_activity_data(
    source: str,
    protein_name: str,
    output_path: Annotated[str, "Full path to the output CSV file"],
    protein_id: Optional[str] = None,
    affinity_type: str = "IC50",
    cutoff: int = 10000,
) -> str:
    """
    Unified data retrieval tool for biochemical databases.

    Fetches protein-ligand interaction or activity data from BindingDB or ChEMBL.
    Saves results to the given CSV file path.

    Args:
        source (str): Data source ("bindingdb" or "chembl").
        protein_name (str): Target protein name.
        output_path (str): Full path to the output CSV file.
        protein_id (str, optional): Target protein id. If passed, protein_name is ignored.
        affinity_type (str, optional): Type of affinity (Ki, Kd, IC50). Defaults to "IC50".
        cutoff (int, optional): Optional threshold (nM) for BindingDB. Defaults to 10000.

    Returns:
        str: On success, path to the saved file and dataset info. On failure, error message.
    """
    source = source.lower().strip()
    if affinity_type not in VALID_AFFINITY_TYPES:
        return f"Invalid affinity type '{affinity_type}'. Must be one of {VALID_AFFINITY_TYPES}"

    try:
        async with aiohttp.ClientSession() as session:
            if source == "bindingdb":
                target_id = protein_id
                if not target_id:
                    resolved_id = await fetch_uniprot_id(session, protein_name)
                    if not resolved_id:
                        return f"[BindingDB] Could not find UniProt ID for '{protein_name}'"
                    target_id = resolved_id

                results = await fetch_affinity_bindingdb(
                    session, target_id, affinity_type, cutoff
                )
            elif source == "chembl":
                results = await fetch_chembl_data(
                    target_name=protein_name,
                    target_id=protein_id,
                    affinity_type=affinity_type,
                    session=session,
                )
            else:
                return f"Unsupported data source '{source}'. Use 'bindingdb' or 'chembl'."

        output_path = output_path.strip()
        if isinstance(results, list):
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            df = pd.DataFrame(results)
            if len(df) > 0:
                df.to_csv(output_path, index=False)
                buffer = StringIO()
                df.info(buf=buffer)
                info_str = buffer.getvalue()
                return (
                    f"The data was saved to {os.path.abspath(output_path)}. "
                    f"Here is info about dataset: {info_str}"
                )
            return "The data was not saved because it is empty."
        return results
    except Exception as e:
        return f"[fetch_activity_data] Error: {str(e)}"


@mcp.tool()
def name2smiles(
    mol: Annotated[str, "Name of a molecule"],
):
    """
    Convert a molecule name to its SMILES representation.
    
    This method retrieves the SMILES string for a given molecule name via PubChem (pubchempy).
    
    Args:
        mol (str): The name of the molecule to convert.
    
    Returns:
        str: The SMILES string representation of the molecule if successful,
             an error message if the request fails,
             or a "couldn't obtain smiles" message if the name is invalid.
    """
    try:
        compound = pcp.get_compounds(mol, "name")
        if not compound:
            return "I've couldn't obtain smiles, the name is wrong"
        return compound[0].canonical_smiles
    except requests.RequestException as e:
        return f"Failed to execute. Error: {repr(e)}"
    except (IndexError, AttributeError):
        return "I've couldn't obtain smiles, the name is wrong"


@mcp.tool()
def smiles2name(smiles: Annotated[str, "SMILES of a molecule"]):
    """
    Converts a SMILES string representing a molecule into its IUPAC name.
    
    Args:
        smiles (str): The SMILES string of the molecule.
    
    Returns:
        str: The IUPAC name of the molecule, or an error message if the conversion fails.
    """

    url = (
        "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/"
        f"{smiles}/property/IUPACName/JSON"
    )
    try:
        response = requests.get(url, timeout=60)
        if response.status_code == 200:
            data = response.json()
            properties = data.get("PropertyTable", {}).get("Properties", [])
            if not properties:
                return 'Could not find such IUPAC'  # or raise, or return a sentinel
            prop = properties[0]
            # CID 0 means PubChem couldn't resolve the SMILES
            if prop.get("CID", 0) == 0:
                return 'Could not find such IUPAC'
            return prop.get("IUPACName", 'Could not find such IUPAC')  # may still be None if name isn't available
        return "I've couldn't get iupac name"
    except requests.RequestException as e:
        return f"Failed to execute. Error: {repr(e)}"
    except (KeyError, IndexError, json.JSONDecodeError):
        return "I've couldn't get iupac name"


@mcp.tool()
def smiles2prop(
    smiles: Annotated[str, "SMILES of a molecule"], iupac: Optional[str] = None
):
    """
    Calculate molecular properties from a SMILES string or IUPAC name.
    
    Args:
        smiles (str): The SMILES string of the molecule.
        iupac (str, optional): The IUPAC name of the molecule. If provided, the SMILES string will be derived from it. Defaults to None.
    
    Returns:
        CalcMolDescriptors: An object containing calculated molecular properties. 
                             Returns an error message as a string if the calculation fails.
    """

    try:
        if iupac:
            compound = pcp.get_compounds(iupac, "name")
            if len(compound):
                smiles = compound[0].canonical_smiles

        res = CalcMolDescriptors(Chem.MolFromSmiles(smiles))
        return res
    except BaseException as e:
        return f"Failed to execute. Error: {repr(e)}"


@mcp.tool()
def visualize_molecule(
    smiles: Annotated[str, "SMILES of a molecule"],
) -> str:
    """
    Visualizes a molecule from its SMILES and returns a temporary presigned URL to an HTML file.

    The file is stored in S3 under 'visualizations/' with a auto-generated name
    and is intended for one-time viewing (URL expires in 1 hour).

    Args:
        smiles: SMILES string of the molecule.

    Returns:
        str: Presigned URL to the HTML visualization, or an error message.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return f"Invalid SMILES: {smiles}"

        AllChem.AddHs(mol, addCoords=True)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)

        view = py3Dmol.view(
            data=Chem.MolToMolBlock(mol),
            style={"stick": {}, "sphere": {"scale": 0.3}},
            width=600,
            height=400,
        )
        view.setBackgroundColor("#b8bfcc")
        view.zoomTo()

        s3_key = s3_service.upload_bytes(
            "chemical_mcp/molecule_visualizations",
            f"{uuid.uuid4()}.html",
            view.write_html().encode("utf-8"),
        )

        return s3_service.generate_presigned_url(s3_key, expiration=3600)

    except Exception as e:
        return f"Failed to visualize molecule. Error: {repr(e)}"


@mcp.tool()
def extract_reactions(
    image_urls: Annotated[List[str], "List of public HTTP(S) URLs of images"],
) -> Dict:
    """Detect chemical reactions in images loaded by URLs (ChemService).

    Each URL is processed in turn (in memory, no local cache). Annotated JPEGs are uploaded to S3
    under ``chemical_mcp/annotated_images``; presigned URLs are listed in metadata.

    Args:
        image_urls: One or more direct image links.

    Returns:
        dict: ``answer`` maps labels (from URL paths; disambiguated with ``__{index}`` on collision)
        to reaction dicts. ``metadata`` has ``annotated_image_presigned_urls``, ``source_urls``,
        and optionally ``failed`` (per-URL errors) if some URLs could not be processed.
    """
    try:
        response = extract_reactions_from_image_urls(image_urls)
        return response
    except ChemServiceError as e:
        logger.error("extract_reactions ChemServiceError: %s", e)
        return {"answer": f"ChemService reaction extraction failed: {e}"}
    except requests.RequestException as e:
        logger.error("extract_reactions download ERROR: %s", e)
        return {"answer": f"Could not download an image: {e}"}
    except Exception as e:
        logger.error("extract_reactions ERROR: %s", e)
        return {"answer": f"Could not extract reactions from images. Error: {e}"}


@mcp.tool()
def extract_molecules(
    image_urls: Annotated[List[str], "List of public HTTP(S) URLs of images"],
) -> Dict:
    """Detect molecular structures in images loaded by URLs (ChemService).

    Each URL is processed in turn. Annotated JPEGs go to S3 under ``chemical_mcp/annotated_images``;
    presigned URLs are returned in metadata.

    Args:
        image_urls: One or more direct image links.

    Returns:
        dict: ``answer`` maps labels to ``smiles`` / ``errors``. ``metadata`` includes
        ``annotated_image_presigned_urls``, ``source_urls``, and optionally ``failed``.
    """
    try:
        response = extract_molecules_from_image_urls(image_urls)
        return response
    except ChemServiceError as e:
        logger.error("extract_molecules ChemServiceError: %s", e)
        return {"answer": f"ChemService molecule extraction failed: {e}"}
    except requests.RequestException as e:
        logger.error("extract_molecules download ERROR: %s", e)
        return {"answer": f"Could not download an image: {e}"}
    except Exception as e:
        logger.error("extract_molecules ERROR: %s", e)
        return {"answer": f"Could not extract molecules from images. Error: {e}"}



@mcp.tool()
def calculate_docking(
    smiles: str,
    pdb_id: str,
) -> dict:
    """
    Calculate docking score for a molecule; upload the HTML visualization to S3 and return a presigned URL.

    Args:
        smiles: SMILES string of the molecule.
        pdb_id: PDB identifier for the receptor structure.

    Returns:
        dict: ``answer`` with affinity and errors; ``metadata`` may include ``docking_html_presigned_url``
        (temporary link, same TTL as molecule visualizations) when a visualization is returned.
    """
    try:
        response = chem_service.calculate_docking_score(smiles, pdb_id)
    except ChemServiceError as e:
        return {
            "answer": {"affinity": None, "errors": str(e)},
            "metadata": {},
        }

    if isinstance(response, dict) and "data" in response:
        data = response.get("data")
        errors = response.get("error")
    else:
        data = response if isinstance(response, dict) else None
        errors = response.get("error") if isinstance(response, dict) else None

    affinity = None
    presigned_url: Optional[str] = None

    if data:
        affinity = data.get("affinity")
        visualization = data.get("visualization")
        if visualization:
            if isinstance(visualization, (bytes, bytearray)):
                html_content = bytes(visualization)
            else:
                html_content = base64.b64decode(visualization)
            filename = f"docking_{pdb_id}_{uuid.uuid4()}.html"
            s3_key = s3_service.upload_bytes(
                "chemical_mcp/docking_results",
                filename,
                html_content,
            )
            presigned_url = s3_service.generate_presigned_url(s3_key, expiration=3600)

    return {
        "answer": {"affinity": affinity, "errors": errors},
        "metadata": (
            {"docking_html_presigned_url": presigned_url} if presigned_url else {}
        ),
    }
    
@mcp.tool()
def retrosynthesis_tree_search(
    smiles: Annotated[str, "Target molecule SMILES"],
    mode: Annotated[str, "One of: fast, balanced, deep"] = "fast",
) -> Dict:
    """
    Plan a retrosynthesis route for a target molecule.

    Use this when the user asks for possible synthetic routes or precursors
    for a target SMILES. This calls the retrosynthesis service and returns
    ASKCOS-like routes with steps, reactants, and scores.

    Args:
        smiles (str): Target molecule SMILES.
        mode (str): Search depth/quality preset ("fast", "balanced", "deep").

    Returns:
        dict: Retrosynthesis result payload with:
            - target (str | None): input target SMILES returned by ASKCOS.
            - routes (List[Dict]): list of retrosynthesis routes:
                - id (str): unique route identifier.
                - depth (int | None): longest path length in the route.
                - precursor_cost (float | None): summed precursor cost metric.
                - score (float | None): overall route score.
                - min_step_plausibility (float | None): lowest step plausibility.
                - avg_step_plausibility (float | None): average step plausibility.
                - steps (List[Dict]): ordered reaction steps:
                    - reaction_smiles (str): step reaction SMILES.
                    - mapped_smiles (str | None): atom-mapped reaction SMILES.
                    - plausibility (float | None): step plausibility score.
                    - precursor_rank (int | None): ranking of precursor set.
                    - precursor_score (float | None): model score for precursors.
                    - model_score (float | None): model score for the step.
                    - template (Dict | None): template metadata:
                        reaction_smarts (str): reaction SMARTS pattern.
                        template_rank (int | None): rank among templates.
                        num_examples (int | None): template training examples count.
                    - reactants (List[Dict]): precursor molecules:
                        smiles (str): molecule SMILES.
                        terminal (bool | None): True if purchasable/terminal.
                        buy_link (str | None): vendor link if available.
                        stoichiometry (int): reagent count (default 1).
                    - products (List[Dict]): products, same schema as reactants.
            - metadata (Dict): visualization info:
                - route_images (List[Dict]): one entry per route with:
                    - route_id (str): route identifier.
                    - s3_key (str): S3 object key.
                    - presigned_url (str): temporary URL to view the image (1 h TTL).
        On failure returns a dict with an "answer" message.
    """
    try:
        result = retrosynthesis_service.retrosynthesis_result(smiles=smiles, mode=mode)
    except Exception as e:
        logger.error(f"retrosynthesis_tree_search ERROR: {e}")
        return {"answer": "Could not run retrosynthesis tree search."}

    metadata: Dict = {}
    try:
        route_images = []
        for i, route in enumerate(result.get("routes", [])):
            route_id = route.get("id", f"route_{i}")
            img_bytes = draw_route_image(route)
            s3_key = s3_service.upload_bytes(
                "chemical_mcp/retrosynthesis",
                f"{route_id}_{uuid.uuid4()}.png",
                img_bytes,
            )
            route_images.append({
                "route_id": route_id,
                "s3_key": s3_key,
                "presigned_url": s3_service.generate_presigned_url(s3_key, expiration=3600),
            })
        metadata["route_images"] = route_images
    except Exception as e:
        logger.warning("retrosynthesis_tree_search: could not render images: %s", e)

    result["metadata"] = metadata
    return result

@mcp.tool()
def classify_reaction(
    smiles: Annotated[
        List[str],
        "Each entry is one full reaction SMILES: 'A.B>>C' (not separate molecules per list item)",
    ],
    num_results: Annotated[int, "Max classes per reaction (1..50)"] = 10,
) -> Dict:
    """
    Classify reaction SMILES into reaction classes.

    Use this when the user provides reaction SMILES and wants the reaction
    type/class (e.g., named reactions or class labels). Returns ASKCOS-like
    classification hits with ranks and confidence.

    Args:
        smiles (List[str]): One or more reaction strings; each is reactants>>products,
            with multiple reactants joined by "." (e.g. ["CCO.CC(=O)O>>CCOC(=O)C"]).
        num_results (int): Max number of classes per reaction (1..50).

    Returns:
        dict: ASKCOS classification payload with:
            - status_code (int): upstream status code.
            - message (str): upstream message.
            - result (List[Dict]): list of hits with:
                - rank (int): hit rank.
                - reaction_num (str): reaction identifier.
                - reaction_name (str): reaction name.
                - reaction_classnum (str): class number.
                - reaction_classname (str): class name.
                - reaction_superclassnum (str): superclass number.
                - reaction_superclassname (str): superclass name.
                - prediction_certainty (float): confidence score.
        On failure returns a dict with an "answer" message.
    """
    try:
        return retrosynthesis_service.classify_reaction_smiles(smiles=smiles, num_results=num_results)
    except Exception as e:
        logger.error(f"classify_reaction ERROR: {e}")
        return {"answer": "Could not classify reaction SMILES."}

@mcp.tool()
def forward_predict(
    smiles: Annotated[List[str], "Batch of reaction inputs (reactants)"],
    backend: Annotated[str, "One of: wldn5, graph2smiles, augmented_transformer"],
    retrosynthesis_model_name: Annotated[str, "Model name for backend"] = "pistachio",
    reagents: Annotated[str, "Reagents string"] = "",
    solvent: Annotated[str, "Solvent string"] = "",
) -> Dict:
    """
    Predict reaction products from reactants (forward synthesis).

    Use this when the user provides reactants and wants predicted products.
    You can specify backend/model_name and optional reagents/solvent strings.

    Args:
        smiles (List[str]): Batch of reaction inputs (reactants).
        backend (str): One of "wldn5", "graph2smiles", "augmented_transformer".
        retrosynthesis_model_name (str): Model name for the backend (default "pistachio").
        reagents (str): Reagents string as in ASKCOS controller.
        solvent (str): Solvent string as in ASKCOS controller.

    Returns:
        dict: ASKCOS forward payload with:
            - inputs (List[str]): normalized inputs (reactants+reagents+solvent).
            - backend (str): backend identifier used.
            - model_name (str): model name used.
            - predictions (List[Dict]): predicted products:
                - smiles (str): product SMILES.
                - score (float): model probability/score.
            - metadata (Dict): visualization info:
                - predictions_image (Dict):
                    - s3_key (str): S3 object key.
                    - presigned_url (str): temporary URL to view the grid image (1 h TTL).
                - top_reactions_image (Dict): reaction drawings for top 3 products:
                    - s3_key (str): S3 object key.
                    - presigned_url (str): temporary URL to view the reactions image (1 h TTL).
        On failure returns a dict with an "answer" message.
    """
    try:
        result = retrosynthesis_service.forward_predict_products(
            smiles=smiles,
            backend=backend,
            model_name=retrosynthesis_model_name,
            reagents=reagents,
            solvent=solvent,
        )
    except Exception as e:
        logger.error(f"forward_predict ERROR: {e}")
        return {"answer": "Could not run forward prediction."}

    metadata: Dict = {}
    try:
        predictions = result.get("predictions", [])
        smiles_list = [p["smiles"] for p in predictions if p.get("smiles")]
        labels = [
            f"score: {p['score']:.3f}" if isinstance(p.get("score"), float) else ""
            for p in predictions
            if p.get("smiles")
        ]
        img_bytes = draw_molecules_grid(smiles_list, labels=labels)
        s3_key = s3_service.upload_bytes(
            "chemical_mcp/forward_prediction",
            f"predictions_{uuid.uuid4()}.png",
            img_bytes,
        )
        metadata["predictions_image"] = {
            "s3_key": s3_key,
            "presigned_url": s3_service.generate_presigned_url(s3_key, expiration=3600),
        }
    except Exception as e:
        logger.warning("forward_predict: could not render images: %s", e)

    try:
        predictions = result.get("predictions", [])
        inputs = result.get("inputs", smiles)
        reactants_smi = ".".join(inputs) if isinstance(inputs, list) else str(inputs)
        top = [p for p in predictions if p.get("smiles")][:3]
        reactions = []
        for i, p in enumerate(top):
            rxn_smi = f"{reactants_smi}>>{p['smiles']}"
            score = p.get("score")
            label = f"Top {i + 1}"
            if isinstance(score, float):
                label += f"   score: {score:.3f}"
            reactions.append((rxn_smi, label))
        if reactions:
            rxn_img_bytes = draw_reactions_strip(reactions)
            rxn_s3_key = s3_service.upload_bytes(
                "chemical_mcp/forward_prediction",
                f"top_reactions_{uuid.uuid4()}.png",
                rxn_img_bytes,
            )
            metadata["top_reactions_image"] = {
                "s3_key": rxn_s3_key,
                "presigned_url": s3_service.generate_presigned_url(rxn_s3_key, expiration=3600),
            }
    except Exception as e:
        logger.warning("forward_predict: could not render reaction images: %s", e)

    result["metadata"] = metadata
    return result

def main() -> None:
    """Entry point for the MCP server."""
    settings = get_settings()
    mcp.run(
        transport="http",
        host=settings.chem_mcp_host,
        port=settings.chem_mcp_port,
        path=settings.chem_mcp_path,
    )


if __name__ == "__main__":
    main()
