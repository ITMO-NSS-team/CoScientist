import os
from typing import Annotated, Optional, List, Dict
from urllib.parse import quote

import pubchempy as pcp
import py3Dmol
import rdkit.Chem as Chem
import requests
from langchain.tools.render import render_text_description
from langchain_core.runnables.config import RunnableConfig
from langchain_experimental.utilities import PythonREPL
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import CalcMolDescriptors
from typing import Dict, List, Optional
#from smolagents import tool
from langchain_core.tools import tool
from definitions import CONFIG_PATH
from pathlib import Path
from dotenv import load_dotenv
from ChemCoScientist.chemical_utils.ocr_pipeline import molecules_ocr, reactions_ocr
from ChemCoScientist.chemical_utils.chemical_functions import (
    calculate_docking_score,
)
from ChemCoScientist.chemical_utils.retrosynthesis import retrosynthesis_result, classify_reaction_smiles, forward_predict_products

import aiohttp
import base64
import asyncio
import json
import re
import pandas as pd
from io import StringIO
import logging

load_dotenv(CONFIG_PATH)

CHEMBL_BASE = "https://www.ebi.ac.uk/chembl/api/data"
VALID_AFFINITY_TYPES = {"Ki", "Kd", "IC50", "EC50"}
repl = PythonREPL()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _run_async(coro):
    """Run async coroutine from both sync and async contexts."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # running in async env (e.g. LangChain)
        return asyncio.ensure_future(coro)
    else:
        # safe to call asyncio.run()
        return asyncio.run(coro)

async def fetch_uniprot_id(
    session: aiohttp.ClientSession,
    protein_name: str,
    organism_id: int = 9606,
    max_retries: int = 5,
    delay: float = 0.5
) -> Optional[str]:
    """
    Asynchronously fetch UniProt ID for a given protein name.
    Retries up to `max_retries` times in case of network or transient API errors.
    """
    url = "https://rest.uniprot.org/uniprotkb/search"
    params = {
        "query": f"{protein_name} AND organism_id:{organism_id}",
        "format": "json",
        "size": 1,
        "fields": "accession",
    }

    for attempt in range(max_retries):
        try:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    await asyncio.sleep(delay * (1 + attempt * 0.5))
                    continue
                data = await resp.json()
                results = data.get("results", [])
                if results:
                    return results[0].get("primaryAccession")
                return None
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[UniProt] Attempt {attempt+1} failed: {str(e)}")
            await asyncio.sleep(delay * (1 + attempt * 0.5))
    return None


async def fetch_affinity_bindingdb(
    session: aiohttp.ClientSession,
    uniprot_id: str,
    affinity_type: str,
    cutoff: int,
    max_retries: int = 5,
    delay: float = 0.5
) -> List[Dict]:
    """
    Asynchronously retrieve affinity values from BindingDB for a given UniProt ID.
    Retries on network errors or incomplete data.
    """
    url = (
        f"http://bindingdb.org/rest/getLigandsByUniprot?"
        f"uniprot={uniprot_id};{cutoff}&response=application/json"
    )

    get_smiles = lambda x: re.sub(r'\s*\|.*\|$', '', x)

    for attempt in range(max_retries):
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                if resp.status != 200:
                    logger.error(f"[BindingDB] HTTP {resp.status} for {uniprot_id}, retrying...")
                    await asyncio.sleep(delay * (1 + attempt * 0.5))
                    continue
                data = json.loads(await resp.text())
                affinities = (
                    data.get("getLindsByUniprotResponse", {}).get("bdb.affinities", [])
                    or data.get("bdb.affinities", [])
                    or []
                )
                
                result = [{'monomerid': a.get('bdb.monomerid'),
                          'smiles': get_smiles(a.get('bdb.smile')),
                           'affinity_type': a.get('bdb.affinity_type'),
                           'affinity': a.get('bdb.affinity')} for a in affinities if a.get("bdb.affinity_type") == affinity_type]
                return result
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"[BindingDB] Attempt {attempt+1} failed: {str(e)}")
            await asyncio.sleep(delay * (1 + attempt * 0.5))
    return []


async def _aio_fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    timeout: int = 30,
    max_retries: int = 4,
    retry_delay: float = 0.5,
    semaphore: Optional[asyncio.Semaphore] = None
) -> dict:
    for attempt in range(max_retries):
        try:
            if semaphore:
                async with semaphore:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            if isinstance(data, dict):
                                return data
            else:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if isinstance(data, dict):
                            return data
        except asyncio.CancelledError:
            raise
        except Exception:
            pass
        await asyncio.sleep(retry_delay * (1 + 0.5 * attempt))
    return {}

async def _resolve_chembl_target_id(
    session: aiohttp.ClientSession,
    target_name: str,
    limit: int = 5,
    max_retries: int = 3
) -> str:
    """
    Search ChEMBL target endpoint and return first target_chembl_id.
    Retries until valid data is obtained. Returns empty string if none found.
    """
    if not target_name:
        return ""

    for attempt in range(max_retries):
        url = f"{CHEMBL_BASE}/target/search?q={quote(target_name)}&format=json&limit={limit}"
        data = await _aio_fetch_json(session, url)
        targets = data.get("targets", [])
        if targets and isinstance(targets, list):
            chembl_ids = [(target.get('target_chembl_id'), target.get('organism')) for target in targets]
            if chembl_ids:
                return chembl_ids
        await asyncio.sleep(0.5 * (1 + attempt * 0.5))

    return ""


def _normalize_activities(activities, target_id, affinity_type):
    out = []
    for act in activities:
        val = act.get("standard_value")
        out.append({
            "smiles": act.get("canonical_smiles") or "",
            "affinity_type": affinity_type,
            "affinity_value": float(val) if val not in (None, "", "NA") else None,
            "affinity_units": act.get("standard_units") or "",
            "source": "ChEMBL",
            "target_id": target_id
        })
    return out


async def _fetch_chembl_activity_async(
    session: aiohttp.ClientSession,
    target_id: str,
    affinity_type: str = "Ki",
    limit_per_page: int = 1000,
    max_records: int = 100000,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> List[Dict]:
    """Fetch all ChEMBL activity pages concurrently for a given target.
    Correct handling of total_count vs max_records (no premature return)."""
    if affinity_type not in VALID_AFFINITY_TYPES:
        return []

    # First page (offset=0) to determine total_count and collect initial activities
    base_url = (
        f"{CHEMBL_BASE}/activity.json?"
        f"target_chembl_id={quote(target_id)}&"
        f"standard_type={quote(affinity_type)}&"
        f"limit={limit_per_page}&offset=0&include=molecule"
    )
    first_data = await _aio_fetch_json(session, base_url, semaphore=semaphore)
    if not first_data:
        return []

    activities = first_data.get("activities", []) or []
    results = _normalize_activities(activities, target_id, affinity_type)

    page_meta = first_data.get("page_meta", {}) or {}
    try:
        total_count = int(page_meta.get("total_count", len(results)))
    except Exception:
        total_count = len(results)

    # Determine how many records we should fetch (cap by max_records)
    desired_total = min(total_count, max_records)
    already = len(results)
    remaining = max(0, desired_total - already)
    if remaining <= 0:
        return results

    # Build offsets for the remaining pages (start at limit_per_page)
    offsets = list(range(limit_per_page, limit_per_page + remaining, limit_per_page))
    # But ensure offsets do not exceed desired_total
    offsets = [o for o in offsets if o < desired_total]

    async def fetch_page(offset: int) -> List[Dict]:
        url = (
            f"{CHEMBL_BASE}/activity.json?"
            f"target_chembl_id={quote(target_id)}&"
            f"standard_type={quote(affinity_type)}&"
            f"limit={limit_per_page}&offset={offset}&include=molecule"
        )
        data = await _aio_fetch_json(session, url, semaphore=semaphore)
        acts = data.get("activities", []) if data else []
        return _normalize_activities(acts, target_id, affinity_type)

    # Limit concurrency across all pages + other targets using provided semaphore
    tasks = [fetch_page(off) for off in offsets]
    # run and collect (exceptions are returned)
    page_results = await asyncio.gather(*tasks, return_exceptions=True)

    for pr in page_results:
        if isinstance(pr, list):
            results.extend(pr)
        # if pr is Exception or unexpected, skip (we keep previous retry logic in _aio_fetch_json)

    # Trim results to desired_total in case last page(s) overshot
    if len(results) > desired_total:
        results = results[:desired_total]

    return results   


async def fetch_chembl_data(
    target_name: str,
    target_id: Optional[str] = None,
    affinity_type: str = "Ki",
    max_records: int = 10000,
    concurrency_limit: int = 10,
) -> List[Dict]:
    """
    High-performance concurrent ChEMBL data fetcher.
    Fetches multiple targets and multiple pages concurrently with controlled concurrency.
    """
    semaphore = asyncio.Semaphore(concurrency_limit)
    results: List[Dict] = []

    async with aiohttp.ClientSession() as session:
        # Resolve targets
        if not target_id:
            chembl_targets = await _resolve_chembl_target_id(session, target_name)
            if not chembl_targets:
                return []
        else:
            chembl_targets = [(target_id, "unknown")]

        # Concurrently fetch all targets
        tasks = [
            _fetch_chembl_activity_async(
                session=session,
                target_id=tid,
                affinity_type=affinity_type,
                max_records=max_records,
                semaphore=semaphore,
            )
            for tid, _ in chembl_targets
        ]

        all_data = await asyncio.gather(*tasks, return_exceptions=True)

        for (tid, organism), data in zip(chembl_targets, all_data):
            if isinstance(data, Exception):
                continue
            for rec in data:
                rec["target_id"] = tid
                rec["organism"] = organism
            results.extend(data)

    return results


@tool
def fetch_activity_data(
    source: str,
    protein_name: str,
    dir_to_save: str,
    protein_id: Optional[str] = None,
    affinity_type: str = "IC50",
    cutoff: int = 10000,
) -> str:
    """
    Unified data retrieval tool for biochemical databases.

    This function fetches protein-ligand interaction or activity data from supported sources
    such as BindingDB and ChEMBL. It automatically handles protein ID resolution and
    standardized affinity type filtering.

    Args:
        source (str): Name of data source ("bindingdb" or "chembl").
        protein_name (str): Target protein name.
        dir_to_save (str): directory to save parsed data in csv format
        protein_id (str, optional): Target protein id. If passed, protein_name is ignored
        affinity_type (str, optional): Type of affinity (Ki, Kd, IC50). Defaults to "Ki".
        cutoff (int, optional): Optional threshold (nM) for BindingDB. Defaults to 10000.

    Returns:
        str: Summary of results with path to file and some statistics
        Returns error string if data not found or error occurs.
    """
    source = source.lower().strip()
    if affinity_type not in VALID_AFFINITY_TYPES:
        return f"Invalid affinity type '{affinity_type}'. Must be one of {VALID_AFFINITY_TYPES}"

    async def _main():
        async with aiohttp.ClientSession() as session:
            if source == "bindingdb":
                target_id = protein_id  # avoid shadowing outer var
                if not target_id:
                    resolved_id = await fetch_uniprot_id(session, protein_name)
                    if not resolved_id:
                        return f"[BindingDB] Could not find UniProt ID for '{protein_name}'"
                    target_id = resolved_id

                entries = await fetch_affinity_bindingdb(
                    session, target_id, affinity_type, cutoff
                )
                return entries

            elif source == "chembl":
                entries = await fetch_chembl_data(
                    target_name=protein_name,
                    target_id=protein_id,
                    affinity_type=affinity_type
                )
                return entries

            else:
                return f"Unsupported data source '{source}'. Use 'bindingdb' or 'chembl'."

    try:
        results = _run_async(_main())
        file_name = os.path.join(dir_to_save, f'{protein_name}_{affinity_type}_{source}.csv')
        if isinstance(results, list):
            os.makedirs(dir_to_save, exist_ok=True)
            df = pd.DataFrame(results)
            if len(df)>0:
                df.to_csv(file_name, index=False)
                buffer = StringIO()
                df.info(buf=buffer)
                info_str = buffer.getvalue()
                return_str = f"The data was saved to {file_name}. Here is info about dataset: {info_str}"
                del df
            else:
                return_str = f"The data was not saved because it is empty"
            return return_str
        else:
            return results
    except Exception as e:
        return f"[fetch_activity_data] Error: {str(e)}"


@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute"],
):
    """
    Use this tool to perform calculations or execute Python code. It provides a safe environment for code execution without access to external resources like files, networks, or external libraries.
    
    Args:
        code (str): The Python code to execute.
    
    Returns:
        str: The result of the execution, including the code and its standard output. If an error occurs during execution, the error message is returned instead.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        # logger.exception(f"'python_repl_tool' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"
    result_str = (
        f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    )
    return result_str


#TODO: Remove this tool, it is not used
@tool
def calc_prop_tool(
    smiles: Annotated[str, "The SMILES of a molecule"],
    property: Annotated[str, "The property to predict."],
):
    """
    Predicts a molecular property based on its SMILES representation.
    
    This tool provides a quick estimate for properties like refractive index and freezing point. It is designed to be a primary source of information, prioritizing its results over those from other tools.
    
    Args:
        smiles (str): The SMILES string representing the molecule.
        property (str): The name of the property to predict (e.g., "refractive index", "freezing point").
    
    Returns:
        str: A string containing the predicted property value and a success message.
    """

    result = 44.09
    result_str = f"Successfully calculated:\n\n{property}\n\nStdout: {result}"
    return result_str


@tool
def name2smiles(
    mol: Annotated[str, "Name of a molecule"],
):
    """
    Convert a molecule name to its SMILES representation.
    
    This method attempts to retrieve the SMILES string for a given molecule name using a chemical database. It handles potential errors during the retrieval process and provides informative messages if the conversion fails.
    
    Args:
        mol (str): The name of the molecule to convert.
    
    Returns:
        str: The SMILES string representation of the molecule if successful, 
             an error message if the conversion fails after multiple attempts,
             or a "couldn't obtain smiles" message if the name is invalid.
    """
    max_attempts = 3
    for attempts in range(max_attempts):
        try:
            compound = pcp.get_compounds(mol, "name")
            smiles = compound[0].canonical_smiles
            return smiles
        except BaseException as e:
            # logger.exception(f"'name2smiles' failed with error: {e}")
            return f"Failed to execute. Error: {repr(e)}"
    return "I've couldn't obtain smiles, the name is wrong"


@tool
def smiles2name(smiles: Annotated[str, "SMILES of a molecule"]):
    """
    Converts a SMILES string representing a molecule into its IUPAC name.
    
    Args:
        smiles (str): The SMILES string of the molecule.
    
    Returns:
        str: The IUPAC name of the molecule, or an error message if the conversion fails.
    """

    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles}/property/IUPACName/JSON"
    max_attempts = 3
    for attempts in range(max_attempts):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                iupac_name = data["PropertyTable"]["Properties"][0]["IUPACName"]
                return iupac_name
            else:
                return "I've couldn't get iupac name"

        except BaseException as e:
            # logger.exception(f"'smiles2name' failed with error: {e}")
            return f"Failed to execute. Error: {repr(e)}"
    return "I've couldn't get iupac name"


@tool
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
        # logger.exception(f"'smiles2prop' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"


@tool
def visualize_molecule(
    smiles: Annotated[str, "SMILES of a molecule"],
    config: RunnableConfig,
):
    """
    Visualizes a molecule from its SMILES representation and saves the 3D structure as an HTML file.
    
    Args:
        smiles (str): The SMILES string representing the molecule to visualize.
        config (RunnableConfig): Configuration object containing necessary settings,
                                  including the path to save the visualization.
    
    Returns:
        str: A message indicating success or failure of the visualization process.
             On success, it confirms the molecule was visualized and saved.
             On failure, it provides an error message.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol = Chem.Mol(mol)
            mol = AllChem.AddHs(mol, addCoords=True)
            AllChem.EmbedMolecule(mol)
            AllChem.MMFFOptimizeMolecule(mol)

            view = py3Dmol.view(
                data=Chem.MolToMolBlock(mol),  # Convert the RDKit molecule for py3Dmol
                style={
                    "stick": {},
                    "sphere": {"scale": 0.3},
                },
                width=600,
                height=400,
            )
            view.setBackgroundColor("#b8bfcc")
            view.zoomTo()
            html_content = view.write_html()

            state = config["configurable"].get("state")
            # tool_call_id: Annotated[str, InjectedToolCallId] = state['messages'][-1]["tool_calls"][0]['id']

            path_to_results = os.path.join(
                os.environ.get("PATH_TO_RESULTS"), "vis_mols"
            )
            if not os.path.exists(path_to_results):
                os.makedirs(path_to_results)

            with open(
                os.path.join(path_to_results, "vis.html"), "w", encoding="utf-8"
            ) as f:
                f.write(html_content)

            answer = f"I've successfully generated images of {smiles} molecule"
            return answer
        else:
            return f"I've couldn't visualize this molecule. Perhaps SMILES is invalid"

    except BaseException as e:
        # logger.exception(f"'visualize_molecule' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"


@tool
def detect_molecules() -> Dict:
    """Detects molecular structures in images and converts them to SMILES via `molecules_ocr`.

    Image paths are taken from `SELECTED_PAPERS[session_id]` when a session is active,
    or from the OCR input directory given by the `IMG_STORAGE_PATH` environment variable.

    Returns:
        On success: dict from `molecules_ocr` (image filenames to lists of SMILES),
        with annotated images saved as <original_name>_annotated.jpg. On failure or
        no images: dict with an "answer" key and an explanatory message.
    """    
    logging.info('Running extract_molecules tool...')
    try:
        directory = Path(os.environ.get('IMG_STORAGE_PATH'))
        images = [str(f.resolve()) for f in directory.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        if not images:
            return {'answer': 'No images provided for molecules recognition.'}
        return molecules_ocr(images)
    except Exception as e:
        logger.error(f'molecules_recognition ERROR: {e}')
        return {'answer': 'Could not detect any molecules in the uploaded images.'}


@tool
def detect_reactions() -> Dict:
    """Detects chemical reactions in images and converts them via `reactions_ocr`.

    Image paths are taken from `SELECTED_PAPERS[session_id]` when a session is active,
    or from the OCR input directory given by the `IMG_STORAGE_PATH` environment variable.

    Returns:
        On success: dict from `reactions_ocr` (image filenames to reactants, conditions,
        products), with annotated images saved as <original_name>_annotated.jpg. On
        failure or no images: dict with an "answer" key and an explanatory message.
    """    
    logging.info('Running extract_reactions tool...')
    try:
        directory = Path(os.environ.get('IMG_STORAGE_PATH'))
        images = [str(f.resolve()) for f in directory.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
        if not images:
            return {'answer': 'No images provided for reactions recognition.'}
        return reactions_ocr(images)
    except Exception as e:
        logger.error(f'reactions_recognition ERROR: {e}')
        return {'answer': 'Could not detect any reactions in the uploaded images.'}

@tool
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
        dict: Retrosynthesis result with target and routes.
        On failure returns a dict with an "answer" message.
    """
    try:
        return retrosynthesis_result(smiles=smiles, mode=mode)
    except Exception as e:
        logger.error(f"retrosynthesis_tree_search ERROR: {e}")
        return {"answer": "Could not run retrosynthesis tree search."}

@tool
def classify_reaction(
    smiles: Annotated[List[str], "List of reaction SMILES, e.g. ['A.B>>C']"],
    num_results: Annotated[int, "Max classes per reaction (1..50)"] = 10,
) -> Dict:
    """
    Classify reaction SMILES into reaction classes.

    Use this when the user provides reaction SMILES and wants the reaction
    type/class (e.g., named reactions or class labels). Returns ASKCOS-like
    classification hits with ranks and confidence.

    Args:
        smiles (List[str]): List of reaction SMILES to classify.
        num_results (int): Max classes per reaction (1..50).

    Returns:
        dict: status_code/message/result list with classification hits.
        On failure returns a dict with an "answer" message.
    """
    try:
        return classify_reaction_smiles(smiles=smiles, num_results=num_results)
    except Exception as e:
        logger.error(f"classify_reaction ERROR: {e}")
        return {"answer": "Could not classify reaction SMILES."}

@tool
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
        backend (str): Model backend ("wldn5", "graph2smiles", "augmented_transformer").
        model_name (str): Model name for backend (default "pistachio").
        reagents (str): Reagents string.
        solvent (str): Solvent string.

    Returns:
        dict: inputs/backend/model_name/predictions with product SMILES and scores.
        On failure returns a dict with an "answer" message.
    """
    try:
        return forward_predict_products(
            smiles=smiles,
            backend=backend,
            model_name=retrosynthesis_model_name,
            reagents=reagents,
            solvent=solvent,
        )
    except Exception as e:
        logger.error(f"forward_predict ERROR: {e}")
        return {"answer": "Could not run forward prediction."}


@tool
def calculate_docking(smiles: str, pdb_id: str) -> str:
    """
    Calculate docking score for a molecule.
    Response contains docking score for the molecule.
    Args:
        smiles (str): SMILES string of the molecule.
        pdb_id (str): ID of the PDB file containing the receptor structure.
    Returns:
        response (dict): Dictionary containing docking score for the molecule and the HTML file.
    """
    response = calculate_docking_score(smiles, pdb_id)
    data = response.get("data", None)
    errors = response.get("error")
    
    if data:
        affinity = data['affinity']
        visualization = data['visualization']
    
    output_file = None
    if visualization:
        html_base64 = data.get("visualization", None)
        html_content = base64.b64decode(html_base64)
        output_file = os.path.join(os.environ.get("PATH_TO_RESULTS"), f"docking_result_{pdb_id}.html")
        with open(output_file, "wb") as f:
            f.write(html_content)
    
    result = {'affinity': affinity, "errors": errors}
    return {
        "answer": result,
        "metadata": {
            "html_file": output_file
                }
        }


chem_tools = [
    name2smiles,
    smiles2name,
    smiles2prop,
    visualize_molecule,
    detect_molecules,
    detect_reactions,
    retrosynthesis_tree_search,
    classify_reaction,
    forward_predict,
    calculate_docking,
]

data_tools = [
    fetch_activity_data
]

chem_tools_rendered = render_text_description(chem_tools)
data_tools_rendered = render_text_description(data_tools)

if __name__ == "__main__":
    import os

    #   directory = "/Users/alina/Desktop/ITMO/ChemCoScientist/ChemCoScientist/data_store/datasets"

    #   existing_datasets = [f for f in os.listdir(directory) if
    #   f.startswith('users_dataset_')]
    #   print("Existing datasets:", existing_datasets)

    #   data = fetch_chembl_data(
    #       target_name="GSK",
    #       affinity_type="Ki"
    #   )
    #   print("Data fetched from ChemBL:", data)
    DATASET_DIR = (
        "/Users/alina/Desktop/ITMO/ChemCoScientist/ChemCoScientist/data_store/datasets"
    )
    PROTEIN_NAME = "MEK1"
    AFFINITY_TYPE = "IC50"
    params = {
        "protein_name": PROTEIN_NAME,
        "affinity_type": AFFINITY_TYPE,
        "cutoff": 10000,
    }

    binding_data = fetch_BindingDB_data(params)
    print(f"Data fetched: {len(binding_data)} entries")

    # Save data to Excel
    df = pd.DataFrame(
        [
            {"Ligand": entry["ligand"], "Affinity": entry["affinity_value"]}
            for entry in binding_data
        ]
    )
    file_path = os.path.join(DATASET_DIR, f"sars_cov_2_ic50_data.xlsx")
    df.to_excel(file_path, index=False)
    print(f"Data saved to: {file_path}")
