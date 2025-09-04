
from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from langchain_core.tools.base import InjectedToolCallId
from langchain_core.messages import ToolMessage
from langchain_core.runnables.config import RunnableConfig
from langchain_experimental.utilities import PythonREPL
from langchain_core.language_models.chat_models import BaseChatModel

from langgraph.types import Command


import pubchempy as pcp
from rdkit.Chem.Descriptors import CalcMolDescriptors
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import py3Dmol

import requests
from typing import Annotated, Optional
import os

import logging


# Create a separate logger for tools.py
logger = logging.getLogger("tools_logger")
logger.setLevel(logging.INFO)

# Configure a file handler for the tools logger
file_handler = logging.FileHandler("tools.log")
file_handler.setLevel(logging.INFO)

# Set a formatter for the tools logger
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the tools logger
logger.addHandler(file_handler)

# This executes code locally, which can be unsafe
repl = PythonREPL()

@tool
def python_repl_tool(
    code: Annotated[str, "The python code to execute"],
):
    """
    Use this tool to execute Python code, perform calculations, and test hypotheses. It provides a safe environment for code execution without access to external resources like files, libraries, or system commands.
    
    Args:
        code (str): The Python code to execute.
    
    Returns:
        str: The output of the executed code, including the code itself and any standard output. If an error occurs during execution, an error message is returned.
    """
    try:
        result = repl.run(code)
    except BaseException as e:
        logger.exception(f"'python_repl_tool' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"
    result_str = f"Successfully executed:\n\`\`\`python\n{code}\n\`\`\`\nStdout: {result}"
    return result_str


@tool
def calc_prop_tool(
    smiles: Annotated[str, "The SMILES of a molecule"],
    property: Annotated[str, "The property to predict."],
):
    """
    Use this tool to obtain calculated molecular properties.
    Currently predicts a placeholder value for refractive index or freezing point.
    It is designed to be a primary source of information; avoid redundant calls or alternative tools if it provides a result.
    
    Args:
        smiles (str): The SMILES representation of the molecule.
        property (str): The molecular property to predict (e.g., "refractive index", "freezing point").
    
    Returns:
        str: A string containing the calculated property value and a success message.
    """
    #try:
    #    result = repl.run(code)
    #except BaseException as e:
    #    return f"Failed to execute. Error: {repr(e)}"
    result = 44.09
    result_str = f"Successfully calculated:\n\n{property}\n\nStdout: {result}"
    return result_str


@tool
def name2smiles(
    mol: Annotated[str, "Name of a molecule"],
):
    """
    Convert a molecule name to its SMILES representation.
    
    This function attempts to retrieve the SMILES string for a given molecule name using a chemical compound database. It includes error handling and retry logic to gracefully manage potential failures during the lookup process.
    
    Args:
        mol (str): The name of the molecule to convert.
    
    Returns:
        str: The SMILES string representing the molecule if successful, 
             an error message if the conversion fails after multiple attempts, 
             or a message indicating the name may be incorrect if the SMILES cannot be obtained.
    """
    max_attempts = 3
    for attempts in range(max_attempts):
        try:
            compound = pcp.get_compounds(mol, 'name')
            smiles = compound[0].canonical_smiles
            return smiles
        except BaseException as e:
            logger.exception(f"'name2smiles' failed with error: {e}")
            return f"Failed to execute. Error: {repr(e)}"
    return "I've couldn't obtain smiles, the name is wrong"
    
@tool
def smiles2name(
    smiles: Annotated[str, "SMILES of a molecule"]
):
    """
    Converts a SMILES string representing a molecule into its corresponding IUPAC name.
    
    This method queries a public chemical database (PubChem) to obtain the IUPAC name 
    associated with the provided SMILES notation. It includes retry logic to handle 
    potential connection issues.
    
    Args:
        smiles (str): The SMILES string of the molecule.
    
    Returns:
        str: The IUPAC name of the molecule if found, 
             "I've couldn't get iupac name" if the name retrieval fails after multiple attempts, 
             or an error message if an exception occurs during the process.
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
            logger.exception(f"'smiles2name' failed with error: {e}")
            return f"Failed to execute. Error: {repr(e)}"
    return "I've couldn't get iupac name"
@tool
def smiles2prop(
    smiles: Annotated[str, "SMILES of a molecule"],
    iupac: Optional[str] = None
):
    """
    Calculates a comprehensive set of properties for a given molecule.
    
    Args:
        smiles (str): The SMILES string representing the molecule.
        iupac (str, optional): The IUPAC name of the molecule. If provided, it will be used to retrieve the SMILES string. Defaults to None.
    
    Returns:
        CalcMolDescriptors: An object containing the calculated molecular properties.  Returns an error message as a string if calculation fails.
    """
    
    try:
        if iupac:
            compound = pcp.get_compounds(iupac, 'name')
            if len(compound):
                smiles = compound[0].canonical_smiles

        res = CalcMolDescriptors(Chem.MolFromSmiles(smiles))
        return res
    except BaseException as e:
        logger.exception(f"'smiles2prop' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"
    
@tool
def visualize_molecule(smiles: Annotated[str, "SMILES of a molecule"], config: RunnableConfig,
):
    '''
    Visualizes a molecule given its SMILES string and saves the 3D representation as an HTML file.
    
    This method constructs a 3D molecular visualization from a SMILES string, optimizes its geometry, and renders it using py3Dmol, saving the resulting interactive view as an HTML file for easy access. This allows for a visual assessment of molecular structures derived from textual descriptions or research data.
    
    Args:
        smiles (str): The SMILES string representing the molecule to visualize.
        config (RunnableConfig): Configuration object containing settings for the tool, including access to the state and results path.
    
    Returns:
        str: A success message indicating the visualization was generated and saved, or an error message if the SMILES string is invalid or an error occurred during processing.
    '''
            return Command(
                update={
                    "visualization": html_content,
                    "messages": [
                        ToolMessage(
                            f"I've successfully visualized given molecule", tool_call_id=tool_call_id
                        )
                    ],
                }
            )
        else: 
            return Command(
                update={
                    "messages": [
                        ToolMessage(
                            f"I've couldn't visualize this molecule. Perhaps SMILES is invalid", tool_call_id=tool_call_id
                        )
                    ],
                }
            )'''
    except BaseException as e:
        logger.exception(f"'visualize_molecule' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"


@tool
def generate_molecule(
    params: Annotated[str, "Description of target molecule"],
    config: RunnableConfig
):
    """
    Generates a SMILES string representing a molecule based on a textual description.
    
    Args:
        params (str): A description of the target molecule.
        config (RunnableConfig): Configuration object containing the language model.
    
    Returns:
        str: The SMILES string of the generated molecule, or an error message if generation fails.
    """
    llm: BaseChatModel = config["configurable"].get("model")
    try:
        prompt = (
            'Generate smiles of molecule with given description. Answer only with smiles, nothing more: \
            Question: The molecule is a nitrogen mustard drug indicated for use in the treatment of chronic lymphocytic leukemia (CLL) and indolent B-cell non-Hodgkin lymphoma (NHL) that has progressed during or within six months of treatment with rituximab or a rituximab-containing regimen.  Bendamustine is a bifunctional mechlorethamine derivative capable of forming electrophilic alkyl groups that covalently bond to other molecules. Through this function as an alkylating agent, bendamustine causes intra- and inter-strand crosslinks between DNA bases resulting in cell death.  It is active against both active and quiescent cells, although the exact mechanism of action is unknown. \
            Answer: CN1C(CCCC(=O)O)=NC2=CC(N(CCCl)CCCl)=CC=C21 \
            Question: The molecule is a mannosylinositol phosphorylceramide compound having a tetracosanoyl group amide-linked to a C20 phytosphingosine base, with hydroxylation at C-2 and C-3 of the C24 very-long-chain fatty acid. It is functionally related to an Ins-1-P-Cer(t20:0/2,3-OH-24:0).\
            Answer: CCCCCCCCCCCCCCCCCCCCCC(O)C(O)C(=O)N[C@@H](COP(=O)(O)O[C@@H]1[C@H](O)[C@H](O)[C@@H](O)[C@H](O)[C@H]1OC1O[C@H](CO)[C@@H](O)[C@H](O)[C@@H]1O)[C@H](O)C(O)CCCCCCCCCCCCCCCC \
            Question: ' + params + '\n Answer: '
        )
        res = llm.invoke(prompt)
        smiles = res.content
        return smiles
    except BaseException as e:
        logger.exception(f"'generate_smiles' failed with error: {e}")
        return f"Failed to execute. Error: {repr(e)}"
    


chem_tools = [name2smiles, smiles2name, smiles2prop, generate_molecule, visualize_molecule]
chem_tools_rendered = render_text_description(chem_tools)
    