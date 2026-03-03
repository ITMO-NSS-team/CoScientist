import os, json, base64
import logging
import pandas as pd
from dotenv import load_dotenv
import fitz
from io import BytesIO, StringIO
from pathlib import Path
from PIL import Image
from langchain_core.messages import SystemMessage
from protollm.connectors import create_llm_connector, get_allowed_providers
import uuid

logger = logging.getLogger(__name__)

from ChemCoScientist.chemical_utils.chemical_functions import extract_molecules_from_figure
from ChemCoScientist.paper_analysis.settings import allowed_providers
from ChemCoScientist.paper_analysis.prompts import extract_mol_properties_prompt
from ChemCoScientist.chemical_utils.ocr_pipeline import render_molecule_detections

from definitions import CONFIG_PATH, ROOT_DIR

load_dotenv(CONFIG_PATH)

DATASETS_LLM_URL = os.environ["DATASETS_LLM_URL"]

def convert_pdf_pages_to_images(path: str) -> list:
    """Converts PDF pages into fitz.Pixmap images."""
    doc = fitz.open(path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=500)
        images.append(pix)
    return images
        
def extract_smiles_from_images(images: list) -> list:
    """Uses OpenChemIE tool to extract molecular SMILES and ID from PDF screenshots."""
    results = []
    for img in images:
        pil_image = Image.open(BytesIO(img.tobytes("png")))
        buffered = BytesIO()
        pil_image.save(buffered, format="JPEG")
        image_file = buffered.getvalue()
        res = extract_molecules_from_figure(image=image_file)
        results.append(res.get("data", []))
    return results

def mols_to_csv(results):
    """"Formats OpenChemIE JSONs with molecular SMILES and IDs into pandas DataFrame."""
    df = pd.DataFrame()
    
    all_refs = []
    all_smiles = []
    for res in results:
        res = res[0]
        corefs = res["corefs"]
        mols_idxs = [i[0] for i in corefs]
        refs_idxs = [i[1] for i in corefs]
        smiles = [res["bboxes"][i]["smiles"] for i in mols_idxs]
        refs = [res["bboxes"][i]["text"] for i in refs_idxs]
        for i in range(len(corefs)):
            ref = ";".join(refs[i]) if refs[i] != [] else "Unknown ID"
            all_refs.append(ref)
            all_smiles.append(smiles[i])

    df["id"] = all_refs
    df["smiles"] = all_smiles

    return df

def extract_props(model_url: str, question: str, pdfs: list) -> dict:
    """
    Queries a language model with a question and a list of PDF documents to collect a dataset of
    molecules and their properties from scientific papers.
    """
    from ChemCoScientist.frontend.utils import update_activity

    if pdfs:
        update_activity(os.path.dirname(pdfs[0]))
    llm = create_llm_connector(model_url, extra_body={"provider": {"only": get_allowed_providers()}})

    content = []

    for paper_pdf in pdfs:
        with open(paper_pdf, "rb") as f:
            base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        paper_part = {
            "type": "file",
            "file": {
                "filename": paper_pdf,
                "file_data": f"data:application/pdf;base64,{base64_pdf}",
            },
        }
        content.append(paper_part)

    text_part = {"type": "text", "text": f"USER QUESTION: {question}"}
    content.append(text_part)
    from langchain_core.messages import HumanMessage

    messages = [
        SystemMessage(content=extract_mol_properties_prompt),
        HumanMessage(content=content)
    ]

    res = llm.invoke(messages)
    clean_text = res.content.split("```csv")[-1].split("```")[0].strip()
    df = pd.read_csv(StringIO(clean_text))
    return df

def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorders DataFrame columns into a consistent layout."""
    first_cols = ['id', 'smiles']
    last_cols = ['units', 'source']

    existing_first = [col for col in first_cols if col in df.columns]
    existing_last = [col for col in last_cols if col in df.columns]
    middle_cols = [col for col in df.columns if col not in existing_first + existing_last]

    new_order = existing_first + middle_cols + existing_last

    return df[new_order]


def extract_mols_prop_dataset(model_url: str, question: str, pdfs: list, session_id: str) -> (Path, Path):
    """
    Extracts a dataset with molecular SMILES and properties from PDF documents
    by calling the OpenChemIE tool and quering a language model. It returns the resulting dataset
    along with the original PDF pages, annotated with bounding boxes highlighting the detected
    molecular structures.
    
    Args:
        model_url (str): The URL of the language model to use for querying.
        question (str): The question to ask the language model.
        pdfs (list): A list of paths to PDF documents.
        session_id (str): Session ID.

    Returns:
        (str, str): Path to the file containing the extracted dataset,
                    path to the processed PDF pages with bounding boxes around extracted
                    molecular structures..
    """
    res_img_path = ROOT_DIR / os.environ.get("IMG_STORAGE_PATH") / "paper_images" / session_id / str(uuid.uuid4())
    res_img_path.mkdir(parents=True, exist_ok=True)
    final_dataset_path = res_img_path / "final_dataset.csv"
    all_datasets = []
    for pdf in pdfs:
        try:
            images = convert_pdf_pages_to_images(pdf)
            results = extract_smiles_from_images(images)
            mols_df = mols_to_csv(results)
            mols_df['id'] = mols_df['id'].astype(str)
            props_df = extract_props(model_url, question, [pdf])
            props_df['id'] = props_df['id'].astype(str)
            merged_df = pd.merge(props_df, mols_df, on="id", how="inner")
            merged_df["source"] = os.path.basename(pdf)
            all_datasets.append(merged_df)
        except Exception as e:
            logger.error(f"Error processing PDF {pdf}: {e}")
            logger.info("Skipping this PDF and continuing with others...")
            continue
    
    if not all_datasets:
        raise ValueError("No PDFs were successfully processed")

    combined_dataset = pd.concat(all_datasets, ignore_index=True)
    final_dataset = reorder_columns(combined_dataset)
    final_dataset.to_csv(final_dataset_path, sep="\t", index=False)
    return final_dataset_path, res_img_path


# if __name__ == "__main__":
#     pdfs = [r"C:\Users\computer\Documents\GitHub\CoScientist\ChemCoScientist\paper_analysis\papers\187152108785908820.pdf",
#             r"C:\Users\computer\Documents\GitHub\CoScientist\ChemCoScientist\paper_analysis\papers\ph16040516.pdf"]
#     pdfs = ['/Users/lizzy/Downloads/ph16040516.pdf']
#     question = "Collect a dataset of molecules and their MIC values against Staphylococcus aureus."
#     extract_mols_prop_dataset(DATASETS_LLM_URL, question, pdfs, '1')
