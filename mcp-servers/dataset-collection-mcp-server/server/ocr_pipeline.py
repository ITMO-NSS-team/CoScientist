import fitz
import io
import os
import logging
from pathlib import Path
from PIL import Image, ImageDraw
from pprint import pprint

from chemical_functions import extract_molecules_from_figure, extract_reactions_from_figure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def draw_bboxes_on_image(
    image: bytes | fitz.Pixmap,
    bboxes: list[list] | dict[str, list[list]],
) -> bytes:
    """
    Draw bounding boxes of deceted molecules and reactions on the provided image.
    
    Args:
        image (bytes | fitz.Pixmap): Original image as bytes or Pixmap
        bboxes (List[List] | Dict[str, List[List]]): Bounding boxes [x1, y1, x2, y2]
    
    Returns:
        bytes: JPEG image with rectangles drawn.
    """
    if isinstance(image, fitz.Pixmap):
        image_bytes = image.tobytes("png")
    else:
        image_bytes = image

    img = Image.open(io.BytesIO(image_bytes))
    draw = ImageDraw.Draw(img)

    w, h = img.size
    for bbox in bboxes:
        x1 = bbox[0] * w
        y1 = bbox[1] * h
        x2 = bbox[2] * w
        y2 = bbox[3] * h
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
    output = io.BytesIO()
    img.save(output, format="JPEG", quality=95)
    return output.getvalue()


def molecules_ocr(images: list[str]) -> dict:
    """
    Extracts molecules from a list of image paths using OpenChemIE tools and
    saves annotated versions of each image with bounding boxes around detected
    molecular structures.

    Parameters
    ----------
    images : list[str]
        List of paths to input images.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping each image filename to a list of extracted SMILES
        strings derived from detected molecules in that image.

    Side Effects
    ------------
    - Saves an annotated image for each input image as <original_name>_annotated.jpg,
      containing bounding boxes around detected molecules.
    """
    
    result = dict()
    annotated_images = []
    
    for img_path in images:
        img_path = Path(img_path)
        img_bytes = img_path.read_bytes()
        
        openchemie_result = extract_molecules_from_figure(img_bytes)
        recognitions = openchemie_result.get("data", [])
        errors = openchemie_result.get("errors", None)
        entries = []
        if recognitions:  
            entries = recognitions[0].get("bboxes", [])
        bboxes, smiles = [], []
        
        for entry in entries:
            smi = entry.get("smiles")
            if smi:
                smiles.append(smi)
                bboxes.append(entry.get("bbox"))
        
        if bboxes:
            annotated_img = draw_bboxes_on_image(img_bytes, {"molecules": bboxes})
            os.makedirs(Path(os.environ.get('PROCESSED_IMG_STORAGE_PATH')), exist_ok=True)
            out_path = Path(os.environ.get('PROCESSED_IMG_STORAGE_PATH'), f"{img_path.stem}_annotated.jpg")
            out_path.write_bytes(annotated_img)
            annotated_images.append(out_path.as_posix())

        result[img_path.name] = dict()
        result[img_path.name].update({"smiles": smiles})
        result[img_path.name].update({"errors": errors})
    
    return {
        "answer": result,
        "metadata": {
            "annotated_images": annotated_images
                }
        }


def reactions_ocr(images: list[str]) -> dict:
    """
    Extracts reactions from a list of image paths using OpenChemIE tools
    and saves annotated versions of each image with bounding boxes of detected reaction elements.

    Parameters
    ----------
    images : list[str]
        List of paths to input images.

    Returns
    -------
    dict[str, list[str]]
        A dictionary mapping each image filename to a list of extracted reaction elements
        such as reactants, conditions and products.
    
    Side Effects
    ------------
    - Saves an annotated image for each input image as <original_name>_annotated.jpg
      containing bounding boxes around detected reaction elements.
    """
    result = dict()
    annotated_images = []

    for img_path in images:
        img_path = Path(img_path)
        img_bytes = img_path.read_bytes()
        result[img_path.name] = dict()
        
        openchemie_result = extract_reactions_from_figure(img_bytes)
        recognitions = openchemie_result.get("data", [])
        errors = openchemie_result.get("errors", None)
        
        reactions = []
        if recognitions:  
            reactions = recognitions[0].get("reactions", [])
        
        bboxes = {"reagents": [], "products": [], "conditions": []}
        for reaction_id, reaction in enumerate(reactions):
            result[img_path.name][f"reaction_{reaction_id}"] = {"reactants": [], "products": [], "conditions": []}
            for r in reaction.get("reactants", []):
                bboxes["reagents"].append(r["bbox"])
                try:
                    result[img_path.name][f"reaction_{reaction_id}"]["reactants"].append(r["smiles"])
                except:
                    result[img_path.name][f"reaction_{reaction_id}"]["reactants"].append(r["text"])

            for p in reaction.get("products", []):
                bboxes["products"].append(p["bbox"])
                try:
                    result[img_path.name][f"reaction_{reaction_id}"]["products"].append(p["smiles"])
                except:
                    result[img_path.name][f"reaction_{reaction_id}"]["products"].append(p["text"])

            for c in reaction.get("conditions", []):
                bboxes["conditions"].append(c["bbox"])
                try:
                    result[img_path.name][f"reaction_{reaction_id}"]["conditions"].append(c["smiles"])
                except:
                    if c["text"] != []:
                        result[img_path.name][f"reaction_{reaction_id}"]["conditions"].append(c["text"])
        
        if any(bboxes.values()):
            annotated_img = draw_bboxes_on_image(img_bytes, bboxes)
            os.makedirs(Path(os.environ.get('PROCESSED_IMG_STORAGE_PATH')), exist_ok=True)
            out_path = Path(os.environ.get('PROCESSED_IMG_STORAGE_PATH'), f"{img_path.stem}_annotated.jpg")
            out_path.write_bytes(annotated_img)
            annotated_images.append(out_path.as_posix())

    return {
        "answer": result,
        "metadata": {
            "annotated_images": annotated_images
        }
    }


def render_molecule_detections(images: list, bboxes_list: list, res_path: str) -> None:
    """
    Renders bounding boxes around molecular structures that were extracted by
    OpenChemIE tools and saves annotated versions of each image.

    Parameters
    ----------
    images : list
        List of images.

    bboxes_list: list
        Coordinates of boxes.

    res_path: str
        Path to resulting images.

    Returns
    -------
        None

    Side Effects
    ------------
    - Saves an annotated image for each input image as <original_name>_annotated.jpg,
      containing bounding boxes around detected molecules.
    """

    for i, img_bytes in enumerate(images):

        entries = bboxes_list[i][0].get('bboxes')

        if entries:
            bboxes = []

            for entry in entries:
                smi = entry.get("smiles")
                if smi:
                    bboxes.append(entry.get("bbox"))

            if bboxes:
                annotated_img = draw_bboxes_on_image(img_bytes, bboxes)
                os.makedirs(Path(res_path), exist_ok=True)
                out_path = Path(res_path, f"{i}_annotated.jpg")
                out_path.write_bytes(annotated_img)


if __name__ == "__main__":
    directory = Path(r"C:\Users\computer\Documents\GitHub\CoScientist\ChemCoScientist\data_store\reactions")
    image_paths = [p.resolve() for p in directory.iterdir() if p.is_file()]
    # pprint(molecules_ocr(image_paths))
    pprint(reactions_ocr(image_paths))
