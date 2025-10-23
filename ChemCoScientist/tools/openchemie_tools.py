from langchain_core.tools import tool
import requests
from typing import BinaryIO


BASE_URL = "http://localhost:8000"

@tool
def extract_reactions_from_pdf(file: BinaryIO):
    """
    Extract reactions information from a PDF file.
    Response contains list of reactions for each page of the PDF.
    Each reaction contains list of reactants, products and conditions.
    
    Args:
        file (BinaryIO): PDF file to extract reactions from.
    Returns:
        response (List[Dict]): List of reactions in pdf file for each page.
    """
    response = requests.post(f"{BASE_URL}/extract_reactions_from_pdf/", files={"file": file})
    return response.json()["response"]


@tool
def extract_reactions_from_figure(image: BinaryIO):
    """
    Extract reactions information from an image.
    
    Response contains list of reactions on the image.
    Each reaction contains list of reactants, products and conditions.
    Args:
        image (BinaryIO): Image to extract reactions from.
    Returns:
        response (List[Dict]): List of reactions on the image.
    """
    response = requests.post(f"{BASE_URL}/extract_reactions_from_figure/", files={"image": image})
    return response.json()["response"]  


@tool
def extract_molecules_from_pdf(file: BinaryIO):
    """
    Extract molecules information from a PDF file.
    Response contains list of molecules for each page of the PDF.
    Each molecule contains bbox and smiles.
    
    Args:
        file (BinaryIO): PDF file to extract molecules from.
    Returns:
        response (List[Dict]): List of molecules in pdf file for each page.
    """
    response = requests.post(f"{BASE_URL}/extract_molecules_from_pdf/", files={"file": file})
    return response.json()["response"]


@tool
def extract_molecules_from_figure(image: BinaryIO):
    """
    Extract molecules information from an image.
    Response contains list of molecules on the image.
    Each molecule contains bbox and smiles.
    
    Args:
        image (BinaryIO): Image to extract molecules from.
    Returns:
        response (List[Dict]): List of molecules on the image.
    """
    response = requests.post(f"{BASE_URL}/extract_molecules_from_figure/", files={"image": image})
    return response.json()["response"]


@tool
def convert_image_to_smiles(image: BinaryIO):
    """
    Convert an image to a smiles string.
    Response contains smiles string of the image.
    Args:
        image (BinaryIO): Image to convert to smiles.
    Returns:
        response (str): SMILES string of the image.
    """
    response = requests.post(f"{BASE_URL}/convert_image_to_smiles/", files={"image": image})
    return response.json()["response"]
