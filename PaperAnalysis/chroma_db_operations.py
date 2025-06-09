import base64
import chromadb
import os
import uuid

from chromadb.api.models import Collection
from chromadb.api.types import EmbeddingFunction, Documents
from chromadb.utils.data_loaders import ImageLoader
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage
from protollm.connectors import create_llm_connector

# client = chromadb.Client()
client = chromadb.PersistentClient(path='./chromadb')
DATA_LOADER = ImageLoader()

image_decs_prompt = """Describe an image from a chemistry research paper by following these guidelines:
1. Image Type : Specify what is depicted (e.g., reaction scheme, data graphs, microscopy image, spectrum, 3D molecular model, etc.).
2. Chemical Components : List ALL substances, compounds, catalysts, or solvents, including chemical formulas and
concentrations, shown in the image (if available).
3. Experimental Conditions : Note temperature, pressure, pH, reaction time, or other relevant parameters (if available).
4. Image Data :
    - For graphs: Axes labels, value ranges, trends (e.g., linear correlation, peaks).
    - For spectra: Spectrum type (IR, UV-Vis, mass spectrometry), key signals, and their interpretation.
    - For reaction schemes: Reaction steps, intermediates, yields, or side products.
5. Key Terms : Extract 5–10 keywords/terminology for semantic search (e.g., oxidative fluorination , microporous materials , thermogravimetric analysis ) if possible.

Use precise scientific language. Focus on objective details visible in the image to maximize searchability in a
vector database. Do not add Markdown elements to the result."""


def image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        base64_string = base64.b64encode(image_file.read()).decode('utf-8')
    return base64_string


def image_to_text(image_path: str) -> str:
    url = 'https://api.vsegpt.ru/v1;vis-google/gemini-2.0-flash-001'
    sys_prompt = "This is an image from a scientific paper in chemistry. " \
                 "Write a short but succinct description of the image that reflects its essence." \
                 "Be as concise as possible. " \
                 "Only use data from image, do NOT make anything up."
    model = create_llm_connector(url, temperature=0.015, top_p=0.95)
    messages = [HumanMessage(
        content=[
            {"type": "text", "text": sys_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(image_path)}"},
            },
        ],
    )]
    res = model.invoke(messages)
    return res.content


def get_or_create_chroma_collection(collection: str,
                                    embedding_function: EmbeddingFunction[Documents]) -> Collection:
    return client.get_or_create_collection(
        name=collection,
        embedding_function=embedding_function,
        data_loader=DATA_LOADER
    )


def store_text_chunks_in_chromadb(collection: Collection,
                                  content: list,
                                  paper_name: str):
    # Upload text
    for text_chunk in content:
        collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text_chunk.page_content],
            metadatas=[{"type": "text", **text_chunk.metadata}]
        )


def store_images_in_chromadb_txt_format(collection: Collection,
                                        image_dir: str,
                                        paper_name: str) -> None:

    # Upload images
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)
            
            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[image_to_text(img_path)],
                metadatas=[{"type": "image", "source": paper_name, "image_path": img_path}]
            )



def store_images_in_chromadb_mm_format(collection: Collection,
                                       image_dir: str,
                                       paper_name: str) -> None:

    # Upload images and tables
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(image_dir, filename)

            collection.add(
                ids=[str(uuid.uuid4())],
                documents=[img_path],
                metadatas=[{"type": "image", "source": paper_name, "image_path": img_path}]
            )


def query_chromadb(collection: Collection,
                   query_text: str,
                   metadata_filter: dict = None,
                   chunk_num: int = 3) -> dict:
    return collection.query(
        query_texts=[query_text],
        n_results=chunk_num,
        where=metadata_filter,
        include=["documents", "metadatas", "distances"]
    )
