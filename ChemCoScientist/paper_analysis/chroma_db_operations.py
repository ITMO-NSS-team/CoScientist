import base64
import logging
import os
import uuid
from pathlib import Path

import chromadb
import numpy as np
from chromadb import Documents, EmbeddingFunction, Embeddings
from chromadb.api.models import Collection
from chromadb.utils.data_loaders import ImageLoader
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage
from protollm.connectors import create_llm_connector
from pydantic import BaseModel, Field
import requests

from ChemCoScientist.paper_analysis.prompts import summarisation_prompt
from ChemCoScientist.paper_analysis.settings import allowed_providers
from ChemCoScientist.paper_analysis.settings import settings as default_settings
from CoScientist.paper_parser.s3_connection import s3_service
from CoScientist.paper_parser.parse_and_split import (
    clean_up_html,
    html_chunking,
    clean_up_after_processing
)
from definitions import CONFIG_PATH, ROOT_DIR

load_dotenv(CONFIG_PATH)
DATA_LOADER = ImageLoader()
IMAGES_PATH = os.path.join(ROOT_DIR, os.environ["PARSE_RESULTS_PATH"])
CHROMA_DB_PATH = os.path.join(ROOT_DIR, os.environ["CHROMA_STORAGE_PATH"])
VISION_LLM_URL = os.environ["VISION_LLM_URL"]
SUMMARY_LLM_URL = os.environ["SUMMARY_LLM_URL"]
PAPERS_PATH = os.path.join(ROOT_DIR, os.environ["PAPERS_STORAGE_PATH"])
USE_S3 = os.getenv("USE_S3") == "True"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExpandedSummary(BaseModel):
    """
    Expanded version of paper's summary.
    """

    paper_summary: str = Field(description="Summary of the paper.")
    paper_title: str = Field(
        description="Title of the paper. If the title is not explicitly specified, use the default value - 'NO TITLE'"
    )
    publication_year: int = Field(
        description=(
            "Year of publication of the paper. If the publication year is not explicitly specified, use the default"
            " value - 9999."
        )
    )


class CustomEmbeddingFunction(EmbeddingFunction):
    """
    Creates embeddings from text using a custom function.
    
        This class provides a way to generate embeddings for text data using a 
        user-defined function. It takes the embedding function as a constructor argument.
    
        Attributes:
        - embedding_function: The function used to generate embeddings.
    """

    def __call__(self, texts: Documents) -> Embeddings:
        """
        Retrieves embeddings for a list of documents using a ChromaDBPaperStore.
        
        Args:
            self: The instance of the class.
            texts: The documents to retrieve embeddings for (list of strings).
        
        Returns:
            Embeddings: The embeddings for the input documents (list of lists of floats).
        
        This method transforms text into numerical vector representations (embeddings).
        These embeddings capture the semantic meaning of the documents, 
        allowing for efficient comparison and retrieval of relevant information.
        """
        embeddings = ChromaDBPaperStore.get_embeddings(texts)
        return embeddings


class ChromaClient:
    """
    A client for interacting with a Chroma database.
    
        This class provides methods to manage Chroma collections, including
        creating, querying, and deleting them. It abstracts the underlying
        ChromaDB client for easier use.
    
        Attributes:
        - client: The ChromaDB client instance.
        - collection_name: The name of the Chroma collection to use.
        - embedding_function: The embedding function used for vectorizing data.
    """

    def __init__(self):
        """
        Initializes the ChromaDB client.
        
        Connects to a ChromaDB instance to enable storage and retrieval of scientific paper data for question answering.
        
        Args:
            self: The instance of the class.
        
        Initializes the following class fields:
            client: The ChromaDB client object used for interacting with the database. 
                    It is initialized using the host, port, and reset settings 
                    from the default settings.
        
        Returns:
            None
        """
        # self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
        self.client = chromadb.HttpClient(
            host=default_settings.chroma_host,
            port=default_settings.chroma_port,
            settings=chromadb.Settings(allow_reset=default_settings.allow_reset),
        )

    def get_or_create_chroma_collection(
        self,
        collection: str,
        embedding_function: EmbeddingFunction[Documents] | None = None,
    ) -> Collection:
        """
        Gets or creates a Chroma collection.
        
        This method ensures a Chroma collection exists for storing and retrieving document data. It prioritizes retrieving an existing collection by name. If a collection with the given name doesn't exist, it creates a new one, configured with the specified embedding function and a default data loader. This enables efficient storage and search of scientific documents.
        
        Args:
            collection (str): The name of the Chroma collection to retrieve or create.
            embedding_function (EmbeddingFunction[Documents] | None, optional): An optional embedding function to use for the collection. If None, the default embedding function is used. Defaults to None.
        
        Returns:
            Collection: The Chroma collection.
        """
        return self.client.get_or_create_collection(
            name=collection,
            embedding_function=embedding_function,
            data_loader=DATA_LOADER,
        )

    @staticmethod
    def query_chromadb(
        collection: Collection,
        query_text: str,
        metadata_filter: dict = None,
        chunk_num: int = 3,
    ) -> dict:
        """
        Queries a ChromaDB collection to find relevant information based on a text query.
        
        Args:
            collection: The ChromaDB collection to query.
            query_text: The text query to perform.  This is the information the user is seeking.
            metadata_filter: Optional dictionary to filter results based on metadata.
            chunk_num: The number of results to return.  Determines how many of the most relevant documents will be returned.
        
        Returns:
            dict: A dictionary containing the query results, including:
                - 'documents': The text of the retrieved documents.
                - 'metadatas': The metadata associated with each document.
                - 'distances':  A measure of similarity between the query and each document.
        """
        return collection.query(
            query_texts=[query_text],
            n_results=chunk_num,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )
    
    def delete_collection(self, name: str):
        """
        Deletes a collection and its associated data from the database.
        
        Args:
            name: The name of the collection to delete.
        
        Returns:
            None
        """
        self.client.delete_collection(name)
        
    def show_collections(self):
        """
        Retrieves the names of all collections stored within the connected Chroma database.
        
        Args:
            self: The instance of the ChromaClient class.
        
        Returns:
            list: A list of strings, where each string is the name of a collection.
        """
        return self.client.list_collections()


class ChromaDBPaperStore:
    """
    A class for storing and searching paper data using ChromaDB.
    
        This class provides functionality to store text chunks and images (converted to text)
        from research papers in a ChromaDB vector database and perform similarity searches
        to retrieve relevant context. It leverages embeddings for semantic search and 
        optionally uses a reranker to improve search results.
    
        Attributes:
        - client: ChromaDB client instance.
        - collection_name: Name of the ChromaDB collection.
        - llm_name: Name of the Large Language Model.
        - embedding_function: Function for generating embeddings.
        - reranker: Reranker model for refining search results.
    """

    def __init__(self):
        """
        Initializes the multimodal retriever.
        
        This constructor establishes connections to the language model and Chroma database,
        and prepares collections for storing and retrieving information from summaries, texts,
        and images extracted from scientific papers. This setup enables efficient semantic
        search and question answering over the paper content.
        
        Args:
            self: The object instance.
        
        Initializes the following class fields:
            llm_url (str): The URL for the Large Language Model (LLM). Defaults to VISION_LLM_URL.
            client (ChromaClient): An instance of ChromaClient for interacting with the Chroma database.
            sum_collection_name (str): The name of the Chroma collection for summaries, read from environment variables.
            txt_collection_name (str): The name of the Chroma collection for texts, read from environment variables.
            img_collection_name (str): The name of the Chroma collection for images, read from environment variables.
            sum_chunk_num (int): The number of chunks for summaries. Defaults to 15.
            final_sum_chunk_num (int): The number of chunks for final summaries. Defaults to 3.
            txt_chunk_num (int): The number of chunks for texts. Defaults to 15.
            img_chunk_num (int): The number of chunks for images. Defaults to 2.
            sum_collection (ChromaCollection): The Chroma collection for summaries.
            txt_collection (ChromaCollection): The Chroma collection for texts.
            img_collection (ChromaCollection): The Chroma collection for images.
            workers (int): The number of worker threads to use. Defaults to 2.
        
        Returns:
            None
        """
        self.llm_url = VISION_LLM_URL

        self.client = ChromaClient()

        self.sum_collection_name = os.getenv("SUMMARIES_COLLECTION_NAME")
        self.txt_collection_name = os.getenv("TEXTS_COLLECTION_NAME")
        self.img_collection_name = os.getenv("IMAGES_COLLECTION_NAME")

        self.sum_chunk_num = 15
        self.final_sum_chunk_num = 3
        self.txt_chunk_num = 15
        self.img_chunk_num = 2

        self.sum_collection = self.client.get_or_create_chroma_collection(
            self.sum_collection_name, CustomEmbeddingFunction()
        )
        self.txt_collection = self.client.get_or_create_chroma_collection(
            self.txt_collection_name, CustomEmbeddingFunction()
        )
        self.img_collection = self.client.get_or_create_chroma_collection(
            self.img_collection_name, CustomEmbeddingFunction()
        )
        self.workers = 2

    @staticmethod
    def _image_to_base64(image_path: str) -> str:
        """
        Encodes an image file to a base64 string representation.
        
        This is needed to store images alongside paper data in a format suitable for vector database embedding and retrieval.  Converting the image to a base64 string allows it to be included as text within the document representation.
        
        Args:
            image_path (str): The path to the image file.
        
        Returns:
            str: The base64 encoded string representation of the image.
        """
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_string

    def _image_to_text(self, image_path: str) -> str:
        """
        Extracts a concise textual description of an image, focusing on its core content as it relates to scientific papers.
        
        Args:
            image_path (str): The path to the image file.
        
        Returns:
            str: A succinct text description of the image, geared towards understanding its role within a scientific context.
        """
        sys_prompt = (
            "This is an image from a scientific paper in chemistry. "
            "Write a short but succinct description of the image that reflects its essence."
            "Be as concise as possible. "
            "Only use data from image, do NOT make anything up."
        )
        model = create_llm_connector(
            self.llm_url, temperature=0.015, top_p=0.95, extra_body={"provider": {"only": allowed_providers}}
        )
        messages = [
            HumanMessage(
                content=[
                    {"type": "text", "text": sys_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{self._image_to_base64(image_path)}"
                        },
                    },
                ],
            )
        ]
        res = model.invoke(messages)
        return res.content

    def store_text_chunks_in_chromadb(self, content: list) -> None:
        """
        Stores text chunks into ChromaDB for efficient retrieval.
        
        Args:
            content: A list of text chunks to be stored.
        
        Returns:
            None
        
        The method transforms the input text chunks into numerical embeddings and stores them, along with the original text and associated metadata, into a ChromaDB collection. This allows for semantic search and retrieval of relevant information based on the content of the scientific papers. Unique IDs are assigned to each chunk to ensure proper indexing within the database.
        """
        embeddings = self.get_embeddings([text_chunk.page_content for text_chunk in content])
        self.txt_collection.add(
            ids=[str(uuid.uuid4()) for _ in range(len(content))],
            documents=[text_chunk.page_content for text_chunk in content],
            embeddings=embeddings,
            metadatas=[{"type": "text", **text_chunk.metadata} for text_chunk in content]
        )


    def store_images_in_chromadb_txt_format(self, image_dir: str, paper_name: str) -> None:
        """
        Stores images from a directory in ChromaDB in text format.
        
        Reads images from the given directory, generates text descriptions for each image, 
        and stores these descriptions along with associated metadata in a ChromaDB collection. 
        This allows for semantic search and retrieval of images based on their content.
        
        Args:
            image_dir (str): The path to the directory containing the images.
            paper_name (str): The name of the paper associated with the images, used for metadata.
        
        Returns:
            None
        """

        image_descriptions = []
        image_paths = []
        image_counter = 0
        valid_paths = list(url_mapping.keys())

        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(image_dir, filename)
                if img_path in valid_paths:
                    image_descriptions.append(self._image_to_text(img_path))
                    image_paths.append(url_mapping[img_path])
                    image_counter += 1

        embeddings = self.get_embeddings(image_descriptions)
        self.img_collection.add(
            ids=[str(uuid.uuid4()) for _ in range(image_counter)],
            documents=image_descriptions,
            embeddings=embeddings,
            metadatas=[
                {"type": "image", "source": paper_name, "image_path": img_path} for img_path in image_paths
            ]
        )


    def search_for_papers(self,
                          query: str,
                          chunks_num: int = None,
                          final_chunks_num: int = None) -> dict:
        """
        Retrieves relevant papers from the database based on a user's query.
        
        This method performs an initial search to identify potentially relevant papers 
        and then refines the results using a reranker to improve accuracy and relevance. 
        The final results are returned as a list of paper sources.
        
        Args:
            self: The instance of the ChromaDBPaperStore class.
            query (str): The search query string.
            chunks_num (int, optional): The number of chunks to retrieve initially. 
                                         Defaults to self.sum_chunk_num if None.
            final_chunks_num (int, optional): The number of final chunks to return. 
                                               Defaults to self.final_sum_chunk_num if None.
        
        Returns:
            dict: A dictionary containing a list of paper sources under the key 'answer'.
        """
        chunks_num = chunks_num if chunks_num else self.sum_chunk_num
        final_chunks_num = final_chunks_num if final_chunks_num else self.final_sum_chunk_num

        raw_docs = self.client.query_chromadb(self.sum_collection, query, chunk_num=chunks_num)
        docs = self.search_with_reranker(query, raw_docs, top_k=final_chunks_num)
        res = [doc[2]["source"] for doc in docs]
        return {'answer': res}

    def retrieve_context(
            self, query: str, relevant_papers: list = None
    ) -> tuple[list, dict]:
        """
        Retrieves relevant information from text and images associated with scientific papers based on a user query.
        
        This method aims to identify and extract the most pertinent data from a collection of scientific documents,
        facilitating quick access to key insights related to a specific research question. It first identifies relevant papers and then utilizes vector similarity search
        to find corresponding text and image chunks.
        
        Args:
            query (str): The search query used to identify relevant information.
            relevant_papers (list, optional): A list of pre-identified relevant papers. Defaults to None, in which case a search for relevant papers is initiated.
        
        Returns:
            tuple[list, dict]: A tuple containing the retrieved text and image context.
                - text_context (list): A list of text chunks deemed most relevant to the query.
                - image_context (dict): A dictionary containing image data associated with the query.
        """
        if not relevant_papers:
            relevant_papers = self.search_for_papers(query)

        raw_text_context = self.client.query_chromadb(
            self.txt_collection,
            query,
            {"source": {"$in": relevant_papers['answer']}},
            self.txt_chunk_num,
        )
        image_context = self.client.query_chromadb(
            self.img_collection,
            query,
            {"source": {"$in": relevant_papers['answer']}},
            self.img_chunk_num,
        )
        text_context = self.search_with_reranker(query, raw_text_context, top_k=5)
        return text_context, image_context
    
    def search_with_reranker(self, query: str, initial_results, top_k: int = 1) -> list[tuple[str, str, dict, float]]:
        """
        Refines initial search results by assessing the relevance of each document to the query.
        
        Args:
            self: The instance of the class.
            query: The search query string.
            initial_results: A dictionary containing the initial search results with keys 'documents', 'metadatas', and 'ids'. 
                             Each key maps to a list of corresponding values.
            top_k: The number of top results to return after reranking (default is 1).
        
        Returns:
            list[tuple[str, str, dict, float]]: A list of tuples, where each tuple contains the document ID,
                the document text, its metadata, and the reranking score. The list is sorted by the reranking score
                in descending order, and only the top_k results are included.
        """
        metadatas = initial_results['metadatas'][0]
        documents = initial_results["documents"][0]
        ids = initial_results["ids"][0]

        pairs = [[query, doc.replace("passage: ", "")] for doc in documents]
        
        rerank_scores = self.rerank(pairs)

        scored_docs = list(zip(ids, documents, metadatas, rerank_scores))
        scored_docs.sort(key=lambda x: x[3], reverse=True)

        return scored_docs[:top_k]

    def add_paper_summary_to_db(self, paper_name: str, parsed_paper: str, llm) -> None:
        """
        Adds a paper summary to the document collection for efficient information retrieval.
        
        This method processes a paper's content by generating an expanded summary using a language model.
        It then stores this summary, along with relevant metadata, in a vector database 
        to enable semantic search and question answering.
        
        Args:
            self: The instance of the ChromaDBPaperStore class.
            paper_name (str): The name of the scientific paper.
            parsed_paper (str): The text content of the parsed paper.
            llm: The language model used to generate the summary.
        
        Returns:
            None
        """
        expanded_summary: ExpandedSummary = llm.invoke([HumanMessage(content=summarisation_prompt + parsed_paper)])
        doc = Document(
            page_content=expanded_summary.paper_summary,
            metadata={
                "source": paper_name,
                "paper_title": expanded_summary.paper_title,
                "publication_year": expanded_summary.publication_year
            }
        )
        embedding = self.get_embeddings([doc.page_content])
        self.sum_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[doc.page_content],
            embeddings=embedding,
            metadatas=[{"type": "text", **doc.metadata}]
        )
        print(f"Summary loaded for: {paper_name}")

    def run_marker_pdf(self, p_path, out_path) -> None:
        """
        Executes a shell script to extract marker data from a PDF file.
        
        This method utilizes a shell script to parse markers from a given PDF, 
        converting the paper content into a structured format suitable for analysis 
        and storage. The number of processing workers is configurable to manage 
        performance based on system resources.
        
        Args:
            p_path (str): The path to the input PDF file.
            out_path (str): The path to save the output file containing marker data.
        
        Returns:
            None
        """
        try:
            os.system(
                " ".join(
                    [
                        "sh",
                        os.path.join(ROOT_DIR, "ChemCoScientist/paper_analysis/marker_parsing.sh"),
                        str(p_path),
                        str(out_path),
                        str(self.workers)
                    ]
                )
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise
    
    @staticmethod
    def get_embeddings(texts: list[str]) -> list[np.ndarray]:
        """
        Retrieves numerical representations of text for semantic understanding and comparison.
        
        Args:
            texts: A list of strings to be converted into embeddings.
        
        Returns:
            list[np.ndarray]: A list of embedding vectors, one corresponding to each input text.
        
        Raises:
            Exception: If the embedding service is unavailable or returns an error.
        """
        embedding_service_url = "http://" + default_settings.embedding_host + ":"\
                                + str(default_settings.embedding_port)\
                                + default_settings.embedding_endpoint
        try:
            response = requests.post(
                embedding_service_url,
                json=texts,
                timeout=300
            )
            response.raise_for_status()
            return response.json()["embeddings"]
        except Exception as e:
            logger.error(f"Embedding service error: {str(e)}")
            raise
    
    @staticmethod
    def rerank(pairs: list[list[str]]) -> list[float]:
        """
        Reranks a list of document pairs to determine their relevance.
        
        This method sends the pairs to a dedicated reranker service and retrieves scores 
        indicating the relevance of each pair. This helps to refine search results 
        and prioritize the most pertinent documents.
        
        Args:
            pairs: A list of pairs to rerank. Each pair is a list of strings representing documents.
        
        Returns:
            list[float]: A list of reranking scores, where each score corresponds to the input pair. 
                         Higher scores indicate greater relevance.
        
        Raises:
            Exception: If the reranker service is unavailable or returns an error.
        """
        reranker_service_url = "http://" + default_settings.reranker_host + ":" \
                                + str(default_settings.reranker_port) \
                                + default_settings.reranker_endpoint
        try:
            response = requests.post(
                reranker_service_url,
                json=pairs,
                timeout=300
            )
            response.raise_for_status()
            return response.json()["scores"]
        except Exception as e:
            logger.error(f"Reranker service error: {str(e)}")
            raise


process_local_store: ChromaDBPaperStore = None


def init_process():
    """
    Initializes a process-local storage for papers.
    
    This method creates an isolated storage instance for each process,
    allowing concurrent access and modification of paper data without interference.
    This ensures data consistency and avoids race conditions when multiple processes
    are analyzing papers simultaneously.
    
    Args:
        None
    
    Returns:
        None
    """
    global process_local_store
    process_local_store = ChromaDBPaperStore()


def process_single_document(folder_path: Path):
    """
    Processes a single document (paper) from a given folder path.
    
    This method extracts text from an HTML representation of a scientific paper, 
    cleans and structures the content, and then prepares it for efficient 
    knowledge retrieval by storing it in a vector database (ChromaDB). 
    This involves summarizing the paper, breaking it down into smaller chunks, 
    and indexing associated images as text.
    
    Args:
        folder_path (Path): The path to the folder containing the paper's HTML and PDF files.
    
    Returns:
        None
    """
    paper_name = folder_path.name.replace("_marker", "")
    paper_name_to_load = Path(paper_name + ".pdf")
    parsed_file_path = Path(folder_path, paper_name + ".html")
    with open(parsed_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    try:
        print(f"Starting post-processing paper: {paper_name}")
        if USE_S3:
            parsed_paper, mapping = clean_up_html(folder_path, paper_name, text, s3_service, paper_name)
        else:
            parsed_paper, mapping = clean_up_html(folder_path, paper_name, text)
        print(f"Finished post-processing paper: {paper_name}")
        documents = html_chunking(parsed_paper, paper_name)
        
        llm = create_llm_connector(SUMMARY_LLM_URL, extra_body={"provider": {"only": allowed_providers}})
        struct_llm = llm.with_structured_output(schema=ExpandedSummary)
        
        print(f"Starting loading paper: {paper_name}")
        process_local_store.add_paper_summary_to_db(str(paper_name_to_load), parsed_paper, struct_llm)
        process_local_store.store_text_chunks_in_chromadb(documents)
        process_local_store.store_images_in_chromadb_txt_format(str(folder_path), str(paper_name_to_load), mapping)
        print(f"Finished loading paper: {paper_name}")
        if USE_S3:
            clean_up_after_processing(folder_path)
    except Exception as e:
        print(f"Error in {paper_name}: {str(e)}")


def process_all_documents(base_dir: Path):
    """
    Processes documents within subdirectories of a given base directory in parallel.
    
    This method identifies subdirectories within the provided base directory and 
    processes each one concurrently using a thread pool. This allows for faster 
    processing of large collections of documents.
    
    Args:
      base_dir: The base directory containing the subdirectories, each representing a document.
    
    Returns:
      None
    """
    folders = [d for d in base_dir.iterdir() if d.is_dir()]
    with ThreadPoolExecutor(max_workers=2, initializer=init_process()) as pool:
        pool.map(process_single_document, [folder for folder in folders])


if __name__ == "__main__":

    p_path = PAPERS_PATH
    res_path = IMAGES_PATH
    
    p_store = ChromaDBPaperStore()
    p_store.run_marker_pdf(p_path, res_path)
    del p_store
    process_all_documents(Path(res_path))
    
    # p_store.client.delete_collection(name="test_paper_summaries_img2txt")
    # p_store.client.delete_collection(name="test_text_context_img2txt")
    # p_store.client.delete_collection(name="test_image_context")
    # print(p_store.client.show_collections())
    
    # print(ChromaDBPaperStore.get_embeddings(["hello", "world"]))
    # print(ChromaDBPaperStore.rerank([["hello", "world"], ["hello", "there"]]))
