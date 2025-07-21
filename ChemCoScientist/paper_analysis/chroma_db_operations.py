import base64
import os
import uuid
from pathlib import Path

import chromadb
from chromadb.api.models import Collection
from chromadb.api.types import Documents, EmbeddingFunction
from chromadb.utils import embedding_functions
from chromadb.utils.data_loaders import ImageLoader
from dotenv import load_dotenv
from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage
from protollm.connectors import create_llm_connector
from sentence_transformers import CrossEncoder

from ChemCoScientist.paper_analysis.prompts import summarisation_prompt
from CoScientist.paper_parser.parse_and_split import (
    clean_up_html,
    html_chunking,
    parse_with_marker,
    simple_conversion,
)
from definitions import CONFIG_PATH, ROOT_DIR

load_dotenv(CONFIG_PATH)
DATA_LOADER = ImageLoader()
IMAGES_PATH = os.path.join(ROOT_DIR, os.environ["PARSE_RESULTS_PATH"])
CHROMA_DB_PATH = os.path.join(ROOT_DIR, os.environ["CHROMA_STORAGE_PATH"])
VISION_LLM_URL = os.environ["VISION_LLM_URL"]
SUMMARY_LLM_URL = os.environ["SUMMARY_LLM_URL"]
PAPERS_PATH = os.path.join(ROOT_DIR, os.environ["PAPERS_STORAGE_PATH"])


class ChromaClient:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

    def get_or_create_chroma_collection(
        self,
        collection: str,
        embedding_function: EmbeddingFunction[Documents] | None = None,
    ) -> Collection:
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
        return collection.query(
            query_texts=[query_text],
            n_results=chunk_num,
            where=metadata_filter,
            include=["documents", "metadatas", "distances"],
        )


class ChromaDBPaperStore:
    def __init__(self):
        self.llm_url = VISION_LLM_URL
        self.summary_llm = SUMMARY_LLM_URL

        self.client = ChromaClient()

        self.sum_collection_name = "paper_summaries_img2txt"
        self.txt_collection_name = "text_context_img2txt"
        self.img_collection_name = "image_context"

        self.sum_chunk_num = 15
        self.txt_chunk_num = 15
        self.img_chunk_num = 2

        self.sum_embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-m3", normalize_embeddings=True
            )
        )

        self.rag_embedding_function = (
            embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="BAAI/bge-m3", normalize_embeddings=True
            )
        )

        self.sum_collection = self.client.get_or_create_chroma_collection(
            self.sum_collection_name, self.sum_embedding_function
        )
        self.txt_collection = self.client.get_or_create_chroma_collection(
            self.txt_collection_name, self.rag_embedding_function
        )
        self.img_collection = self.client.get_or_create_chroma_collection(
            self.img_collection_name, self.rag_embedding_function
        )

        self.reranker = CrossEncoder(
            "Alibaba-NLP/gte-multilingual-reranker-base",
            max_length=2048,
            trust_remote_code=True,
        )

    @staticmethod
    def _image_to_base64(image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            base64_string = base64.b64encode(image_file.read()).decode("utf-8")
        return base64_string

    def _image_to_text(self, image_path: str) -> str:
        sys_prompt = (
            "This is an image from a scientific paper in chemistry. "
            "Write a short but succinct description of the image that reflects its essence."
            "Be as concise as possible. "
            "Only use data from image, do NOT make anything up."
        )
        model = create_llm_connector(self.llm_url, temperature=0.015, top_p=0.95)
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

    def _store_text_chunks_in_chromadb(self, content: list) -> None:
        # Upload text
        for text_chunk in content:
            self.txt_collection.add(
                ids=[str(uuid.uuid4())],
                documents=[text_chunk.page_content],
                metadatas=[{"type": "text", **text_chunk.metadata}],
            )

    def _store_images_in_chromadb_txt_format(
        self, image_dir: str, paper_name: str
    ) -> None:

        # Upload images
        for filename in os.listdir(image_dir):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(image_dir, filename)

                self.img_collection.add(
                    ids=[str(uuid.uuid4())],
                    documents=[self._image_to_text(img_path)],
                    metadatas=[
                        {"type": "image", "source": paper_name, "image_path": img_path}
                    ],
                )

    def retrieve_context(
        self, query: str, relevant_papers: list = None
    ) -> tuple[list, dict]:
        if not relevant_papers:
            raw_docs = self.client.query_chromadb(
                self.sum_collection, query, chunk_num=self.sum_chunk_num
            )
            docs = self.search_with_reranker(query, raw_docs, top_k=3)
            relevant_papers = [doc[2]["source"] for doc in docs]

        raw_text_context = self.client.query_chromadb(
            self.txt_collection,
            query,
            {"source": {"$in": relevant_papers}},
            self.txt_chunk_num,
        )
        image_context = self.client.query_chromadb(
            self.img_collection,
            query,
            {"source": {"$in": relevant_papers}},
            self.img_chunk_num,
        )
        text_context = self.search_with_reranker(query, raw_text_context, top_k=5)
        return text_context, image_context

    def search_with_reranker(
        self, query: str, initial_results, top_k: int = 1
    ) -> list[tuple[str, str, dict, float]]:
        metadatas = initial_results["metadatas"][0]
        documents = initial_results["documents"][0]
        ids = initial_results["ids"][0]

        pairs = [[query, doc.replace("passage: ", "")] for doc in documents]

        rerank_scores = self.reranker.predict(pairs)

        scored_docs = list(zip(ids, documents, metadatas, rerank_scores))
        scored_docs.sort(key=lambda x: x[3], reverse=True)

        return scored_docs[:top_k]

    def _add_paper_summary_to_db(self, paper_path: str) -> None:
        llm = create_llm_connector(self.summary_llm)
        paper = os.path.basename(paper_path)
        conv_res = simple_conversion(paper_path)
        summary = llm.invoke(
            [HumanMessage(content=summarisation_prompt + conv_res)]
        ).content
        doc = Document(page_content=summary, metadata={"source": paper})
        self.sum_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[doc.page_content],
            metadatas=[{"type": "text", "source": paper}],
        )

    def upload_paper(self, paper_path: str) -> None:
        # Load summary for the paper
        self._add_paper_summary_to_db(paper_path)

        # Load text chunks
        f_name, dir_name = parse_with_marker(paper_name=paper_path)
        parsed_paper = clean_up_html(paper_name=f_name, doc_dir=dir_name)
        documents = html_chunking(html_string=parsed_paper, paper_name=f_name)
        self._store_text_chunks_in_chromadb(documents)

        # Load images
        parsed_images_path = os.path.join(
            IMAGES_PATH, Path(paper_path).stem + "_marker"
        )
        self._store_images_in_chromadb_txt_format(
            parsed_images_path, os.path.basename(paper_path)
        )

    def prepare_db(self, papers_path: str) -> None:
        for paper in os.listdir(papers_path):
            paper_path = os.path.join(papers_path, paper)
            self.upload_paper(paper_path)


if __name__ == "__main__":
    papers_path = PAPERS_PATH

    paper_store = ChromaDBPaperStore()
    paper_store.prepare_db(papers_path)
