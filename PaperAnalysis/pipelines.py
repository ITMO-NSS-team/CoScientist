import os
import uuid
from pathlib import Path
from typing import Callable

from chromadb.api.models import Collection
from chromadb.api.types import EmbeddingFunction, Documents
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
import pandas as pd
from protollm.connectors import create_llm_connector

from answer_question import query_llm, summarisation_prompt
from chroma_db_operations import (get_or_create_chroma_collection, store_text_chunks_in_chromadb,
                                  store_images_in_chromadb_txt_format, store_images_in_chromadb_mm_format,
                                  query_chromadb)
from parse_and_split import html_chunking, clean_up_html, parse_with_marker, parse_and_clean, simple_conversion, loader


IMAGES_PATH = './parse_results'
load_dotenv('../config.env')


def process_questions(questions_path: str,
                      answers_path: str,
                      llm_url: str,
                      sum_collection: Collection,
                      txt_collection: Collection,
                      img_collection: Collection,
                      sum_chunk_num: int,
                      txt_chunk_num: int,
                      img_chunk_num: int) -> None:
    df = pd.read_csv(questions_path)
    for index, row in df.iterrows():
        question = row.question
        txt_data, img_data = retrieve_context(sum_collection, txt_collection, img_collection,
                                              sum_chunk_num, txt_chunk_num, img_chunk_num, "query: " + question,
                                              [row.paper_name])
        txt_context = ''
        img_paths = set()

        for chunk in txt_data['documents'][0]:
            txt_context += '\n\n' + chunk.replace("passage: ", "")
        for chunk_meta in txt_data['metadatas'][0]:
            img_paths.update(eval(chunk_meta["imgs_in_chunk"]))
        for img in img_data['metadatas'][0]:
            img_paths.add(img['image_path'])

        ans, metadata = query_llm(llm_url, question, txt_context, list(img_paths))

        df.at[index, 'chroma_text_context'] = txt_context
        df.at[index, 'chroma_images_context'] = str(img_paths)
        df.at[index, 'llm_answer'] = ans

    df.to_csv(answers_path, index=False)
    
    
def process_questions_on_summaries(
        questions_path: str,
        answers_path: str,
        chroma_collection: Collection
) -> None:
    df = pd.read_csv(questions_path)
    for index, row in df.iterrows():
        question = row.question
        context = query_chromadb(chroma_collection, question, chunk_num=3)
        df.at[index, "sources"] = context["metadatas"]
    df.to_csv(answers_path, index=False)


def load_data_to_chroma(papers_path: str,
                        chroma_collection: Collection,
                        store_function: Callable) -> None:
    for paper in os.listdir(papers_path):
        paper_path = os.path.join(papers_path, paper)
        parsed_images_path = os.path.join(IMAGES_PATH, Path(paper).stem)

        documents = loader(parse_and_clean(paper_path))
        store_function(chroma_collection, documents, parsed_images_path, paper)


def load_summary_to_chroma(model_url: str,
                           papers_path: str,
                           chroma_collection: Collection) -> None:
    llm = create_llm_connector(model_url)
    for paper in os.listdir(papers_path):
        paper_path = os.path.join(papers_path, paper)
        conv_res = simple_conversion(paper_path)
        summary = llm.invoke([HumanMessage(content=summarisation_prompt + conv_res)]).content
        doc = Document(
            page_content=summary,
            metadata={"source": paper}
        )
        chroma_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[doc.page_content],
            metadatas=[{"type": "text", "source": paper}]
        )


def add_paper_summary_to_db(model_url: str,
                            paper_path: str,
                            chroma_collection: Collection) -> None:
    llm = create_llm_connector(model_url)
    paper = os.path.basename(paper_path)
    conv_res = simple_conversion(paper_path)
    summary = llm.invoke([HumanMessage(content=summarisation_prompt + conv_res)]).content
    doc = Document(
        page_content=summary,
        metadata={"source": paper}
    )
    chroma_collection.add(
        ids=[str(uuid.uuid4())],
        documents=[doc.page_content],
        metadatas=[{"type": "text", "source": paper}]
    )


def prepare_db(sum_collection_name: str,
               txt_collection_name: str,
               img_collection_name: str,
               sum_ef: EmbeddingFunction[Documents],
               txt_ef: EmbeddingFunction[Documents],
               img_ef: EmbeddingFunction[Documents],
               process_img_func: Callable,
               papers_path: str,
               sum_model_url: str) -> tuple[Collection, Collection, Collection]:
    # Create collections
    summaries_collection = get_or_create_chroma_collection(sum_collection_name, sum_ef)
    text_collection = get_or_create_chroma_collection(txt_collection_name, txt_ef)
    image_collection = get_or_create_chroma_collection(img_collection_name, img_ef)

    for paper in os.listdir(papers_path):
        paper_path = os.path.join(papers_path, paper)

        # Load summary for the paper
        # add_paper_summary_to_db(sum_model_url, paper_path, summaries_collection)

        # Load text chunks
        f_name, dir_name = parse_with_marker(paper_name=paper_path)
        parsed_paper = clean_up_html(paper_name=f_name, doc_dir=dir_name)
        documents = html_chunking(html_string=parsed_paper, paper_name=f_name)
        store_text_chunks_in_chromadb(text_collection, documents, paper)

        # Load images
        parsed_images_path = os.path.join(IMAGES_PATH, Path(paper).stem + "_marker")
        process_img_func(image_collection, parsed_images_path, paper)

    return summaries_collection, text_collection, image_collection


def retrieve_context(sum_collection: Collection,
                     txt_collection: Collection,
                     img_collection: Collection,
                     sum_chunk_num: int,
                     txt_chunk_num: int,
                     img_chunk_num: int,
                     query: str,
                     relevant_papers: list = None) -> tuple[dict, dict]:
    # docs = query_chromadb(sum_collection, query, chunk_num=sum_chunk_num)
    # relevant_papers = [doc['source'] for doc in docs['metadatas'][0]]

    text_context = query_chromadb(txt_collection, query, {"source": {"$in": relevant_papers}}, txt_chunk_num)
    image_context = query_chromadb(img_collection, query, {"source": {"$in": relevant_papers}}, img_chunk_num)
    return text_context, image_context



def run_mm_rag():
    sum_collection_name = 'paper_summaries_mm'
    txt_collection_name = 'text_context_mm'
    img_collection_name = 'mm_image_context'
    sum_chunk_num = 1
    txt_chunk_num = 3
    img_chunk_num = 2

    mm_embedding_function = embedding_functions.OpenCLIPEmbeddingFunction()
    reg_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        # model_name="intfloat/multilingual-e5-large",
        model_name="intfloat/multilingual-e5-small",
        normalize_embeddings=True
    )

    llm_url = 'https://api.vsegpt.ru/v1;vis-google/gemini-2.0-flash-001'

    papers_path = './papers'
    questions_path = './questions/complex_questions.csv'
    answers_path = './questions/complex_mm_answers.csv'

    sum_col, txt_col, img_col = prepare_db(sum_collection_name, txt_collection_name, img_collection_name,
                                           reg_embedding_function, reg_embedding_function, mm_embedding_function,
                                           store_images_in_chromadb_mm_format, papers_path, llm_url)

    process_questions(questions_path, answers_path, llm_url,
                      sum_col, txt_col, img_col,
                      sum_chunk_num, txt_chunk_num, img_chunk_num)


def run_img2txt_rag():
    sum_collection_name = 'paper_summaries_img2txt'
    txt_collection_name = 'text_context_img2txt'
    img_collection_name = 'image_context'
    sum_chunk_num = 1
    txt_chunk_num = 3
    img_chunk_num = 2

    reg_embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large",
        normalize_embeddings=True
    )

    llm_url = 'https://api.vsegpt.ru/v1;vis-google/gemini-2.0-flash-001'

    papers_path = './papers'
    questions_path = './questions/complex_questions.csv'
    answers_path = './questions/complex_text_answers.csv'

    sum_col, txt_col, img_col = prepare_db(sum_collection_name, txt_collection_name, img_collection_name,
                                           reg_embedding_function, reg_embedding_function, reg_embedding_function,
                                           store_images_in_chromadb_txt_format, papers_path, llm_url)

    process_questions(questions_path, answers_path, llm_url,
                      sum_col, txt_col, img_col,
                      sum_chunk_num, txt_chunk_num, img_chunk_num)

    
def run_summary_rag():
    llm_url = 'https://api.vsegpt.ru/v1;google/gemini-2.0-flash-lite-001'
    collection_name = "paper_summaries"
    papers_path = './papers'
    questions_path = './questions/simple_questions_summaries.csv'
    answer_path = './questions/simple_answers_on_summaries.csv'
    
    embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large",
        normalize_embeddings=True
    )
    chroma_collection = get_or_create_chroma_collection(collection_name, embedding_function)
    load_summary_to_chroma(llm_url, papers_path, chroma_collection)
    process_questions_on_summaries(questions_path, answer_path, chroma_collection)
    

if __name__ == "__main__":
    # run_mm_rag()
    run_img2txt_rag()
    # run_summary_rag()
