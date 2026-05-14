import asyncio
import base64
import os
import time
import pikepdf

from langchain_core.messages import SystemMessage, HumanMessage
from protollm.connectors import create_llm_connector
from pydantic import BaseModel, Field, field_validator, model_validator
from pypdf import PdfReader, PdfWriter
from io import BytesIO

from CoScientist.paper_analysis.chroma_db_operations import ChromaDBPaperStore
from CoScientist.paper_analysis.prompts import sys_prompt, explore_my_papers_prompt, extract_query_filters_prompt
from CoScientist.paper_analysis.research_taxonomy import (
    DOMAIN_TO_SUBDOMAINS,
    ResearchDomain,
    get_sub_domains_for_domain,
)
from CoScientist.paper_analysis.settings import allowed_providers
from CoScientist.paper_parser.utils import convert_to_base64, prompt_func, load_image_as_binary
from CoScientist.chemical_utils.chemical_functions import *
from CoScientist.paper_analysis.domain_metadata import format_domain_metadata, add_domain_metadata_to_img_info

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

VISION_LLM_URL = os.getenv("LLM__VISION_URL")


class QueryFilters(BaseModel):
    """Metadata filters extracted from user question."""
    paper_authors: list[str] | None = Field(
        description="Author names mentioned in the question",
        default=None
    )
    publication_year_min: int | None = Field(
        description="Minimum publication year for filtering",
        default=None
    )
    publication_year_max: int | None = Field(
        description="Maximum publication year for filtering",
        default=None
    )
    publication_year_exact: int | None = Field(
        description="Exact publication year when specified",
        default=None
    )
    publication_source: str | None = Field(
        description="Journal or publication source name",
        default=None
    )
    research_domain: list[ResearchDomain] | None = Field(
        description="Broad research domains",
        default=None
    )
    research_sub_domain: list[str] | None = Field(
        description="Specific research sub-domains within the selected research domains"
        " Must match the selected research_domain based on this mapping:\n"
        f"{DOMAIN_TO_SUBDOMAINS}\n",
        default=None
    )

    @field_validator("paper_authors", "research_domain", "research_sub_domain", mode="before")
    def normalize_list_fields(cls, v):
        if v is None:
            return None
        if isinstance(v, list):
            return v
        return [v]

    @model_validator(mode="after")
    def validate_domain_sub_domain_pair(self):
        if not self.research_sub_domain:
            return self

        if not self.research_domain:
            raise ValueError("research_domain must be set when research_sub_domain is provided")

        allowed_sub_domains = []
        for domain in self.research_domain:
            allowed_sub_domains.extend(get_sub_domains_for_domain(domain))

        invalid_sub_domains = [sd for sd in self.research_sub_domain if sd not in allowed_sub_domains]
        if invalid_sub_domains:
            raise ValueError(
                "research_sub_domain must belong to one of the selected research_domain values"
            )
        return self


def extract_metadata_filters(question: str) -> QueryFilters:
    """
    Uses LLM to extract metadata filters from user question.
    
    Args:
        question: The user's question string
        
    Returns:
        QueryFilters: Structured filters including authors, years, source, domain, and sub-domain
    """
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            llm = create_llm_connector(
                VISION_LLM_URL,
                extra_body={"provider": {"only": allowed_providers}},
                temperature=0.1
            )
            
            struct_llm = llm.with_structured_output(schema=QueryFilters)
            
            prompt = extract_query_filters_prompt + f"\n\nUSER QUESTION: {question}"
            
            filters: QueryFilters = struct_llm.invoke([HumanMessage(content=prompt)])
            return filters
        except Exception as e:
            print(f"Error extracting metadata filters (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 1.5 ** attempt
                print(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"Failed to extract metadata filters for question: {question}")
                # Return empty QueryFilters on final error to allow pipeline to continue
                return QueryFilters()
    
    return QueryFilters()


def build_chroma_where_filter(filters: QueryFilters) -> dict | None:
    """
    Converts QueryFilters to ChromaDB where clause format.
    
    Args:
        filters: QueryFilters object with extracted metadata
        
    Returns:
        dict: ChromaDB where clause ready for collection.query(), or None if no filters
        
    Example output:
        {"paper_authors": {"$in": ["Smith"]}}
        {
            "$and": [
                {"paper_authors": {"$in": ["Smith"]}},
                {"publication_year": {"$gte": 2020}}
            ]
        }
    """
    conditions = []
    
    if filters.paper_authors is not None:
        conditions.append({"paper_authors": {"$in": filters.paper_authors}})
    
    if filters.publication_year_exact is not None:
        conditions.append({"publication_year": {"$eq": filters.publication_year_exact}})
    elif filters.publication_year_min is not None or filters.publication_year_max is not None:
        year_condition = {}
        if filters.publication_year_min is not None:
            year_condition["$gte"] = filters.publication_year_min
        if filters.publication_year_max is not None:
            year_condition["$lte"] = filters.publication_year_max
        if year_condition:
            conditions.append({"publication_year": year_condition})
    
    if filters.publication_source is not None:
        conditions.append({"publication_source": {"$eq": filters.publication_source}})
    
    if filters.research_domain is not None:
        conditions.append({"research_domain": {"$in": filters.research_domain}})
    
    if filters.research_sub_domain is not None:
        conditions.append({"research_sub_domain": {"$in": filters.research_sub_domain}})
    
    if not conditions:
        return None
    
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def query_llm(
    model_url: str,
    question: str,
    system_prompt: str,
    txt_context: str,
    img_paths: list[str]
) -> tuple:
    """
    Queries a Large Language Model (LLM) to answer questions using provided context.

    This method constructs a query incorporating both textual and visual information, then sends it to the specified
    LLM. This allows the LLM to leverage diverse data sources for a more informed response.

    Args:
        model_url (str): The URL of the LLM model to use for querying.
        question (str): The question to be answered by the LLM.
        txt_context (str): Textual information to provide context for the question.
        img_paths (list[str]): A list of file paths to images to be used as context.

    Returns:
        tuple: A tuple containing the LLM's response content (str) and a dictionary of response metadata (dict).
    """
    llm = create_llm_connector(model_url, extra_body={"provider": {"only": allowed_providers}}, temperature=0.05)

    class ResScheme(BaseModel):
        answer: str = Field(description="The answer to the query", default="")
        explanation: str = Field(description="The logical reasoning for the answer", default="")
        chunk_explanation: str = Field(description="The explanation why the chosen chunk/chunks are relevant to the answer", default="")
        img_explanation: str = Field(description="The explanation why the chosen image/images are relevant to the answer", default="")
        relevant_text: list[int] = Field(description="A list of integers representing the relevant text chunk numbers, numeration of chunks starts with 1", default=[])
        relevant_images: list[int] = Field(description="A list of integers representing the relevant image numbers, numeration of images starts with 1", default=[])

    structured_llm = llm.with_structured_output(schema=ResScheme)

    img_context = list(map(convert_to_base64, img_paths))
    messages = [
        SystemMessage(content=system_prompt),
        prompt_func(
            {
                "text": f"USER QUESTION: {question}\n\nCONTEXT: {txt_context}",
                "image": img_context,
            }
        ),
    ]

    for attempt in range(3):
        try:
            res = structured_llm.invoke(messages)
            content = {
                'answer': res.answer,
                'explanation': res.explanation,
                'chunk_explanation': res.chunk_explanation,
                'img_explanation': res.img_explanation,
                'relevant_text': res.relevant_text,
                'relevant_images': res.relevant_images
            }
            return content
        except Exception as e:
            last_error = e
            messages.append(
                    HumanMessage(
                        content="Previous response was invalid JSON. Respond with ONLY valid JSON."
                    )
                )
            continue
    
    raise RuntimeError(
        f"Failed to get valid structured response after 3 attempts. "
        f"Last error: {last_error}"
    ) from last_error



def simple_query_llm(
    model_url: str,
    question: str,
    system_prompt: str,
    pdfs: list,
    img_descriptions: str) -> dict:
    """
    Queries a language model with a question and a list of PDF documents to provide context for answering the question.

    Args:
        model_url (str): The URL of the language model to use for querying.
        question (str): The question to ask the language model.
        pdfs (list): A list of paths to PDF documents to provide as context.

    Returns:
        dict: A dictionary containing the answer from the language model. The dictionary has a single key, 'answer',
            which holds the answer string.
    """

    llm = create_llm_connector(model_url)

    content = []
    
    writer = PdfWriter()

    # Merge all PDFs
    for paper_pdf in pdfs:
        reader = PdfReader(paper_pdf)
        for page in reader.pages:
            writer.add_page(page)

    merged_buffer = BytesIO()
    writer.write(merged_buffer)
    merged_buffer.seek(0)
   
    # Linearize merged PDF
    clean_buffer = BytesIO()
    with pikepdf.open(merged_buffer) as pdf:
        pdf.save(clean_buffer, linearize=True)
    clean_buffer.seek(0)

    base64_pdf = base64.b64encode(clean_buffer.read()).decode("utf-8")
    paper_part = {
        "type": "file",
        "file": {
            "filename": "merged_papers.pdf",
            "file_data": f"data:application/pdf;base64,{base64_pdf}",
        },
    }
    content.append(paper_part)

    text_part = {"type": "text", "text": f"USER QUESTION: {question}\n\n{img_descriptions}"}
    content.append(text_part)
    from langchain_core.messages import HumanMessage

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ]
    
    for attempt in range(3):
        try:
            res = llm.invoke(messages)
            return {'answer': res.content}
        except Exception as e:
            print(f"LLM query error: {str(e)}. Retrying ({attempt + 1}/3)")
            time.sleep(1.2 ** attempt)
            
    return {'answer': 'LLM invocation failed after 3 attempts.'}


def process_question(
    question: str,
    system_prompt: str,
    store: ChromaDBPaperStore) -> dict:
    """
    Processes a question by retrieving relevant text and image context from scientific papers and querying a Large Language Model (LLM) to generate an answer.

    Args:
        question (str): The input question string.

    Returns:
        dict: A dictionary containing the answer and associated metadata:
            'answer' - the answer generated by the LLM based on the provided context;
            'metadata' - a dictionary containing:
                'text_context' - the concatenated text from relevant paper chunks, including metadata;
                'image_context' - the set of image paths identified as relevant to the question;
                'metadata' - Additional metadata returned by the LLM query.
    """
    meta_filter = extract_metadata_filters(question)
    meta_filter_chroma = build_chroma_where_filter(meta_filter)
    
    txt_data, img_data = store.retrieve_context(question, meta_filter=meta_filter_chroma)
    txt_context = ""
    relevant_txt_context = []
    img_paths = []

    # Combine text context
    for idx, chunk in enumerate(txt_data, start=1):
        txt_context += (
            f"{idx}. "
            + "\nChunk: "
            + chunk[1].replace("passage: ", "")
            + "\n\n"
        )
    
    # Combine images for context (from chunk text and fom DB)
    for chunk_meta in [chunk[2] for chunk in txt_data]:
        for img_path in eval(chunk_meta["imgs_in_chunk"]):

            img_info = {
                'path': img_path,
                'Source': chunk_meta['source'],
                'Paper': chunk_meta['title'],
                'Year': chunk_meta['year']
            }

            image_data = store.client.query_chromadb(
                    store.img_collection,
                    "",
                    {"image_path": img_path}
                )
            img_meta = image_data["metadatas"][0][0]
            img_info = add_domain_metadata_to_img_info(meta_filter.research_domain, img_meta, img_info)
            img_paths.append(img_info)
    
    for img_meta in img_data["metadatas"][0]:
        if img_meta['image_path'] not in [d['path'] for d in img_paths]:
            img_info = {
                'path': img_meta['image_path'],
                'Source': chunk_meta['source'],
                'Paper': chunk_meta['title'],
                'Year': chunk_meta['year']
            }
            image_data = store.client.query_chromadb(
                    store.img_collection,
                    "",
                    {"image_path": img_meta['image_path']}
                )
            img_meta = image_data["metadatas"][0][0]
            img_info = add_domain_metadata_to_img_info(meta_filter.research_domain, img_meta, img_info)
            img_paths.append(img_info)

    img_paths_list = set([d['path'] for d in img_paths])
    
    domain_metadata = format_domain_metadata(meta_filter.research_domain, img_paths)
    if domain_metadata != "":
        txt_context += f"Domain metadata\n{domain_metadata}\n\n"
    else:
        txt_context += "No domain metadata found for context."

    ans = query_llm(VISION_LLM_URL, question, system_prompt, txt_context, list(img_paths_list))

    # Separate relevant context
    relevant_txt_data = [txt_data[num - 1] for num in ans['relevant_text']]
    relevant_img_context = [img_paths[num - 1] for num in ans['relevant_images']]

    for idx, chunk in enumerate(relevant_txt_data, start=1):
        relevant_txt_context.append({
            'chunk': f"Chunk {idx}: \n"
                     + chunk[1].replace("passage: ", "")
                     + "\n\n",
            'Source': chunk[2]['source'],
            'Paper': chunk[2]['title'],
            'Year': chunk[2]['year'],
        })

    return {
        "chunk_metadata": txt_data,
        "img_metadata": img_data,
        "answer": ans['answer'],
        "explanation": ans['explanation'],
        "chunk_explanation": ans.get('chunk_explanation', ''),
        "img_explanation": ans.get('img_explanation', ''),
        "metadata": {
            "text_context": relevant_txt_context,
            "image_context": relevant_img_context,
        },
    }


if __name__ == "__main__":
    # file_paths = []  # Enter list of paths to images here
    #
    # images = list(map(convert_to_base64, file_paths))
    #
    # llm = create_llm_connector(VISION_LLM_URL)
    #
    # # question = ("Какая реакция идет протекает на 6 стадии Total Synthesis of (−)-Glionitrin A/B? Какие реагенты"
    # #             " участвовали в реакции и какой продукт получили? Какой получился выход?")
    # question = ("I need all the compounds that were used in the experiments. Obligatorily I need all results to be in"
    #             " the form of a table of 2 columns where in the first column were the names by IUPAC numberclature and"
    #             " in the second column in SMILES notation. Don't add it to this list of reaction products for me. Can"
    #             " you do that?")
    # context = ""
    #
    # messages = [
    #     SystemMessage(content=sys_prompt),
    #     prompt_func({"text": f"USER QUESTION: {question}\n\nCONTEXT: {context}", "image": images})
    # ]
    # # messages = [
    # #     SystemMessage(content="You're a useful assistant. You only ever reply in the form of valid JSON."),
    # #     prompt_func(
    # #         {
    # #             "text": "For the provided images, generate a detailed clear description. If there is a table in the"
    # #                     " image, parse it and return it in HTML format. If you see chemical compounds in the figures,"
    # #                     " output the names of the compounds according to IUPAC nomenclature.\n"
    # #                     " As a response, return ONLY JSON of the following form: {‘figure_1’:"
    # #                     " ‘figure_1_description’, ‘figure_2’: ‘figure_2_description’, ‘table_1’:"
    # #                     " ‘table_1_description’...}",
    # #             "image": images
    # #         }
    # #     )
    # # ]
    #
    # res = llm.invoke(messages)
    # print(res.content)
    # print(res.response_metadata)

    #######################################################

    paper_store = ChromaDBPaperStore()
    # question = 'What aliphatic hydroxy acids are present in the papers published in 2022? Give me their SMILES.'
    question = 'What components are involved in the synthesis of BASHY dyes, and what are the uses of these dyes?'
    # question = 'What IC50 values do weakly active and highly active Bruton\'s tyrosine kinase inhibitors have?'
    # question = 'How does the synthesis of Glionitrin A/B happen?'

    # res = simple_query_llm(VISION_LLM_URL, question, [paper])
    result = process_question(question, sys_prompt, paper_store)
    from pprint import pprint
    pprint(result)
