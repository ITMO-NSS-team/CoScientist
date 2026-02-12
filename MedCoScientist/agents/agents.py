import ast
import json
import os
import time
from typing import Annotated, List
import operator
import streamlit as st
from langchain_core.language_models import BaseChatModel
from langchain.agents import create_react_agent

from langgraph.types import Command
from langgraph.graph import END

from MedCoScientist.agents.agents_prompts import worker_prompt, argument_extraction_prompt
from MedCoScientist.tools import extract_pico_node, extract_keywords_node, query_pubmed_node
from MedCoScientist.tools.pubmed_tools import PubMedArticle, PubMedArticleEncoder


def get_all_files(directory: str):
    """
    Traverses a directory and its subdirectories to locate all files.
    
    Args:
        directory (str): The path to the directory to search.
    
    Returns:
        list: A list of strings, where each string represents the absolute path to a 
        file within the directory and its subdirectories.
    
    This method is used to identify all relevant data files within a specified 
    location, ensuring that all potential information sources are included for 
    further processing and analysis.
    """
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

def hypothesis_pico_agent(state: dict, config: dict) -> Command:
    """
    Decomposes a hypothesis into PICO elements.
    
    This method instantiates an agent using a specified LLM, configured with credentials and settings from the provided configuration. 
    It then utilizes this agent to process the task described in the input state and generates a response. The method is designed to automate 
    PICO extraction.
    
    All tools are client functions that can launch training of models (ML-models or transformer model) or call inference.
    
    Args:
        state (dict): A dictionary containing the task to be performed, accessible via the "task" key.
        config (dict): A dictionary containing configuration details.
    
    Returns:
        Command: A command object containing the agent's textual response, the updated task history (`past_steps`) including the current task and response,
            and a record of the agent call (`nodes_calls`) detailing the agent used and its input/output.
    """
    task = state["task"]
    plan = state["plan"]
    llm: BaseChatModel = config["configurable"]["model"]
    arg = (argument_extraction_prompt | llm).invoke({"prompt": task}).content
    pico: str = extract_pico_node(arg, llm)

    return Command(update={
        "past_steps": Annotated[set, operator.or_](set([(task, pico)]))
    })
    

def related_pubmed_literature_agent(state: dict, config: dict) -> Command:
    """
    Finds relevant papers in PubMed database.
    
    This method extracts keywords from the hypothesis. Uses them to query PubMed. Then returns PICO-decomposition of relevant papers.
    The method is designed to automate relevant literature search.
    
    Args:
        state (dict): A dictionary containing the task to be performed, accessible via the "task" key.
        config (dict): A dictionary containing configuration details.
    
    Returns:
        Command: A command object containing the agent's textual response, the updated task history (`past_steps`) including the current task and response,
            and a record of the agent call (`nodes_calls`) detailing the agent used and its input/output.
    """
    user_input = state['input']
    task = state["task"]
    plan = state["plan"]
    paper_key = 'found_papers_' + user_input

    llm: BaseChatModel = config["configurable"]["model"]
    # arg = (argument_extraction_prompt | llm).invoke({"prompt": task}).content
    # embedder_config = config["configurable"]["embedder"]
    # keywords = extract_keywords_node(arg, llm, embedder_config)
    keywords = 'reperfusion therapy, post-thrombotic syndrome'

    papers: List[PubMedArticle] = query_pubmed_node(keywords)
    for paper in papers:
        paper.abstract = extract_pico_node(paper.abstract, llm)

    result = json.dumps(papers, cls=PubMedArticleEncoder, indent=2, ensure_ascii=False)
    return Command(update={
        "past_steps": Annotated[set, operator.or_](set([(task, result)])),
        'metadata': Annotated[dict, operator.or_]({paper_key: result}) 
    })