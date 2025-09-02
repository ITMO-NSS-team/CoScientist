import os

from definitions import CONFIG_PATH
from dotenv import load_dotenv

load_dotenv(CONFIG_PATH)

from protollm.connectors import create_llm_connector
from protollm.agents.universal_agents import web_search_node

from ChemCoScientist.agents.agents import (
    chemist_node,
    dataset_builder_agent,
    ml_dl_agent,
    nanoparticle_node,
    paper_analysis_agent,
)
from CoScientist.scientific_agents.agents import coder_agent
from ChemCoScientist.tools import chem_tools_rendered, nano_tools_rendered, tools_rendered, paper_analysis_tools_rendered


# description for agent WITHOUT langchain-tools
automl_agent_description = """
'ml_dl_agent' - trains generative models to create SMILES and predictive models to estimate properties. It also
stores ready-made models for inference and can prepare an existing dataset (be specific in your request).
Use this agent only when the user explicitly asks for model training, inference, or molecule generation. It is not
suited for literature questions.\n
"""
dataset_builder_agent_description = ("'dataset_builder_agent' - collects data from ChemBL and BindingDB. \
    It requires either a protein name or a specific database id to gather data from one or both sources. All data is saved \
    locally and can be cleaned: junk values removed, empty cells dropped, optional filtering applied.\n\
    Use this agent only when the user asks to collect or preprocess data from chemical databases, never for literature \
    queries or model training.\n")

coder_agent_description = (
    "'coder_agent' - writes simple scientific Python code using rdkit and other chemical libraries for calculations.\n"
    "Use this agent solely when the user requests code generation or numerical computations.\n"
)

paper_analysis_node_description = (
        "'paper_analysis_node' - retrieves and analyzes information from a database of chemical scientific papers."
    " Activate this agent when the user asks about articles or research findings. For such questions, first plan this"
    " agent, then follow with 'web_search' for additional internet information. Do not involve other agents unless the"
    " user explicitly requires them."
)
web_search_description = ("'web_search' - searches the internet to complement results from 'paper_analysis_node'. Use it right after"
    " 'paper_analysis_node' and avoid calling it alone or with unrelated agents.")

additional_agents_description = (
    automl_agent_description
    + dataset_builder_agent_description
    + coder_agent_description
    + paper_analysis_node_description
    + web_search_description
)

conf = {
    # maximum number of recursions
    "recursion_limit": 25,
    "configurable": {
        "user_id": "1",
        "visual_model": create_llm_connector(os.environ["VISION_LLM_URL"]),
        "img_path": "image.png",
        "llm": create_llm_connector(
            f"{os.environ['MAIN_LLM_URL']};{os.environ['MAIN_LLM_MODEL']}"
        ),
        "max_retries": 3,
        # list of scenario agents
        "scenario_agents": [
            "chemist_node",
            "nanoparticle_node",
            "ml_dl_agent",
            "dataset_builder_agent",
            "coder_agent",
            "paper_analysis_node",
            "web_search"
        ],
        # nodes for scenario agents
        "scenario_agent_funcs": {
            "chemist_node": chemist_node,
            "nanoparticle_node": nanoparticle_node,
            "ml_dl_agent": ml_dl_agent,
            "dataset_builder_agent": dataset_builder_agent,
            "coder_agent": coder_agent,
            "paper_analysis_node": paper_analysis_agent,
            "web_search": web_search_node
        },
        # descripton for agents tools - if using langchain @tool
        # or description of agent capabilities in free format
        "tools_for_agents": {
            "chemist_node": [chem_tools_rendered],
            "nanoparticle_node": [nano_tools_rendered],
            "dataset_builder_agent": [dataset_builder_agent_description],
            "coder_agent": [coder_agent_description],
            "ml_dl_agent": [automl_agent_description],
            "paper_analysis_node": [paper_analysis_tools_rendered],
            "web_search": [web_search_description],
        },
        # full descripton for agents tools
        "tools_descp": tools_rendered + additional_agents_description,
        # add a key with the agent node name if you need to pass something to it
        "additional_agents_info": {
            "dataset_builder_agent": {
                "model_name": os.environ["SCENARIO_LLM_MODEL"],
                "url": os.environ["SCENARIO_LLM_URL"],
                "api_key": os.environ["OPENAI_API_KEY"],
                "ds_dir": os.environ["DS_STORAGE_PATH"],
            },
            "coder_agent": {
                "model_name": os.environ["SCENARIO_LLM_MODEL"],
                "url": os.environ["SCENARIO_LLM_URL"],
                "api_key": os.environ["OPENAI_API_KEY"],
                "ds_dir": os.environ["ANOTHER_STORAGE_PATH"],
            },
            "ml_dl_agent": {
                "model_name": os.environ["SCENARIO_LLM_MODEL"],
                "url": os.environ["SCENARIO_LLM_URL"],
                "api_key": os.environ["OPENAI_API_KEY"],
                "ds_dir": os.environ["DS_STORAGE_PATH"],
            },
        },
        # These prompts will be added in ProtoLLM
        "prompts": {
            "supervisor": {
                "problem_statement": None,
                "problem_statement_continue": None,
                "rules": None,
                "additional_rules": None,
                "examples": None,
                "enhancemen_significance": None,
            },
            "planner": {
                "problem_statement": None,
                "rules": None,
                "desc_restrictions": None,
                "examples": None,
                "additional_hints": "Before starting model training, check data for garbage with 'dataset_builder_agent'. If the user already provides a dataset, go straight to 'ml_dl_agent' and skip 'dataset_builder_agent'.\n \
                    For questions about papers, articles, or research findings, plan exactly two steps: first 'paper_analysis_node', then 'web_search'. Do not schedule any other agents for such research tasks.\n \
                    Always choose the minimal set of agents necessary for the user's request.",
            },
            "chat": {
                "problem_statement": None,
                "additional_hints": """You are a chemical agent system. You can do the following:
                    - train generative models (generate SMILES molecules), train predictive models (predict properties)
                    - prepare a dataset for training
                    - download data from chemical databases: ChemBL, BindingDB
                    - perform calculations with chemical python libraries
                    - solve problems of nanomaterial synthesis
                    - analyze chemical articles
                     Choose only the agents relevant to the user's question. For literature queries, use 'paper_analysis_node'
                    followed by 'web_search' and avoid calling other agents. If user ask something like "What can you do" - make answer yourself!
                    """,
            },
            "summary": {
                "problem_statement": None,
                "rules": None,
                "additional_hints": "Never write full paths! Only file names. If 'paper_analysis_node' and 'web_search' were used, present the final answer as: paper_analysis: <paper_analysis_agent result>   web_search: <web_search_node result>.",
            },
            "replanner": {
                "problem_statement": None,
                "rules": None,
                "examples": None,
                "additional_hints": "Optimize the plan, transfer already existing answers from previous executions! For example, weather values.\
                Don't forget tasks! Plan the Coder Agent to save files.\
                    Be more careful about which tasks can be performed in parallel and which ones can be performed sequentially.\
                        For example, you cannot fill a table and save it in parallel.",
            },
        },
    },
}
