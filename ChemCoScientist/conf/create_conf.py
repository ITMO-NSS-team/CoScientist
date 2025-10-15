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
from ChemCoScientist.tools import chem_tools_rendered, nano_tools_rendered, tools_rendered, \
    paper_analysis_tools_rendered
from definitions import ROOT_DIR


# description for agent WITHOUT langchain-tools
automl_agent_description = """
'ml_dl_agent' - an agent that can run training of a generative model to generate SMILES, training of predictive models 
to predict properties. It also already stores ready-made models for inference. You can also ask him to prepare an 
existing dataset (you need to be specific in your request).
It can generate medicinal molecules. You must use this agent for molecules generation!!!
"""

dataset_builder_agent_description = """
'dataset_builder_agent' - collects data from two databases - ChemBL and BindingDB.
To collect data, it needs either the protein name or a specific id from a specific database. 
It can collect data from one specific database or from both. All data is saved locally.
"""

coder_agent_description = """
'coder_agent' - can write any simple python scientific code. Can use rdkit and other 
chemical libraries. Can perform calculations.
"""

paper_analysis_agent_description = """
Agent name: paper_analysis_agent

Purpose: Retrieve and analyze chemical science papers from the internal database.
When to activate: User asks about chemistry articles, papers, or research findings.
Procedure:
    1) Plan the agent's steps.
    2) Query the internal database for relevant papers.
    3) Call "web_search" to add recent or missing internet information.

Constraints: Do not call other agents unless the user explicitly requests them.
Do not use explore_my_papers tool if user has not provided you his papers.

Inputs:
- user_query: str

Outputs:
- Clear summary of key findings.
- Citations to database record IDs and web source URLs.
- Noted assumptions or gaps.

Failure handling:
If no relevant papers are found, state "no match in database" and still run "web_search".
"""

web_search_description = """
Agent name: web_search_node

Purpose:
Find and summarize up-to-date information from the public internet.

When to activate:
- User explicitly asks to search the web or look online.
- Current or changing data is needed.
- Another agent requests external verification or recency checks.

Procedure:
1) Parse the user request into search intents and keywords.
2) Run web queries and collect top relevant sources.
3) Extract key facts, dates, and figures.
4) Produce a concise summary with citations.
5) Return results to the caller.

Constraints:
- Run as a separate agent, not bundled with others in the same turn.
- Do not fabricate URLs or claims.
- Prefer primary and authoritative sources.

Inputs:
- user_query: str
- optional_context: dict  # e.g., domain hints, date range, locale

Outputs:
- summary: str
- sources: list  # [{"title": str, "url": str, "date": str|None}]

Failure handling:
- If no reliable sources are found, state "no reliable sources" and suggest query refinement.
- If sources are paywalled, note it and provide accessible alternatives when possible.
"""


additional_agents_description = (
    automl_agent_description
    + dataset_builder_agent_description
    + coder_agent_description
    + paper_analysis_agent_description
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
            f"{os.environ['MAIN_LLM_URL']};{os.environ['MAIN_LLM_MODEL']}",
            extra_body={"temperature": 0.0}
        ),
        "max_retries": 3,
        # list of scenario agents
        "scenario_agents": [
            "chemist_node",
            "nanoparticle_node",
            "ml_dl_agent",
            "dataset_builder_agent",
            "coder_agent",
            "paper_analysis_agent",
            "web_search"
        ],
        # nodes for scenario agents
        "scenario_agent_funcs": {
            "chemist_node": chemist_node,
            "nanoparticle_node": nanoparticle_node,
            "ml_dl_agent": ml_dl_agent,
            "dataset_builder_agent": dataset_builder_agent,
            "coder_agent": coder_agent,
            "paper_analysis_agent": paper_analysis_agent,
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
            "paper_analysis_agent": [paper_analysis_tools_rendered],
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
                "ds_dir": os.path.join(ROOT_DIR, os.environ["DS_STORAGE_PATH"]),
            },
            "coder_agent": {
                "model_name": os.environ["SCENARIO_LLM_MODEL"],
                "url": os.environ["SCENARIO_LLM_URL"],
                "api_key": os.environ["OPENAI_API_KEY"],
                "ds_dir": os.path.join(ROOT_DIR, os.environ["ANOTHER_STORAGE_PATH"]),
            },
            "ml_dl_agent": {
                "model_name": os.environ["SCENARIO_LLM_MODEL"],
                "url": os.environ["SCENARIO_LLM_URL"],
                "api_key": os.environ["OPENAI_API_KEY"],
                "ds_dir": os.path.join(ROOT_DIR, os.environ["DS_STORAGE_PATH"]),
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
                "additional_hints": """
                Before starting model training, check data for garbage with 'dataset_builder_agent'. 
                If the user already provides a dataset, go straight to 'ml_dl_agent' and skip 'dataset_builder_agent' 
                For questions about papers, articles, or research findings, plan exactly two steps: 
                first 'paper_analysis_agent', then 'web_search'. 
                Do not schedule any other agents for such research tasks.
                If user asks find something in internet you have to use 'web_search'.
                Always choose the minimal set of agents necessary for the user's request.
                """,
            },
            "chat": {
                "problem_statement": None,
                "additional_hints": """
                You are a chemical agent system. You can do the following:
                - train generative models (generate SMILES molecules), train predictive models (predict properties)
                - prepare a dataset for training
                - download data from chemical databases: ChemBL, BindingDB
                - perform calculations with chemical python libraries
                - solve problems of nanomaterial synthesis
                - analyze chemical articles
                If user ask something like "What can you do" - make answer yourself!
                    """,
            },
            "summary": {
                "problem_statement": None,
                "rules": None,
                "additional_hints": """                
                Never write full paths! Only file names. If 'paper_analysis_agent' and 'web_search' were used,  
                present the final answer as: paper_analysis: <paper_analysis_agent result>   web_search: <web_search_node result>.
                """,
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
