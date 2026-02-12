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
    coder_agent,
    papers_search_agent
)
from ChemCoScientist.tools import chem_tools_rendered, nano_tools_rendered, tools_rendered, data_tools_rendered, \
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
'coder_agent' - Is expert in data science. It can write and execute python code. Always use it for data management and EDA.
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

Special behavior for dataset creation:
- If 'create_dataset_from_papers' is requested but no papers are uploaded:
  1) Augment the user query to search for relevant papers.
  2) Automatically invoke 'papers_search_agent' to find relevant papers with augmented query.
  3) After successful download, retry dataset creation with downloaded papers.
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


papers_search_agent_description = """
Agent name: papers_search_agent

Purpose:
Search OpenAlex for relevant scientific papers, download their PDF files, and return
download metadata for downstream processing.

When to activate:
- User requests finding or downloading papers for a given research topic, author,
    journal, or institution.

Procedure (implementation details):
1) Use an LLM (via the configured `VISION_LLM_URL`) to generate the appropriate
     OpenAlex API request URL for the user's query.
2) Call OpenAlex (with retry logic) and inspect the returned `results`.
3) For each result containing a `content_urls.pdf`, download the PDF and save it to
     the configured `DOWNLOADED_PAPERS_PATH` using a sanitized filename.
4) Return a structured response containing a human-readable `answer` and `metadata`.
    When PDFs were downloaded, `metadata.papers` contains the list of saved file paths.
    For queries that resolve to an entity (author/source/institution), the agent may
    return an `id` in `metadata` instead of (or in addition to) downloaded files.

Two-step / entity-ID flow:
- The agent can be used in a two-step pattern for author/journal/institution queries:
  1) First call the agent to resolve the target entity to an OpenAlex ID (the agent
    will return `metadata.id` when it detects an entity-resolution response).
  2) Then call the agent again (or include the resolved ID in the original query)
    to search for and download papers associated with that entity. This two-step
    approach is supported by the implementation and recommended for precise author or
    source-based searches.

Notes and constraints:
- The agent builds the OpenAlex request via an LLM and then performs the HTTP calls
    directly; network retry/backoff logic is applied for robustness.
- The agent downloads PDFs listed in `content_urls.pdf` from OpenAlex results and
    saves them locally; it does not attempt to bypass paywalls beyond what OpenAlex
    exposes in `content_urls`.

Inputs:
- user_query: str

Outputs:
- A dict with an `answer` string and optional `metadata` dict. When downloads occur,
    `metadata.papers` is a list of downloaded file paths; when an entity ID is resolved,
    `metadata.id` is provided.

Failure handling:
- If no papers are found or downloads fail, the agent returns an explanatory `answer`
    and an empty or absent `metadata.papers`.
"""


additional_agents_description = (
    automl_agent_description
    + dataset_builder_agent_description
    + coder_agent_description
    + paper_analysis_agent_description
    + web_search_description
    + papers_search_agent_description
)

conf = {
    # maximum number of recursions
    "recursion_limit": 25,
    "configurable": {
        "user_id": "1",
        "session_id": "1",
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
            "web_search",
            "papers_search_agent"
        ],
        # nodes for scenario agents
        "scenario_agent_funcs": {
            "chemist_node": chemist_node,
            "nanoparticle_node": nanoparticle_node,
            "ml_dl_agent": ml_dl_agent,
            "dataset_builder_agent": dataset_builder_agent,
            "coder_agent": coder_agent,
            "paper_analysis_agent": paper_analysis_agent,
            "web_search": web_search_node,
            "papers_search_agent": papers_search_agent
        },
        # descripton for agents tools - if using langchain @tool
        # or description of agent capabilities in free format
        "tools_for_agents": {
            "chemist_node": [chem_tools_rendered],
            "nanoparticle_node": [nano_tools_rendered],
            "dataset_builder_agent": [data_tools_rendered],
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
                "problem_statement": """
                    Your task is to analyze the user's objective and design a structured plan
                    consisting of atomic subtasks. Each subtask must be directly executable by one
                    of the system agents listed below. Subtasks that do not depend on each other
                    should be grouped together so they can be executed simultaneously.
                    The output must represent this plan as a list of lists, where each inner list
                    contains tasks that can be performed in parallel.
                    """,
                "rules": """
                    1. Each subtask must correspond to an action that can be handled by one of the available system agents.
                    2. Dependent subtasks must be placed in separate sequential steps.
                    3. Independent subtasks (i.e., that can run in parallel) should appear in the same inner list.
                    4. Every subtask must be expressed clearly as an action (e.g., 'Generate X, Find Y').
                    5. Avoid unnecessary decomposition — only split when separate agents are required or there are dependencies.
                    6. Keep logical order and coherence between subtasks.
                    7. You must include all information you see in user prompt to your plan
                    8. If you get a general question about chemistry first call paper_analysis_agent. Use web search
                    only if paper_analysis_agent has no answer. 
                    9. If you get a query to find or download papers, use papers_search_agent:
                       - For topic-based searches (e.g., "Download papers about CRISPR CAS"), directly search for papers using that topic.
                       - For author, journal, or institution searches, create two sequential subtasks: first resolve the entity's OpenAlex ID, then search for papers using that ID.
                    """,
                "desc_restrictions": """
                    - You cant name agents
                    - The plan must contain no more than 5 steps.
                    - Each step must include at least one subtask.
                    - Each subtask should be short (one concise sentence or phrase).
                    - All task related coding must be collected in one single task. Don't split it.
                    - When you asked to train models using dataset procceed to training
                    """,
                "examples": """
                    Example 1:
                    Request: "Collect spectra for sample A and B, then analyze them"
                    Response: {
                        "steps": [
                            ["Collect spectra for sample A", "Collect spectra for sample B"],
                            ["Analyze spectra of collected samples"]
                        ]
                    }

                    Example 2:
                    Request: "Generate dataset, train model, and predict for molecule1 and molecule2"
                    Response: {
                        "steps": [
                            ["Generate dataset"],
                            ["Train model"],
                            ["Predict for molecule1", "Predict for molecule2"]
                        ]
                    }

                    Example 3:
                    Request: "Generate 5 molecules related to MEK1, make 3 molecules using the GSK model"
                    Response: {
                        "steps": [
                            ["Generate 5 molecules related to MEK1", "Generate 3 molecules using the GSK model"]
                        ]
                    }

                    Example 4 (author search):
                    Request: "Find papers by author 'Jane Q. Researcher' about quantum dots"
                    Response: {
                        "steps": [
                            ["Search OpenAlex for author ID for 'Jane Q. Researcher'"],
                            ["Search OpenAlex for papers by the found author ID about quantum dots"]
                        ]
                    }

                    Example 5 (topic search):
                    Request: "Download papers about CRISPR CAS"
                    Response: {
                        "steps": [
                            ["Search and download papers about CRISPR CAS"]
                        ]
                    }
                    """,
                "additional_hints": """
                    - If multiple molecules, files, or entities are processed in the same way, group those actions together as parallel subtasks.
                    - When an earlier step produces data required for another (e.g., training before prediction), make sure the dependent step comes later.
                    - If the user request is ambiguous, infer a reasonable decomposition based on tool capabilities.
                    - If the user requests molecule or reaction extraction from images call chemical agent.
                    """,
            },
            "chat": {
                "problem_statement": None,
                "additional_hints": """
                You are a chemical agent system. Your language is Russian. You can do the following:
                - train generative models (generate SMILES molecules), train predictive models (predict properties)
                - prepare a dataset for training
                - download data from chemical databases: ChemBL, BindingDB
                - perform calculations with chemical python libraries
                - solve problems of nanomaterial synthesis
                - analyze chemical articles
                - extract molecules and reactions from images
                If user ask something like "What can you do" - make answer yourself!
                
                If user asks to extract some data from images do not answer the question yourself, pass it to the planner
                    """,
            },
            "summary": {
                "problem_statement": """Your task is to compose the **final answer** for the user, based on 
                    `system_response` and `intermediate_thoughts`. Your language is Russian. Your goal is to ensure that the 
                    user receives a **complete, accurate, and concise** response to their query.""",
                "rules": """Your response must be the **direct and final answer** to the user’s query.
                    - Do **not** describe what was done — instead, **present what was achieved**.  
                    - Extract and summarize **all insights, results, and conclusions**. 
                    - In your response, include every piece of information provided by paper_analysis_agent. 
                      Ensure the answer is fully comprehensive and no data is overlooked or omitted. 
                    - Avoid unnecessary filler, explanations, or meta-text.  
                    - When appropriate, organize the answer as a **short report** with sections 
                    such as *Summary*, *Results*, *Findings*, *Conclusion*, etc. (They must be in Russian.)  
                    - The *Results* section must contain all facts provided by the agents
                    - Always ensure your response **directly answers the user’s query**.  
                    - Respond in **markdown** format.
                    - Double-check that your answer is **complete, accurate, and self-contained**.
                    - You must exclude any non-standard or unreadable symbols (hieroglyphs).""",

                "additional_hints": """
                    Never include full file paths — only file names.
                    """,

            },
            "replanner": {
                "problem_statement": """
                    You are a replanning expert. Your job is to optimize and adjust an existing
                    step-by-step plan based on what has already been completed. You must not
                    invent or introduce new tasks — only update, reorder, or remove steps as
                    necessary to reach the final goal efficiently. When all tasks are done,
                    return a final response instead of new steps.
                    """,
                "rules": """
                    1. Each subtask must correspond to an action that can be handled by one of the available system agents.
                    2. Every subtask must be expressed clearly as an action (e.g., 'Generate X, Find Y').
                    3. Do not create new tasks beyond the original plan. You can only adjust current tasks.
                    4. Remove tasks that are already completed.
                    5. You must include past steps results to enrich current tasks. 
                    6. Preserve the logical order of dependent tasks.
                    7. Tasks that can be done in parallel should remain grouped in the same list.
                    8. When all tasks are finished, output the final response (summary or confirmation).
                    9. You cant name agents
                    10. All task related coding must be collected in one single task. Don't split it.
                    11. The plan must contain no more than 5 steps.
                    12. You can not refer to past steps results, you must include them into your plan
                    13. Always return output strictly in JSON format.
                    """,
                "examples": """
                    Example 1:
                    Original plan:
                    [
                        ["Collect data for BTK with IC50 values from ChEMBL using dataset_builder_agent"],
                        ["Clean and preprocess the BTK IC50 dataset"],
                        ["Train model"]
                    ]
                    Past steps:
                    {
                        ("Collect data for BTK with IC50 values from ChEMBL using dataset_builder_agent", "Dataset saved to /data/BTK_IC50.csv")
                    }

                    Response:
                    {
                        "action": "steps",
                        "steps": [
                            ["Clean and preprocess the /data/BTK_IC50.csv dataset"],
                            ["Train model"]
                        ]
                    }

                    Example 2:
                    Original plan:
                    [
                        ["Collect data for BTK with IC50 values from ChEMBL using dataset_builder_agent"]
                    ]
                    Past steps:
                    {
                        ("Collect data for BTK with IC50 values from ChEMBL using dataset_builder_agent", "Dataset saved successfully to /data/BTK_IC50.csv")
                    }

                    Response:
                    {
                        "action": "response",
                        "response": "The dataset for BTK with IC50 values has been successfully collected and stored: /data/BTK_IC50.csv. No further actions are required."
                    }
                    """,
                "additional_hints": """
                    - Optimize the plan using existing completed results.
                    - You must include past steps results into your plan.
                    - Do not create new tasks or agents that were not part of the original plan.
                    - Be careful with task dependencies — never run saving or training before required data is ready.
                    - Remember: return only JSON, no text before or after it.
                    """,
            },
        },
    },
}
