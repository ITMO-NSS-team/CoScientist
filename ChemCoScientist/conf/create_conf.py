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
    coder_agent
)
#from CoScientist.scientific_agents.agents import coder_agent
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
                    """,
                "additional_hints": """
                    - If multiple molecules, files, or entities are processed in the same way, group those actions together as parallel subtasks.
                    - When an earlier step produces data required for another (e.g., training before prediction), make sure the dependent step comes later.
                    - If the user request is ambiguous, infer a reasonable decomposition based on tool capabilities.
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
                "problem_statement": """Your task is to compose the **final answer** for the user, based on 
                    `system_response` and `intermediate_thoughts`. Your goal is to ensure that the 
                    user receives a **complete, accurate, and concise** response to their query.""",
                "rules": """Your response must be the **direct and final answer** to the user’s query.
                    - Do **not** describe what was done — instead, **present what was achieved**.  
                    - Extract and summarize **all key insights, results, and conclusions**.  
                    - Avoid unnecessary filler, explanations, or meta-text.  
                    - When appropriate, organize the answer as a **short report** with sections 
                    such as *Summary*, *Results*, *Findings*, *Conclusion*, etc.  
                    - Always ensure your response **directly answers the user’s query**.  
                    - Respond in **markdown** format.
                    - Double-check that your answer is **complete, accurate, and self-contained**.""",

                "additional_hints": """                
                    Never include full file paths — only file names.  
                    If multiple agents or nodes were involved (e.g., `paper_analysis_agent`, `web_search`), 
                    summarize their contributions clearly, for example:

                    **paper_analysis:** <summary of paper_analysis_agent result>  
                    **web_search:** <summary of web_search result>  
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
                    5. If a step’s outputs are available in past results, skip that step.
                    6. Preserve the logical order of dependent tasks.
                    7. Tasks that can be done in parallel should remain grouped in the same list.
                    8. When all tasks are finished, output the final response (summary or confirmation).
                    9. You cant name agents
                    10. All task related coding must be collected in one single task. Don't split it.
                    11. The plan must contain no more than 5 steps.
                    12. Always return output strictly in JSON format.
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
                    - You must transfer ready outputs (paths, values) from past steps when possible.
                    - Do not create new tasks or agents that were not part of the original plan.
                    - Be careful with task dependencies — never run saving or training before required data is ready.
                    - Remember: return only JSON, no text before or after it.
                    """,
            },
        },
    },
}
