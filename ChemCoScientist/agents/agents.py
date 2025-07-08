import os
import time

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from smolagents import CodeAgent, DuckDuckGoSearchTool, LiteLLMModel, OpenAIServerModel

from ChemCoScientist.agents.agents_prompts import (
    additional_ds_builder_prompt,
    automl_prompt,
    ds_builder_prompt,
    worker_prompt,
)
from ChemCoScientist.paper_analysis.question_processing import process_question
from ChemCoScientist.tools import chem_tools, nanoparticle_tools
from ChemCoScientist.tools.chemist_tools import fetch_BindingDB_data, fetch_chembl_data
from ChemCoScientist.tools.ml_tools import agents_tools as automl_tools


def get_all_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    return file_paths


def dataset_builder_agent(state: dict, config: dict):
    config_cur_agent = config["configurable"]["additional_agents_info"][
        "dataset_builder_agent"
    ]
    plan = state["plan"]
    task = plan[0]
    print(config_cur_agent)
    if "groq.com" in config_cur_agent["url"]:
        model = LiteLLMModel(
            config_cur_agent["model_name"],
            api_base=config_cur_agent["url"],
            api_key=config_cur_agent["api_key"],
        )
    else:

        model = OpenAIServerModel(
            api_base=config_cur_agent["url"],
            model_id=config_cur_agent["model_name"],
            api_key=config_cur_agent["api_key"],
        )

    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool(), fetch_BindingDB_data, fetch_chembl_data],
        model=model,
        additional_authorized_imports=["*"],
    )

    response = agent.run(
        ds_builder_prompt + config_cur_agent["ds_dir"] + "\n"
        "So, user ask: \n" + task + additional_ds_builder_prompt
    )
    state["metadata"]["dataset_builder_agent"] = get_all_files(
        os.environ["DS_STORAGE_PATH"]
    )

    return Command(
        goto="replan_node",
        update={
            "metadata": state["metadata"],
            "past_steps": [(task, str(response))],
            "nodes_calls": [("dataset_builder_agent", str(response))],
        },
    )


def ml_dl_agent(state: dict, config: dict):
    config_cur_agent = config["configurable"]["additional_agents_info"]["ml_dl_agent"]
    plan = state["plan"]
    task = plan[0]

    if "groq.com" in config_cur_agent["url"]:
        model = LiteLLMModel(
            config_cur_agent["model_name"],
            api_base=config_cur_agent["url"],
            api_key=config_cur_agent["api_key"],
        )
    else:
        model = OpenAIServerModel(
            api_base=config_cur_agent["url"],
            model_id=config_cur_agent["model_name"],
            api_key=config_cur_agent["api_key"],
        )

    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool()] + automl_tools,
        model=model,
        additional_authorized_imports=["*"],
    )

    response = agent.run(automl_prompt + task)

    return Command(
        goto="replan_node",
        update={
            "past_steps": [(task, str(response))],
            "nodes_calls": [("ml_dl_agent", str(response))],
        },
    )


def chemist_node(state, config: dict):
    """
    Executes a chemistry-related task using a ReAct-based agent.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state containing the task plan.
    config : dict
        Configuration dictionary containing the LLM model and related settings.

    Returns
    -------
    Command
        An object specifying the next execution step and updates to the state.
    """
    llm: BaseChatModel = config["configurable"]["llm"]
    chem_agent = create_react_agent(
        llm, chem_tools, state_modifier=worker_prompt + "admet = qed"
    )

    plan: list = state["plan"]
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task: str = plan[0]

    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            config["configurable"][
                "state"
            ] = state  # to get tool_call_id, but this shouldn't be implemented like that
            agent_response = chem_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(
                goto="replan_node",
                update={
                    "past_steps": [(task, agent_response["messages"][-1].content)],
                    "nodes_calls": [("chemist_node", agent_response["messages"])],
                },
            )

        except Exception as e:
            print(
                f"Chemist failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})"
            )
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={
            "response": "I can't answer to your question right now. Perhaps there is something else that I can help?"
        },
    )


def nanoparticle_node(state, config: dict):
    """
    Executes a task related to nanoparticle analysis using a large language model (LLM)
    and associated tools, retrying up to a maximum number of times in case of failure.

    Parameters
    ----------
    state : dict | TypedDict
        The current execution state, which includes the planned steps and other contextual
        information for processing.
    config : dict
        A dictionary containing configurable parameters, including the LLM model
        and any relevant tools required for task execution.

    Returns
    -------
    Command
        An object containing the next node to transition to ('replan' or `END`) and
        an update to the execution state with recorded steps and responses.

    Notes
    -----
    - The function constructs a task prompt based on the execution plan and invokes
      an LLM-based agent to process the task.
    - If an exception occurs, it retries the operation with exponential backoff.
    - If all retries fail, it returns a response indicating failure.
    """
    llm: BaseChatModel = config["configurable"]["llm"]
    add_prompt = "You have to respond with results of tool call, do not repharse it"
    nanoparticle_agent = create_react_agent(
        llm, nanoparticle_tools, state_modifier=worker_prompt + add_prompt
    )

    plan: list = state["plan"]
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task: str = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            agent_response = nanoparticle_agent.invoke(
                {"messages": [("user", task_formatted)]}
            )

            return Command(
                goto="replan_node",
                update={
                    "past_steps": [(task, agent_response["messages"][-1].content)],
                    "nodes_calls": [("nanoparticle_node", agent_response["messages"])],
                },
            )

        except Exception as e:
            print(
                f"Nanoparticle failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})"
            )
            time.sleep(1.2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={
            "response": "I can't answer to your question right now. Perhaps there is something else that I can help?"
        },
    )


def paper_analysis_node(state: dict) -> Command:
    """
    Answers the user's question using a DB with chemical scientific papers and a vision LLM.
    Takes into account text, images and tables.

    Args:
        state: The current execution.

    Returns:
        An object containing the next node to transition to ('replan' or `END`) and
        an update to the execution state with recorded steps and responses.
    """
    plan = state["plan"]
    task = plan[0]

    # Process question
    response = process_question(task)

    # Add metadata from response to state
    state_metadata = state.get("metadata", {})
    pa_metadata = {"paper_analysis": response.get("metadata")}
    updated_metadata = state_metadata.copy()
    updated_metadata.update(pa_metadata)

    return Command(
        goto="replan_node",
        update={
            "past_steps": [(task, response.get("answer"))],
            "nodes_calls": [("paper_analysis_node", response.get("answer"))],
            "metadata": updated_metadata,
        },
    )
