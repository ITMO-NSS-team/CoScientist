import ast
import os
import time
from typing import Annotated
import operator
import streamlit as st

from langgraph.types import Command
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent
from smolagents import CodeAgent, LiteLLMModel, OpenAIServerModel

from ChemCoScientist.agents.agents_prompts import (
    additional_ds_builder_prompt,
    automl_prompt,
    ds_builder_prompt,
    worker_prompt,
)
from ChemCoScientist.tools import chem_tools, nanoparticle_tools, paper_analysis_tools
from ChemCoScientist.tools import chem_tools, nanoparticle_tools
from ChemCoScientist.tools.chemist_tools import fetch_BindingDB_data, fetch_chembl_data
from ChemCoScientist.tools.ml_tools import agents_tools as automl_tools

from definitions import ROOT_DIR


def get_all_files(directory):
    file_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths


def dataset_builder_agent(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("Dataset builder agent called")
    print(state["task"])
    print("--------------------------------")
    task = state["task"]

    config_cur_agent = config["configurable"]["additional_agents_info"]["dataset_builder_agent"]
    print(config_cur_agent)

    model = (
        LiteLLMModel(config_cur_agent["model_name"], api_base=config_cur_agent["url"], api_key=config_cur_agent["api_key"])
        if "groq.com" in config_cur_agent["url"]
        else OpenAIServerModel(api_base=config_cur_agent["url"], model_id=config_cur_agent["model_name"], api_key=config_cur_agent["api_key"])
    )

    agent = CodeAgent(
        tools=[fetch_BindingDB_data, fetch_chembl_data],
        model=model,
        additional_authorized_imports=["*"],
    )

    response = agent.run(
        ds_builder_prompt + config_cur_agent["ds_dir"] + "\nSo, user ask: \n" + task + additional_ds_builder_prompt
    )

    files = get_all_files(os.path.join(ROOT_DIR, os.environ["DS_STORAGE_PATH"]))

    return Command(update={
        "past_steps": Annotated[set, operator.or_](set([(task, str(response))])),
        "nodes_calls": Annotated[set, operator.or_](set([
            ("dataset_builder_agent", (("text", str(response)),))
        ])),
        "metadata": Annotated[dict, operator.or_]({
            "dataset_builder_agent": files
        }),
    })


def ml_dl_agent(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("ml_dl agent called")
    print(state["task"])
    print("--------------------------------")

    task = state["task"]
    config_cur_agent = config["configurable"]["additional_agents_info"]["ml_dl_agent"]

    model = (
        LiteLLMModel(config_cur_agent["model_name"], api_base=config_cur_agent["url"], api_key=config_cur_agent["api_key"])
        if "groq.com" in config_cur_agent["url"]
        else OpenAIServerModel(api_base=config_cur_agent["url"], model_id=config_cur_agent["model_name"], api_key=config_cur_agent["api_key"])
    )

    agent = CodeAgent(
        tools=automl_tools,
        model=model,
        additional_authorized_imports=["*"],
    )
    response = agent.run(automl_prompt + task)

    return Command(update={
        "past_steps": Annotated[set, operator.or_](set([(task, str(response))])),
        "nodes_calls": Annotated[set, operator.or_](set([
            ("ml_dl_agent", (("text", str(response)),))
        ])),
    })


def chemist_node(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("Chemist agent called")
    print("Current task:")
    print(state["task"])
    print("--------------------------------")

    task = state["task"]
    plan = state["plan"]
    llm = config["configurable"]["llm"]

    chem_agent = create_react_agent(
        llm, chem_tools, state_modifier=worker_prompt + "admet = qed"
    )

    task_formatted = f"""For the following plan:\n{str(plan)}\n\nYou are tasked with executing: {task}."""

    for attempt in range(3):
        try:
            config["configurable"]["state"] = state
            agent_response = chem_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(update={
                "past_steps": Annotated[set, operator.or_](set([
                    (task, agent_response["messages"][-1].content)
                ])),
                "nodes_calls": Annotated[set, operator.or_](set([
                    (
                        "chemist_node",
                        tuple((m.type, m.content) for m in agent_response["messages"])
                    )
                ])),
            })

        except Exception as e:
            print(f"Chemist failed: {str(e)}. Retrying ({attempt+1}/3)")
            time.sleep(1.2**attempt)

    return Command(goto=END, update={
        "response": "I can't answer your question right now. Perhaps I can help with something else?"
    })


def nanoparticle_node(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("Nano-p agent called")
    print("Current task:")
    print(state["task"])
    print("--------------------------------")

    task = state["task"]
    plan = state["plan"]
    llm = config["configurable"]["llm"]

    nanoparticle_agent = create_react_agent(
        llm, nanoparticle_tools,
        state_modifier=worker_prompt + "You have to respond with results of tool call, do not rephrase it"
    )

    task_formatted = f"""For the following plan:\n{str(plan)}\n\nYou are tasked with executing: {task}."""

    for attempt in range(3):
        try:
            agent_response = nanoparticle_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(update={
                "past_steps": Annotated[set, operator.or_](set([
                    (task, agent_response["messages"][-1].content)
                ])),
                "nodes_calls": Annotated[set, operator.or_](set([
                    (
                        "nanoparticle_node",
                        tuple((m.type, m.content) for m in agent_response["messages"])
                    )
                ])),
            })

        except Exception as e:
            print(f"Nanoparticle error: {str(e)}. Retrying ({attempt+1}/3)")
            time.sleep(1.2**attempt)

    return Command(goto=END, update={
        "response": "I can't answer your question right now. Perhaps I can help with something else?"
    })


def paper_analysis_agent(state: dict, config: dict) -> Command:
    """
    The agent assists users by analyzing information from scientific papers.
    It can do several things:
    - answers the user's question using a DB with chemical scientific papers
    - answers the user's question using papers that were uploaded by the user
    - selects papers relevant to the user's question

    Args:
        state: The current execution.

    Returns:
        An object containing the next node to transition to ('replan' or `END`) and
        an update to the execution state with recorded steps and responses.
    """
    print("--------------------------------")
    print("Paper agent called")
    print("Current task:")
    print(state["task"])
    print("--------------------------------")

    llm: BaseChatModel = config["configurable"]["llm"]

    task = state["task"]

    # TODO: update this when proper frontend is added
    try:
        current_prompt = f'{worker_prompt}/n session_id = {st.session_state.session_id}'
    except:
        current_prompt = f'{worker_prompt}/n session_id is not needed in this case, pass None'

    paper_analysis_agent = create_react_agent(
        llm, paper_analysis_tools, state_modifier=current_prompt
    )

    for attempt in range(3):
        try:
            response = paper_analysis_agent.invoke({"messages": [("user", task)]})

            result = ast.literal_eval(response["messages"][2].content)

            updated_metadata = state.get("metadata", {}).copy()
            pa_metadata = {"paper_analysis": result.get("metadata")}
            if pa_metadata["paper_analysis"]:
                updated_metadata.update(pa_metadata)

            if type(result["answer"]) is list:
                result["answer"] = ', '.join(result["answer"])

            return Command(update={
                "past_steps": Annotated[set, operator.or_](set([
                    (task, result["answer"])
                ])),
                "nodes_calls": Annotated[set, operator.or_](set([
                    ("paper_analysis_agent", (("text", result["answer"]),))
                ])),
                "metadata": Annotated[dict, operator.or_](updated_metadata),
            })
        except Exception as e:
            print(f"Paper analysis agent error: {str(e)}. Retrying ({attempt + 1}/3)")
            time.sleep(1.2 ** attempt)

    return Command(goto=END, update={
        "response": "I cannot answer your question right now using the DB or uploaded papers."
                    "Can I help with something else?"
    })
