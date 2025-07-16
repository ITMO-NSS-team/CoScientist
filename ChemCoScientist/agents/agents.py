import os
import time
from typing import Annotated
from langgraph.types import Command

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent
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
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            file_paths.append(full_path)
    return file_paths


from typing import Annotated

def dataset_builder_agent(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("Dataset builder agent called")
    print(state["task"])
    print("--------------------------------")
    task = state["task"]

    config_cur_agent = config["configurable"]["additional_agents_info"][
        "dataset_builder_agent"
    ]
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

    return Command(update={
        "dataset_builder_outputs": Annotated[list, "accumulate"]([(task, str(response))]),
        "metadata": {"dataset_builder_agent": get_all_files(os.environ["DS_STORAGE_PATH"])},
        "nodes_calls": Annotated[list, "accumulate"]([("dataset_builder_agent", str(response))]),
        "past_steps": Annotated[list, "accumulate"]([(task, str(response))]),
    })


def ml_dl_agent(state: dict, config: dict) -> Command:
    print("--------------------------------")
    print("ml_dl agent called")
    print(state["task"])
    print("--------------------------------")

    config_cur_agent = config["configurable"]["additional_agents_info"]["ml_dl_agent"]
    task = state["task"]

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
        update={
            "past_steps": Annotated[list, "accumulate"]([(task, str(response))]),
            "nodes_calls": Annotated[list, "accumulate"]([("ml_dl_agent", str(response))]),
        }
    )


def chemist_node(state, config: dict) -> Command:
    print("--------------------------------")
    print("Chemist agent called")
    print("Current task:")
    print(state["task"])
    print("--------------------------------")

    task = state["task"]
    plan: list = state["plan"]
    
    llm = config["configurable"]["llm"]
    chem_agent = create_react_agent(
        llm, chem_tools, state_modifier=worker_prompt + "admet = qed"
    )

    task_formatted = f"""For the following plan:
    {str(plan)}\n\nYou are tasked with executing: {task}."""

    max_retries = 3
    for attempt in range(max_retries):
        try:
            config["configurable"]["state"] = state
            agent_response = chem_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(
                update={
                    "past_steps": Annotated[list, "accumulate"](
                        [(task, agent_response["messages"][-1].content)]
                    ),
                    "nodes_calls": Annotated[list, "accumulate"](
                        [("chemist_node", agent_response["messages"])]
                    ),
                },
                goto="replan_node"
            )

        except Exception as e:
            print(f"Chemist failed: {str(e)}. Retrying ({attempt+1}/{max_retries})")
            time.sleep(1.2**attempt)

    return Command(
        goto=END,
        update={
            "response": "I can't answer your question right now. Perhaps I can help with something else?"
        },
    )

def nanoparticle_node(state, config: dict) -> Command:
    print("--------------------------------")
    print("Nano-p agent called")
    print("Current task:")
    print(state["task"])
    print("--------------------------------")

    task = state["task"]
    plan = state["plan"]
    llm = config["configurable"]["llm"]
    
    nanoparticle_agent = create_react_agent(
        llm, nanoparticle_tools, state_modifier=worker_prompt + "You have to respond with results of tool call, do not rephrase it"
    )

    task_formatted = f"""For the following plan:
    {str(plan)}\n\nYou are tasked with executing: {task}."""

    for attempt in range(3):
        try:
            agent_response = nanoparticle_agent.invoke(
                {"messages": [("user", task_formatted)]}
            )

            return Command(
                update={
                    "past_steps": Annotated[list, "accumulate"](
                        [(task, agent_response["messages"][-1].content)]
                    ),
                    "nodes_calls": Annotated[list, "accumulate"](
                        [("nanoparticle_node", agent_response["messages"])]
                    ),
                },
                goto="replan_node"
            )

        except Exception as e:
            print(f"Nanoparticle error: {str(e)}. Retrying ({attempt+1}/3)")
            time.sleep(1.2**attempt)

    return Command(
        goto=END,
        update={
            "response": "I can't answer your question right now. Perhaps I can help with something else?"
        },
    )



def paper_analysis_node(state: dict) -> Command:
    print("--------------------------------")
    print("Paper agent called")
    print("Current task:")
    print(state["task"])
    print("--------------------------------")
    
    task = state["task"]
    response = process_question(task)

    updated_metadata = state.get("metadata", {}).copy()
    updated_metadata["paper_analysis"] = response.get("metadata")

    return Command(
        update={
            "past_steps": Annotated[list, "accumulate"]([(task, response.get("answer"))]),
            "nodes_calls": Annotated[list, "accumulate"]([("paper_analysis_node", response.get("answer"))]),
            "metadata": updated_metadata,
        },
        goto="replan_node"
    )
