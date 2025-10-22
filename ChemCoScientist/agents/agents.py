import ast
import os
import time
from typing import Annotated
import operator
import streamlit as st
from langchain_core.language_models import BaseChatModel

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
from ChemCoScientist.tools import chem_tools, nanoparticle_tools, paper_analysis_tools, data_tools
from ChemCoScientist.tools.ml_tools import agents_tools as automl_tools

from ChemCoScientist.agents.agents_prompts import paper_agent_prompt, coder_prompt
from definitions import ROOT_DIR


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



def dataset_builder_agent(state: dict, config: dict):

    task = state["task"]
    plan = state["plan"]
    llm = config["configurable"]["llm"]

    worker_prompt = f"Save data to this path: {os.path.join(ROOT_DIR, os.environ['DS_STORAGE_PATH'])}"
    data_agent = create_react_agent(
        llm, tools=data_tools, prompt=worker_prompt
    )

    task_formatted = f"""For the following plan:\n{str(plan)}\n\nYou are tasked with executing: {task}."""
    inputs = {"messages": [{"role": "user", "content": task_formatted}]}

    response = data_agent.invoke({"messages": [("user", task_formatted)]})

    files = get_all_files(os.path.join(ROOT_DIR, os.environ["DS_STORAGE_PATH"]))

    return Command(update={
        "past_steps": Annotated[set, operator.or_](set([(task, response["messages"][-1].content)])),
        "nodes_calls": Annotated[set, operator.or_](set([
            ("dataset_builder_agent", (("text", response["messages"][-1].content),))
        ])),
        "metadata": Annotated[dict, operator.or_]({
            "dataset_builder_agent": files
        }),
    })

def coder_agent(state: dict, config: dict):

    task = state["task"]
    plan = state["plan"]

    config_cur_agent = config["configurable"]["additional_agents_info"]["coder_agent"]

    model = (
        LiteLLMModel(config_cur_agent["model_name"], api_base=config_cur_agent["url"], api_key=config_cur_agent["api_key"])
        if "groq.com" in config_cur_agent["url"]
        else OpenAIServerModel(api_base=config_cur_agent["url"], model_id=config_cur_agent["model_name"], api_key=config_cur_agent["api_key"])
    )

    agent = CodeAgent(
        #tools=coder_tools,
        tools = [],
        model=model,
        additional_authorized_imports=["*"],
    )

    agent_input = coder_prompt.format(directory=os.path.join(ROOT_DIR, os.environ['DS_STORAGE_PATH']), task=task)
    response = agent.run(agent_input)

    #TODO manage generated file here

    return Command(update={
        "past_steps": Annotated[set, operator.or_](set([(task, str(response))])),
        "nodes_calls": Annotated[set, operator.or_](set([
            ("ml_dl_agent", (("text", str(response)),))
        ])),
    })


def ml_dl_agent(state: dict, config: dict) -> Command:
    """
    Executes a machine learning/deep learning agent to address a given task, leveraging a large language model.
    
    This method instantiates an agent using a specified LLM, configured with credentials and settings from the provided configuration. 
    It then utilizes this agent to process the task described in the input state and generates a response. The method is designed to automate 
    complex tasks requiring intelligent reasoning and code execution to arrive at a solution.
    
    All tools are client functions that can launch training of models (ML-models or transformer model) or call inference.
    
    Args:
        state (dict): A dictionary containing the task to be performed, accessible via the "task" key.
        config (dict): A dictionary containing configuration details, including LLM credentials (api_key, url) and agent-specific 
            settings found under `config["configurable"]["additional_agents_info"]["ml_dl_agent"]`.
    
    Returns:
        Command: A command object containing the agent's textual response, the updated task history (`past_steps`) including the current task and response,
            and a record of the agent call (`nodes_calls`) detailing the agent used and its input/output.
    """
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
    """
    Executes a chemistry-related task using a language model and specialized tools.
    
    This method takes a task and a plan as input, and uses a Chemist agent—configured 
    with a language model and chemistry tools—to attempt to complete the task. It handles potential errors during execution 
    by retrying the task up to three times with increasing delays. 
    The agent's reasoning and results are recorded for tracking progress.
    
    Args:
        state (dict): A dictionary containing the current task (key: "task") and plan (key: "plan").
        config (dict): A dictionary containing configuration details, including access to the language model (LLM) and other settings (accessible under 'configurable').
    
    Returns:
        Command: A `Command` object.  If successful, it contains the executed task and associated details (`past_steps`, `nodes_calls`). If the task fails after multiple attempts, it returns a `Command` object with a failure message in the `response` field.
    """
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
    """
    Executes a task using a nanoparticle agent and returns a Command object.
    
    This method leverages a ReAct agent to process a given task and plan, providing a structured response
    for integration into a larger workflow. It includes error handling with retries to ensure robustness.
    
    Args:
        state (dict): A dictionary containing the current task and plan.  The 'task' key holds the
                      description of the work to be done, and 'plan' outlines the steps to achieve it.
        config (dict): A dictionary containing configuration details, including the LLM to be used
                       within the ReAct agent ('configurable' -> 'llm').
    
    Returns:
        Command: A Command object containing the results of the agent's execution.  On success,
                 'past_steps' and 'nodes_calls' are updated to reflect the completed task and agent interactions.
                 If the task fails after multiple retries, a 'response' message indicating failure is returned.
    """
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
    Analyzes scientific papers to answer user questions.
    
    This agent utilizes a combination of a vector database of chemical papers and user-uploaded documents
    to provide informed responses. It attempts to extract relevant information and synthesize answers.
    
    Args:
        state (dict): The current state of the interaction, including the user's task.
        config (dict): Configuration settings, including the language model to use.
    
    Returns:
        Command: An object containing the next step in the process ('replan' or `END`) and
        updates to the state, including recorded steps, responses, and extracted metadata.
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
        current_prompt = f'{paper_agent_prompt}\n session_id = {st.session_state.session_id}'
    except:
        current_prompt = f'{paper_agent_prompt}\nsession_id is not needed in this case, pass None'

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
