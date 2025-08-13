
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.exceptions import OutputParserException
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.graph import END
from langgraph.prebuilt import create_react_agent

import subprocess
import json
import re
import os
from pathlib import Path

from graph.states import PlanExecute
from prompts import worker_prompt, memory_prompt, chat_prompt, chat_parser, supervisor_prompt, supervisor_parser, summary_prompt, replanner_prompt, replanner_parser, planner_parser, planner_prompt, translate_prompt, translator_parser, retranslate_prompt
from tools import chem_tools, web_tools, nanoparticle_tools
from pydantic_models import Response

import time
from typing import List
import logging

# Create a separate logger for nodes.py
logger = logging.getLogger("node_logger")
logger.setLevel(logging.INFO)

# Configure a file handler for the node logger
file_handler = logging.FileHandler("node.log")
file_handler.setLevel(logging.INFO)

# Set a formatter for the node logger
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the node logger
logger.addHandler(file_handler)




def in_translator_node(state: PlanExecute, config: dict):
    """
    Translates the input text to English using an LLM agent.
    
    This method attempts to translate the input text to English to ensure compatibility with downstream processing steps. It includes retry logic with exponential backoff to handle potential API errors or temporary service unavailability. It also explicitly handles common error scenarios like invalid API keys and unavailable LLM services.
    
    Args:
        state (PlanExecute): The current state of the plan execution, containing the input text.
        config (dict): A dictionary containing configuration details, including the LLM model to use for translation.
    
    Returns:
        Command: A command object indicating the next step.
            - If the input is already in English, the command transitions to the 'chat' state, updating the language to 'English'.
            - If the translation is successful, the command transitions to the 'chat' state, updating the language and translation fields in the state.
            - If an API key error occurs, the command transitions to the 'END' state with an error message.
            - If the LLM is unavailable (HTTP 404 error), the command transitions to the 'END' state with an error message.
    """

    llm: BaseChatModel = config["configurable"]["model"]
    translator_agent = translate_prompt | llm | translator_parser


    input: str = state["input"]
    max_retries: int = 3


    for attempt in range(max_retries):
        try:
            output =  translator_agent.invoke(input)

            if output.language =='English':
                return Command(
                    goto = 'chat',
                    update= {'language': 'English'}
                )
            else:
                return Command(
                    goto='chat',
                    update={'language': output.language, 'translation': output.translation}
                )
            
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"InTranslator failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\n State: {state}")
            print(f"InTranslator failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            if 'api' in str(e).lower() and 'key' in str(e).lower():
                return Command(
                    goto=END,
                    update={"response": "Your api key is invalid"}
                )
            if '404' in str(e):
                return Command(
                    goto=END,
                    update={"response": "LLM is unavailabe right now, perhaps you should check your proxy"}
                )
            time.sleep(2**attempt)  # Exponential backoff
    

def re_translator_node(state: PlanExecute, config: dict):
    """
    Re-translates the response to English if it's not already in that language.
    
    This method ensures consistent processing by translating any non-English responses 
    into English using a translator agent. It incorporates retry logic with exponential 
    backoff to handle potential translation errors. If translation repeatedly fails,
    a predefined message is returned, indicating an inability to process the response.
    
    Args:
        state (dict): A dictionary containing the current state of the plan execution, 
                      including the 'response' (the text to potentially translate) 
                      and 'language' (the detected language of the response).
        config (dict): A dictionary containing the configuration for the agent, 
                       specifically the language model to be used for translation.
    
    Returns:
        Command: A Command object. If the translation is successful or the input is 
                 already in English, the 'update' field contains the (potentially translated) 
                 response, and 'goto' is set to END.  If translation fails after multiple 
                 attempts, the 'update' field contains a failure message, and 'goto' is END.
    """
    input: str = state["response"]
    language: str = state['language']

    llm: BaseChatModel = config["configurable"]["model"]
    #memorize = memory_prompt | llm
    max_retries: int = 3

    if language == 'English':
        return Command(
            goto=END,
            update = {'response': input}
        )

    translator_agent = retranslate_prompt | llm 

    for attempt in range(max_retries):
        try:
            output =  translator_agent.invoke({"input": input, "language": language})
            return Command(
                goto=END,
                update = {'response': output.content}
            )
            
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"ReTranslator failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\n State: {state}")
            print(f"ReTranslator failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )


def chat_node(state: PlanExecute, config: dict):
    """
    Processes input text using a chat agent to determine the next action in a plan execution workflow.
    
    This method utilizes a chat agent composed of a chat prompt, a language model, and a chat parser to analyze input and produce a response or suggest a subsequent step. It incorporates a retry mechanism with exponential backoff to handle potential errors during the interaction.  The visualization component is cleared with each interaction to ensure a fresh output.
    
    Args:
        state (PlanExecute): The current state of the planning and execution process, including input text and language.
        config (dict): A dictionary containing configuration settings, especially the language model to be used.
    
    Returns:
        dict: A dictionary representing the chat agent's output.  Possible structures include:
            - {"response": str, 'visualization': None}:  If the agent provides a direct response to the input.
            - {"next": str, 'visualization': None}: If the agent suggests the next action or step.
        Command: If all retry attempts fail, a Command is returned to navigate back to the planner with no response.
    """

    llm: BaseChatModel = config["configurable"]["model"]
    chat_agent = chat_prompt | llm | chat_parser    
    input: str = state["input"] if state.get('language', 'English') == 'English' else state['translation']
    max_retries: int = 3


    for attempt in range(max_retries):
        try:
            output =  chat_agent.invoke(input)

            if isinstance(output.action, Response):
                return {"response": output.action.response, 'visualization': None} #we're setting visualization here to None to delete all previosly generated visualizatons
            else:
                return {"next": output.action.next, 'visualization': None}
            
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Chat failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Chat failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto='planner',
        update={"response": None}
    )

def should_end_chat(state: PlanExecute):
    """
    Checks if a response is available to determine whether to continue planning or transition to retranslation.
    
    Args:
      state (PlanExecute): The current state of the plan execution, expected to contain a 'response' key.
    
    Returns:
      str: 'retranslator' if a non-empty response is present in the state, indicating a result is available for presentation; otherwise, 'planner' to continue the planning phase.
    """
    if "response" in state and state["response"]:
        return 'retranslator'
    else:
        return "planner"


def supervisor_node(state: PlanExecute, config: dict):
    """
    Executes a step from a predefined plan using a language model.
    
    This method retrieves the next task from the plan, formats it for the language model,
    and attempts to execute it. It includes retry logic with exponential backoff to handle
    potential errors during the language model invocation.  The method aims to sequentially
    address parts of a complex request by breaking it down into manageable steps.
    
    Args:
        state (PlanExecute): The current state of the plan execution, containing the plan.
        config (dict): A dictionary containing configuration details, including the language model.
    
    Returns:
        Command: A command object indicating the next step ('goto') and potentially updates 
                 to the state ('update') based on the language model's response.  If the plan
                 is empty or an error occurs after multiple retries, a final command 
                 indicating inability to proceed is returned.
    """

    llm: BaseChatModel = config["configurable"]["model"]
    supervisor = supervisor_prompt | llm | supervisor_parser


    plan: list = state.get("plan")
    if plan is None and not state.get('input'):
        return Command(
            goto=END,
            update="I've couldn't answer to your question, could you ask me once more?-><-"
        )
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task: str = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""
    
    max_retries: int = 3


    for attempt in range(max_retries):
        try:
            agent_response = supervisor.invoke({"input": [("user", task_formatted)]})

            return Command(
                goto=agent_response.next,
                update={"next": agent_response.next}
            )
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Supervisor failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Supervisor failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )



def chemist_node(state: PlanExecute, config: dict):
    """
    Executes a single step of a research plan, leveraging a chemistry-focused agent to perform a specific task.
    
    This method takes the current state and configuration, formulates a task based on the plan, 
    and uses a language model agent to execute it. It then returns a command to either continue 
    the plan with the results or to end the process if the task cannot be completed after several attempts.
    
    Args:
        state (PlanExecute): The current state of the plan execution, containing the plan and other relevant data.
        config (dict): A dictionary containing configuration details, including the language model and tools.
    
    Returns:
        Command: A command indicating the next step.  If the task is successfully executed, it returns a 
            "replan" command with updated `past_steps` and `nodes_calls`. If all retries fail, it returns an 
            "END" command with a failure message.
    """

    llm: BaseChatModel = config["configurable"]["model"]
    chem_agent = create_react_agent(llm, chem_tools, state_modifier=worker_prompt + 'admet = qed')

    plan: list = state["plan"]
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task: str = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            config['configurable']['state'] = state #to get tool_call_id, but this shouldn't be implemented like that
            agent_response = chem_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(
                goto = 'replan',
                update = {'past_steps':[(task, agent_response["messages"][-1].content)],
                          'nodes_calls': [('chemist_node', agent_response["messages"])]}
            )

        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Chemist failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Chemist failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )

def nanoparticle_node(state: PlanExecute, config: dict):
    """
    Executes a single step of a plan using a specialized agent designed for nanoparticle-related tasks.
    
    This method attempts to execute the most immediate task from a given plan, leveraging a pre-configured agent and a set of tools. It handles potential errors during execution with retries and provides updates to the plan's state based on the agent's response.
    
    Args:
        state (PlanExecute): The current state of the plan execution, including the plan itself.
        config (dict): A dictionary containing configuration settings, including the LLM model and available tools.
    
    Returns:
        Command: A command object indicating the next step.
            - If successful, the command directs to 'replan' with the updated plan state, including the executed task and the agent's response.
            - If all retries fail, the command directs to 'END' with a failure message.
    """

    llm: BaseChatModel = config["configurable"]["model"]
    #add_prompt = 'if you are asked to predict nanoparticle shape, directly call corresponding tool'
    add_prompt = 'You have to respond with results of tool call, do not repharse it'
    nanoparticle_agent = create_react_agent(llm, nanoparticle_tools, state_modifier=worker_prompt + add_prompt)

    plan: list = state["plan"]
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task: str = plan[0]
    task_formatted = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            agent_response = nanoparticle_agent.invoke({"messages": [("user", task_formatted)]})

            return Command(
                goto = 'replan',
                update = {'past_steps':[(task, agent_response["messages"][-1].content)],
                          'nodes_calls': [('nanoparticle_node', agent_response["messages"])]}
            )

        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Nanoparticle failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Nanoparticle failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )
    
def automl_node(state: PlanExecute, config: dict):
    """
    Executes an AutoML process using a separate Python environment.
    
    This method initiates an automated machine learning process on a provided dataset.
    It leverages an external script to perform the AutoML task, handles potential dataset issues, 
    and updates the execution state with the results, allowing for subsequent analysis or replanning if necessary.
    It's used to generate insights from the input data to support answering complex queries.
    
    Args:
        state (PlanExecute): The current state of the plan execution, containing the execution plan and intermediate results.
        config (dict): A dictionary containing configuration details for the AutoML process,
            including the path to the dataset directory and AutoML-specific settings.
    
    Returns:
        Command: A command object indicating the next step in the plan execution. 
            This typically involves replanning based on the AutoML results, 
            and updates the state with either the AutoML outcomes or error messages.
    """
    #NOTE: here we call separate venv. Also automl_invoke.py uses state[input], not task

    script_path = "graph/automl_invoke.py"  # Adjust to actual location
    python_bin = "venv310/bin/python3.10"  # Ensure this points to Python 3.10 binary TODO: get it via environ

    plan: list = state["plan"]
    task: str = plan[0]

    dataset_dir_path = Path(config['configurable']['fedot_config']['user_data_dir']).resolve()
    if len(os.listdir(dataset_dir_path)) == 0:
        response_text = 'There is no files of dataset to use. Do not call me again'
        return Command(
            goto='replan',
            update={"past_steps": [(task, response_text)]}
        )
    
    try:
        result = subprocess.run(
            [python_bin, script_path, json.dumps(config['configurable']['fedot_config']), json.dumps(state)],
            capture_output=True, text=True
        )

        re_match = re.search(r'\{.*\}', result.stdout, re.DOTALL)
        response_text = json.loads(re_match.group(0).encode('utf-16', 'surrogatepass').decode('utf-16'))['response']

        #response_text = 'I have done automl job, results are saved'
        return Command(
            goto='replan',
            update={"past_steps": [(task, response_text)],
                    'automl_results': response_text}
        )
    
    except json.JSONDecodeError as e:
        response_text = "I've couldn't do automl job, don't call me again"
        return Command(
            goto='replan',
            update={"past_steps": [(task, response_text)]}
        )
    except Exception as e:
        logger.exception(f"automl failed with error: {str(e)}.\tState: {state}")
        return Command(
            goto=END,
            update={"response": f"I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
        )

def web_search_node(state: PlanExecute, config: dict):
    """
    Executes the first step of a plan by performing a web search and retrieving information.
    
    Args:
        state (PlanExecute): The current state of the plan execution, including the plan itself.
        config (dict): A dictionary containing configuration settings, including the language model.
    
    Returns:
        Command: A command object that either replans with the results of the web search,
            or signals an unrecoverable error if the web search fails after multiple retries.
            The command includes updated state information such as completed steps and tool calls.
    """

    llm: BaseChatModel = config["configurable"]["model"]
    if not web_tools:
        web_agent = create_react_agent(llm, [], state_modifier=worker_prompt)
    else:
        web_agent = create_react_agent(llm, web_tools, state_modifier=worker_prompt)

    plan: list = state["plan"]
    plan_str: str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))

    task: str = plan[0]
    task_formatted: str = f"""For the following plan:
    {plan_str}\n\nYou are tasked with executing: {task}."""
    
    max_retries: int = 3
    for attempt in range(max_retries):
        try:
            agent_response = web_agent.invoke({"messages": [("user", task_formatted)]})
            return Command(
                goto = 'replan',
                update = {'past_steps':[(task, agent_response["messages"][-1].content)],
                          'nodes_calls': [('web_search_node', agent_response["messages"])]}
            )
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Web_searcher failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Web_searcher failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )

def plan_node(state: PlanExecute, config):
    """
    Plans the next node in the execution flow by leveraging a language model.
    
    This method generates a plan based on the current state and configuration, 
    interpreting the language model's output to determine subsequent steps. 
    It includes error handling to robustly extract plan details even from imperfect responses,
    and gracefully handles repeated failures by concluding the process.
    
    Args:
        state (PlanExecute): The current state of the plan execution, including input text
                             (either in English or translated).
        config (dict): A dictionary containing configuration settings, 
                       including the language model and planner components.
    
    Returns:
        dict: A dictionary containing the plan steps if successful, 
              with the key "plan" mapping to a list of steps.
        Command: A command object signaling the end of the execution if planning fails 
                 after multiple retries, and includes a message indicating inability to proceed.
    """

    llm: BaseChatModel = config["configurable"]["model"]
    planner = planner_prompt | llm | planner_parser
    
    max_retries: int = 3
    input: str = state["input"] if state['language'] == 'English' else state['translation']

    for attempt in range(max_retries):
        try:
            plan = planner.invoke({"messages": [("user", input)]})
            return {"plan": plan.steps}
        except OutputParserException as  e:
            match = re.search(r'\{\s*"steps"\s*:\s*\[\s*(?:"[^"]*"\s*,\s*)*"[^"]*"\s*\]\s*\}', str(e), re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str.replace("\\", "\\\\")
                try:
                    structured_output = json.loads(json_str)
                    return {"plan": structured_output['steps']}
                except json.JSONDecodeError as json_err:
                    logger.exception(f"Planner failed with error: {str(json_err), json_str}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
                    print(f"Planner failed with error: {str(json_err)}. Retrying... ({attempt+1}/{max_retries})")
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Planner failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"Planner failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )

def replan_node(state: PlanExecute, config: dict):
    """
    Replans a node in the execution plan based on the current state and configuration, attempting to recover from errors or refine the plan.
    
    This method utilizes a language model to re-evaluate the current plan and input, aiming to generate either a direct response or a revised plan. It includes retry logic with exponential backoff to handle potential failures during the LLM call. If replanning consistently fails, it returns a default response indicating an inability to answer.
    
    Args:
        state (PlanExecute): The current state of the plan execution, containing information such as the input, current plan, and past steps.
        config (dict): A dictionary containing configuration parameters, including the language model to be used for replanning.
    
    Returns:
        dict: A dictionary with one of the following keys:
            - "response": If the replanning results in a direct answer, this key holds the response string.
            - "plan": If the replanning results in an updated plan, this key holds the list of updated plan steps.
        Command: If replanning fails after multiple retries, a Command object is returned with a pre-defined message.
    """

    llm: BaseChatModel = config["configurable"]["model"]
    replanner = replanner_prompt | llm | replanner_parser

    input: str = state["input"] if state['language'] == 'English' else state['translation']
    max_retries: int = 3
    for attempt in range(max_retries):
        try:
            output =  replanner.invoke({'input': input, 'plan': state['plan'], 'past_steps': state['past_steps']})
            if isinstance(output.action, Response):
                return {"response": output.action.response}
            else:
                return {"plan": output.action.steps}
            
        except OutputParserException as  e:
            match = re.search(r'\{\s*"steps"\s*:\s*\[\s*(?:"[^"]*"\s*,\s*)*"[^"]*"\s*\]\s*\}', str(e), re.DOTALL)
            if match:
                json_str = match.group(0)
                json_str.replace("\\", "\\\\")
                try:
                    structured_output = json.loads(json_str)
                    return {"plan": structured_output['steps']}
                except json.JSONDecodeError as json_err:
                    logger.exception(f"Planner failed with error: {str(json_err), json_str}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
                    print(f"RePlanner failed with error: {str(json_err)}. Retrying... ({attempt+1}/{max_retries})")
            
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"RePlanner failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\t State: {state}")
            print(f"RePlanner failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )

def should_end(state: PlanExecute):
    """
    Determines the next step in processing based on whether a relevant response has been found.
    
    Args:
        state (PlanExecute): The current state of the plan execution, potentially containing a response.
    
    Returns:
        str: 'summary' if a response is present and contains data; otherwise, 'supervisor' to request further action.
    """
    if "response" in state and state["response"]:
        return 'summary'
    else:
        return "supervisor"
    
def summary_node(state: PlanExecute, config: dict):
    """
    Generates a concise summary of the interaction between a user query and a system response, incorporating the history of previous steps.
    
    This method uses a language model to distill the key information from the current interaction and its context, providing a focused overview of the progress. It includes robust error handling with retry logic to mitigate temporary issues with the language model.  If summarization repeatedly fails, a default message is returned to the user.
    
    Args:
        state (dict): A dictionary containing the interaction details, including the user's 'input' (or its translation), the system's 'response', and a list of 'past_steps' representing the previous interactions.
        config (dict): A dictionary containing configuration settings, including access to the language model instance via `config["configurable"]["model"]`.
    
    Returns:
        dict: A dictionary containing the generated summary in the 'response' key.  If the summarization process fails after multiple retries, returns a `Command` object with a predefined error message indicating temporary unavailability.
    """
    system_response: str = state["response"]
    query: str = state["input"] if state.get('language', 'English') == 'English' else state['translation']
    past_steps: list = state["past_steps"]
    llm: BaseChatModel = config["configurable"]["model"]
    #memorize = memory_prompt | llm
    max_retries: int = 3

    summary_agent = summary_prompt | llm 

    for attempt in range(max_retries):
        try:
            output =  summary_agent.invoke({'query': query, 'system_response': system_response, 'intermediate_thoughts': past_steps})
            return {'response': output.content}
            
        except Exception as e:  # Handle OpenAI API errors
            logger.exception(f"Summary failed with error: {str(e)}.\t Retrying... ({attempt+1}/{max_retries})\n State: {state}")
            print(f"Summary failed with error: {str(e)}. Retrying... ({attempt+1}/{max_retries})")
            time.sleep(2**attempt)  # Exponential backoff

    return Command(
        goto=END,
        update={"response": "I can't answer to your question right now( Perhaps there is something else that I can help? -><-"}
    )