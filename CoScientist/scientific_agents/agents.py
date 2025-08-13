from smolagents import CodeAgent, OpenAIServerModel
from smolagents import DuckDuckGoSearchTool
from smolagents import LiteLLMModel

from langgraph.types import Command
from typing import Annotated


def coder_agent(state: dict, config: dict):
    """
    Initializes and runs a code generation agent to fulfill a given task.
    
    This method instantiates a code generation agent using llm, based on the configuration provided. 
    It constructs a prompt for the agent that outlines available code libraries and 
    the specific task at hand. The agent then executes, generating code to address the 
    task, and the result is encapsulated in a Command object. This facilitates the 
    automated creation of code components based on project requirements.
    
    Args:
        state (dict): A dictionary representing the current state of the process, including the plan containing the task to be executed.
        config (dict): A dictionary containing configuration settings, including the agent's URL, API key, model name, 
        and the directory where generated files should be saved.
    
    Returns:
        Command: A Command object that contains the record of the agent's call (nodes_calls) and the task with its generated response (past_steps).
    """
    config_cur_agent = config["configurable"]["additional_agents_info"]["coder_agent"]
    plan = state["plan"]
    task = plan[0]

    if 'groq.com' in config_cur_agent["url"]:
        model = LiteLLMModel(
            config_cur_agent["model_name"],
            api_base=config_cur_agent["url"],
            api_key=config_cur_agent["api_key"]
        )
    else:
        model = OpenAIServerModel(
            api_base=config_cur_agent["url"],
            model_id=config_cur_agent["model_name"],
            api_key=config_cur_agent["api_key"],
        )
    agent = CodeAgent(
        tools=[DuckDuckGoSearchTool()], model=model, additional_authorized_imports=["*"]
    )
    main_prompt = (
        "To generate code you have access to libraries: 're', 'rdkit', \
    'smolagents', 'math', 'stat', 'datetime', 'os', 'time', 'requests', 'queue', \
    'random', 'bs4', 'rdkit.Chem', 'unicodedata', 'itertools', 'statistics', 'pubchempy',\
    'rdkit.Chem.Draw', 'collections', 'numpy', 'rdkit.Chem.Descriptors', 'sklearn', 'pickle', 'joblib'. \
    Attention!!! Directory for saving files: "
        + config_cur_agent["ds_dir"]
    )
    response = agent.run(main_prompt + "\n" + task)

    return Command(update={
        "nodes_calls": Annotated[list, "accumulate"]([("coder_agent", str(response))]),
        "past_steps": Annotated[list, "accumulate"]([(task, str(response))]),
    })
