import logging
import os

from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from graph.nodes import (chat_node, chemist_node, in_translator_node,
                         nanoparticle_node, plan_node, re_translator_node,
                         replan_node, should_end, should_end_chat,
                         summary_node, supervisor_node, web_search_node,
                         automl_node)
from graph.states import PlanExecute

# Create a separate logger for nodes.py
logger = logging.getLogger("graph_logger")
logger.setLevel(logging.INFO)

# Configure a file handler for the node logger
file_handler = logging.FileHandler("graph.log")
file_handler.setLevel(logging.INFO)

# Set a formatter for the node logger
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the node logger
logger.addHandler(file_handler)


workflow = StateGraph(PlanExecute)

# Add the in_translator node
workflow.add_node("intranslator", in_translator_node)
# Add the chat node
workflow.add_node("chat", chat_node)
# Add the plan node
workflow.add_node("planner", plan_node)
# Add the supervisor node
workflow.add_node("supervisor", supervisor_node)
# Add the chemist step
workflow.add_node("chemist", chemist_node)
# Add the nanoparticle step
workflow.add_node("nanoparticles", nanoparticle_node)
# Add the web_search step
workflow.add_node("web_search", web_search_node)
# Add a replan node
workflow.add_node("replan", replan_node)
# Add the out_translator node
workflow.add_node("summary", summary_node)
# Add the out_translator node
workflow.add_node("retranslator", re_translator_node)
workflow.add_node("automl", automl_node)


workflow.add_edge(START, "intranslator")
workflow.add_conditional_edges(
    "chat",
    # Next, we pass in the function that will determine which node is called next.
    should_end_chat,
    ["planner", 'retranslator'],
)
# From plan we go to supervisor
workflow.add_edge("planner", "supervisor")

workflow.add_conditional_edges(
    "replan",
    # Next, we pass in the function that will determine which node is called next.
    should_end,
    ["supervisor", 'summary'],
)
workflow.add_edge("summary", "retranslator")
# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
app = workflow.compile()

class App:
    """
    App class for interacting with a remote service.
    
        This class provides methods for invoking remote functions and streaming data
        to and from a remote service. It supports both synchronous and asynchronous
        operations.
    
        Attributes:
        - url: The URL of the remote service.
        - token: Authentication/API token for the service.
        - timeout: Timeout in seconds for requests.
    """

    def __init__(self, main_model_name: str, visual_model_name: str, fedot_model_name: str,  base_url: str, api_key: str, tavily_api_key:str = None):
        """
        Initializes the ChemicalChatBot with necessary models and API keys.
        
        Args:
            main_model_name (str): The name of the main language model to use for general conversation.
            visual_model_name (str): The name of the visual language model to use for analyzing images.
            fedot_model_name (str): The name of the Fedot model used for chemical structure processing.
            base_url (str): The base URL for the OpenAI API.
            api_key (str): The API key for the OpenAI API.
            tavily_api_key (str, optional): The API key for the Tavily search API. Defaults to None.
        
        Initializes the following fields:
            llm: ChatOpenAI model instance for general conversation.
            visual_llm: ChatOpenAI model instance for visual tasks.
            fedot_model: Name of the Fedot model being used.
            api_key: The API key used for OpenAI.
            base_url: The base URL used for API requests.
            app: The application instance.
        
        Returns:
            None
        
        This initialization prepares the chatbot to respond to user queries, incorporating both general knowledge from large language models and specialized chemical information from the Fedot model, and image analysis capabilities.
        """
        self.llm = ChatOpenAI(model=main_model_name,
                        base_url=base_url,
                        api_key=api_key,
                        temperature=0.7,
                        default_headers={"x-title": "ChemicalChatBot"})

        self.visual_llm = ChatOpenAI(model=visual_model_name,
                        base_url=base_url,
                        api_key=api_key,
                        temperature=1.0,
                        default_headers={"x-title": "ChemicalChatBot"})
        
        self.fedot_model = fedot_model_name
        self.api_key = api_key
        self.base_url = base_url

        if tavily_api_key:
            os.environ['TAVILY_API_KEY'] = tavily_api_key
        self.app = app

    def invoke(self, input: dict, config: RunnableConfig):
        """
        Invokes the application to process input data and generate a response.
        
        This method prepares the application by integrating large language models (LLMs) and a specialized scientific model (Fedot). 
        It configures the environment with necessary API keys and user data directories, then streams the processing of the input through the application.
        The final result from this stream is returned.
        
        Args:
            self: The instance of the App class.
            input (dict): The input data for the application to process.
            config (RunnableConfig): The application configuration object.
        
        Returns:
            The final value generated by the application stream, representing the processed result.
        """
        config['configurable']['model'] = self.llm
        config['configurable']['visual_model'] = self.visual_llm

        user_data_dir = config['configurable'].get('user_data_dir')
        fedot_config = {
                        'user_data_dir': user_data_dir,
                        "model_name": self.fedot_model,
                        'openai_api_base': self.base_url,
                        'openai_api_key': self.api_key
                        }
        
        config['configurable']['fedot_config'] = fedot_config
        
        logger.info(f"\n\nINPUT: {input}")
        for event in app.stream(input=input, config=config):
            for k, v in event.items():
                if k != "__end__":
                   logger.info(v)
        #return self.app.invoke(input=input, config=config)
        return v
    
    def stream(self, input: dict, config: RunnableConfig):
        """
        Streams data through the LLM and Fedot model for processing and analysis.
        
        This method prepares the configuration with the necessary models (LLM, visual LLM, and Fedot) and user data directory, then processes the input data using the application's stream function. Intermediate results are logged for monitoring and debugging.
        
        Args:
            self: The instance of the class.
            input (dict): The input data to be processed.  This can include data from scientific papers or user queries.
            config (RunnableConfig): The configuration object containing application settings.
        
        Returns:
            generator: A generator that yields the processed data as it becomes available.
        """
        config['configurable']['model'] = self.llm
        config['configurable']['visual_model'] = self.visual_llm
        
        user_data_dir = config['configurable'].get('user_data_dir')
        fedot_config = {
                        'user_data_dir': user_data_dir,
                        "model_name": self.fedot_model,
                        'openai_api_base': self.base_url,
                        'openai_api_key': self.api_key
                        }
        
        config['configurable']['fedot_config'] = fedot_config
        logger.info(f"\n\nINPUT: {input}")
        for event in app.stream(input=input, config=config):
            for k, v in event.items():
                if k != "__end__":
                   logger.info(v)
                yield v

    async def ainvoke(self, input: dict, config: RunnableConfig):
        """
        Invokes the application with updated language and vision models.
        
        Updates the configuration to use the current instance's language and vision models
        before passing it to the underlying application's ainvoke method. This ensures
        the application uses the correct models for processing the input.
        
        Args:
            input (dict): The input data for the ainvoke call.
            config (RunnableConfig): The configuration object for the ainvoke call.
        
        Returns:
            The result returned by the application's ainvoke method.
        """
        config['configurable']['model'] = self.llm
        config['configurable']['visual_model'] = self.visual_llm
        return self.app.ainvoke(input=input, config=config)

    async def astream(self, input: dict, config: RunnableConfig):
        """
        Streams the input through the application, configuring it with the project's language models.
        
        Args:
            input (dict): The input data for processing.
            config (RunnableConfig): The configuration settings for the application.
        
        Returns:
            The streamed output from the application.
        """
        config['configurable']['model'] = self.llm
        config['configurable']['visual_model'] = self.visual_llm
        return self.app.astream(input=input, config=config)

