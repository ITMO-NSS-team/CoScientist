import json

from langchain.tools.render import render_text_description
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from MedCoScientist.agents.tools_prompts import pico_prompt, pico_examples


def _check_pico_dict(json_string):
    data_dict = json.loads(json_string)
    keys = sorted(data_dict.keys())
    assert keys == ['comparison', 'intervention', 'outcome', 'population'], f'Wrong keys: {keys}'
    for key, value in data_dict.items():
        assert isinstance(value, str), f"Value `{value}` corresponding to the key `{key}` is not a string"

# @tool
def extract_pico_node(text: str, llm: BaseChatModel) -> str:
    """
    Extracts PICO decomposition from a hypothesis or a paper abstract.
    
    Args:
        text (str): The text of a hypothesis or a paper abstract.
        config (RunnableConfig): Configuration object containing the language model to be used for saliency evaluation.
    
    Returns:
        str: string representation of a JSON with PICO decomposition. May return an error message if prediction fails.
    """
    try:
        # inference(shape)
        chain = pico_prompt | llm
        result = chain.invoke({'examples': pico_examples,'abstract': text}).content
        _check_pico_dict(result)
        return result
    except Exception as e:
        return f"I couldn't extract PICO elements because of: {str(e)}, I should move to the next task if any"

# pico_tools = [
#     extract_pico_node,
# ]

# pico_tools_rendered = render_text_description(pico_tools)
