import ast
import re
from typing import Dict
import requests
from sklearn.metrics.pairwise import cosine_similarity

from langchain.tools.render import render_text_description
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from MedCoScientist.agents.tools_prompts import (
    extract_concepts_prompt,
    extract_keywords_prompt,
    masked_words_saliency_prompt
)


def _process_list(text: str) -> list:
    return [item.strip() for item in text.split(",") if item.strip()]

def _process_string_to_dict(text: str) -> dict:
    start_index = text.find('{')
    end_index = text.rfind('}') + 1

    dict_str = text[start_index:end_index].strip()

    return ast.literal_eval(dict_str)

def _mask_text(text: str, words_to_mask: list) -> str:
    """
    Заменяет все слова из списка на [MASKED] в исходном тексте.
    Регистронезависимо, заменяет только целые слова.
    """
    pattern = r'\b(' + '|'.join(map(re.escape, words_to_mask)) + r')\b'
    
    masked_text = re.sub(pattern, '[MASKED]', text, flags=re.IGNORECASE)
    return masked_text

def _get_embeddings(embedder_config, texts):
    api_key = embedder_config["api_key"]
    response = requests.post(
        embedder_config["base_url"],
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": embedder_config["model"],
            "input": texts
        }
    )
    data = response.json()
    return [datum["embedding"] for datum in data["data"]]

def _has_values(dictionary, key):
    value = dictionary.get(key, None)
    return value is not None and len(value) > 0

# @ tool
def extract_keywords_node(hypothesis: str, llm: BaseChatModel, embedder_config: Dict[str, str]) -> str:
    """
    Extracts keywords from the hypothesis by saliency.
    
    This method utilizes a language model to evaluate the saliency.
    
    Args:
        hypothesis (str): A textual formulation of the hypothesis.
        config (RunnableConfig): Configuration object containing the language model to be used for saliency evaluation.
    
    Returns:
        keywords (str): A string containing the list of dectected keywords separated with a comma. May return an error message if prediction fails.
    """
    try:
        key_concepts = (extract_concepts_prompt | llm).invoke(hypothesis).content
        key_words = (extract_keywords_prompt | llm).invoke({"hypothesis": hypothesis, "key_concepts": key_concepts}).content

        key_concepts = _process_list(key_concepts)
        key_words = _process_string_to_dict(key_words)

        # leave only concepts with key words
        key_concepts = list(filter(lambda concept: _has_values(key_words, concept), key_concepts))

        # get masked texts
        chain = masked_words_saliency_prompt | llm
        restored_texts = {}
        for concept in key_concepts:
            masked_text = _mask_text(text=hypothesis, words_to_mask=key_words[concept])
            restored_texts[concept] = chain.invoke(masked_text).content
        
        # compute embedding similarities
        texts_to_embed = list(restored_texts.values()) + [hypothesis]
        embeddings = _get_embeddings(embedder_config, texts_to_embed)
        similarities = [cosine_similarity([embedding], [embeddings[-1]])[0, 0] for embedding in embeddings[:-1]]

        # select top-3 highest influence key words
        k = 3
        top_k = sorted(similarities)[k-1]
        selected_key_words = []
        for (concept, similarity) in zip(key_concepts, similarities):
            if similarity < top_k:
                selected_key_words.extend(key_words[concept])
        return ", ".join(selected_key_words)
    except Exception as e:
        return f"I couldn't extract keywords because of: {str(e)}, I should move to the next task if any"

# keywords_tools = [
#     extract_keywords_node,
# ]

# keywords_tools_rendered = render_text_description(keywords_tools)
