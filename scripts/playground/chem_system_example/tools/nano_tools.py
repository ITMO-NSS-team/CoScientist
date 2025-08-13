from prompts.tools_prompts import properties_prediction_prompt, shape_detection_prompt, synt_prompt, entr_eff_prompt
from tools.models.generative_inference import inference

from langchain_core.tools import tool
from langchain.tools.render import render_text_description
from langchain_core.runnables.config import RunnableConfig
from langchain_core.language_models.chat_models import BaseChatModel

import requests
import json
import time
import re
import logging

# Create a separate logger for tools.py
logger = logging.getLogger("tools_logger")
logger.setLevel(logging.INFO)

# Configure a file handler for the tools logger
file_handler = logging.FileHandler("tools.log")
file_handler.setLevel(logging.INFO)

# Set a formatter for the tools logger
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add the handler to the tools logger
logger.addHandler(file_handler)


def call_for_generation(
    synthesis_generator_system_prompt,
    properties_input,
    synthesis_generator_alpaca_prompt,
    url: str = "http://10.32.2.5:82/call",
    max_attemps = 3,
    **kwargs,
):
    """
    Sends a request to a synthesis generator service to obtain relevant data for scientific inquiries.
    
    This method packages input parameters and repeatedly calls a remote service, handling potential connection errors with retries.
    It aims to provide a robust mechanism for retrieving information required for answering questions based on scientific content.
    
    Args:
        synthesis_generator_system_prompt: The overall instruction or context for the generator.
        properties_input: Specific details or features to be considered during data generation.
        synthesis_generator_alpaca_prompt: A detailed prompt guiding the generator's response.
        url: The endpoint of the synthesis generator service. Defaults to "http://10.32.2.5:82/call".
        max_attemps: The maximum number of retry attempts in case of service unavailability. Defaults to 3.
        **kwargs: Additional parameters to be sent to the synthesis generator service.
    
    Returns:
        The generated data as a dictionary if the request is successful, a descriptive error message 
        if JSON packing fails or the HTTP status code is not 200, and None if all retry attempts fail.
    """

    params = {
        "synthesis_generator_system_prompt": synthesis_generator_system_prompt,
        "properties_input": properties_input,
        "synthesis_generator_alpaca_prompt": synthesis_generator_alpaca_prompt,
        **kwargs,
    }
    for attempt in range(max_attemps):
        try:
            resp = requests.post(url, data=json.dumps(params))

            if resp.status_code == 200:
                try:
                    data = json.loads(resp.json())
                    return data
                except Exception as e:
                    return f'Exception occured during json packing: {e}'
            else: return f'Response status code is {resp.status_code}'

        except requests.ConnectionError as e:
            logger.exception(f"Attempt 'call_for_generation' {attempt + 1}/{max_attemps}: Connection failed with error: {e}")
            print(f"Attempt {attempt + 1}/{max_attemps}: Connection failed with error: {e}")
            time.sleep(1.05 ** attempt)

        except requests.RequestException as e:
            logger.exception(f"Attempt 'call_for_generation' {attempt + 1}: Failed with error: {e}")
            print(f"Attempt {attempt + 1}: Failed with error: {e}")
            break  # Other request-related errors are not retried

    return None

@tool
def synthesis_generation(description: str, config: RunnableConfig) -> str:
    """
    Generates a detailed synthesis procedure based on a given description of the nanoparticles.
    
    Args:
        description (str): A textual description of the desired nanoparticles, specifying their properties and intended use.
        config (RunnableConfig): Configuration object containing the model to use for generation.
    
    Returns:
        synthesis_text (str): A string containing the generated synthesis procedure. Returns an error message if synthesis generation fails.
    """
    try:
        llm: BaseChatModel = config["configurable"]["model"]
        predictor = synt_prompt | llm
        resp = predictor.invoke(description).content
        return resp
    except Exception as e:
        logger.exception(f"'synthesis_generation' failed with error: {e}")
        return f"I couldn't generate synthesis right now"

@tool
def predict_nanoparticle_entrapment_eff(description: str, config: RunnableConfig) -> str:
    """
    Predicts the entrapment efficiency of a nanomaterial based on its descriptive text.
    
    This method uses a language model to infer the entrapment efficiency 
    from the provided nanomaterial description, enabling quick assessment 
    without direct experimentation.
    
    Args:
        description (str): A textual description of the nanomaterial, 
            e.g., "nanoparticles obtained from the dissolution of calcium 
            carbonate in HCl".
        config (RunnableConfig): Configuration object containing the language 
            model to use for prediction.
    
    Returns:
        ent_eff (str): The predicted entrapment efficiency of the nanomaterial. 
            Returns an error message if prediction fails.
    """
    try:
        llm: BaseChatModel = config["configurable"]["model"]
        predictor = entr_eff_prompt | llm
        res = predictor.invoke(description)
        entr_eff = res.content
        return entr_eff
    except Exception as e:
        logger.exception(f"'predict_nanoparticle_entrapment_eff' failed with error: {e}")
        return f"I couldn't predict entrapment efficiency right now"


@tool
def predict_nanoparticle_shape(description: str, config: RunnableConfig) -> str:
    """
    Predicts the shape of a nanomaterial from a textual description by leveraging a language model.
    
    Args:
        description (str): A textual description of the nanomaterial, e.g., "nanoparticles obtained from the dissolution of calcium carbonate in HCl".
        config (RunnableConfig): Configuration object containing the language model to use.
    
    Returns:
        str: The predicted shape(s) of the nanomaterial as determined by the language model. Returns "I couldn't predict shapes" if prediction fails.
    """
    try:
        llm: BaseChatModel = config["configurable"]["model"]
        prompt = (
            properties_prediction_prompt
            + description
        )
        res = llm.invoke(prompt)
        predicted_shapes = res.content
        return predicted_shapes
    except Exception as e:
        logger.exception(f"'predict_nanoparticle_shape' failed with error: {e}")
        return f"I couldn't predict shapes"


@tool
def generate_nanoparticle_images(shape: str) -> str:
    """
    Generates an image representation of a nanoparticle with a given shape, leveraging an underlying inference process. This allows visualizing nanomaterials for analysis and research.
    
    Args:
        shape (str): The desired shape of the nanoparticle ('cube', 'sphere', 'stick', 'flat', 'amorphous').
    
    Returns:
        str: A message indicating success and the nanoparticle shape, or an error message if image generation fails.
    """
    try:
        inference(shape)
        shape = f"I've successfully generated images of {shape} nanoparticles"
        return shape
    except Exception as e:
        logger.exception(f"'generate_nanoparticle_images' failed with error: {e}")
        return f"I've couldn't generate images because of: {str(e)}, I should move to the next task if any"
    

@tool
def analyse_nanoparticle_images(config: RunnableConfig) -> str:
    """
    Analyzes images to determine the shapes of nanoparticles present. This method processes a list of image paths and utilizes a vision language model to identify the shapes.
    
    Args:
        config (RunnableConfig): A configuration object containing the visual model and image paths. 
            It is expected to have a "configurable" key which itself contains keys "visual_model" (a BaseChatModel instance) and "img_path" (a list of base64 encoded image strings).
    
    Returns:
        str: A string containing the predicted shape for each input image, or an error message if no images are provided.
    """
    llm: BaseChatModel = config["configurable"].get("visual_model")

    if llm is None:
        raise ValueError('Visual model is not set')
    

    base64_images = config["configurable"].get("img_path")

    if base64_images is None:
        return "There is no image to process" #TODO: implement human-in-the loop here

    results = ''
    for idx, base64_image in enumerate(base64_images):
        output_message = llm.invoke(
            [
                (
                    "human",
                    [
                        {"type": "text", "text": shape_detection_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"},
                        },
                    ],
                ),
            ]
        )
        results += f'image {idx+1} has {output_message.content} shape\n'
    cleaned_results = re.sub(r"\.", "", results)   
    return cleaned_results

nanoparticle_tools = [synthesis_generation, predict_nanoparticle_shape, generate_nanoparticle_images, analyse_nanoparticle_images, predict_nanoparticle_entrapment_eff]
#nanoparticle_tools = [predict_nanoparticle_shape, generate_nanoparticle_images, analyse_nanoparticle_images]

nano_tools_rendered = render_text_description(nanoparticle_tools)