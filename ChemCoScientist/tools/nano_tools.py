import json
import re
import time

import requests
from langchain.tools.render import render_text_description
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import tool

from ChemCoScientist.agents.tools_prompts import (
    entr_eff_prompt,
    properties_prediction_prompt,
    shape_detection_prompt,
    synt_prompt,
)
from ChemCoScientist.tools.models.generative_inference import inference


def call_for_generation(
    synthesis_generator_system_prompt,
    properties_input,
    synthesis_generator_alpaca_prompt,
    url: str = "http://10.32.2.5:82/call",
    max_attemps=3,
    **kwargs,
):
    """
    Sends a request to a synthesis service with scientific context to generate data.
    
    This method constructs a request containing a system prompt, input properties, 
    and an Alpaca prompt and sends it to a specified service endpoint. It handles
    potential connection errors by retrying the request a certain number of times. 
    The service is expected to process the input and return generated data.
    
    Args:
        synthesis_generator_system_prompt (str): The system-level instructions for the generation service.
        properties_input (dict): The input data or properties to be used in the generation process.
        synthesis_generator_alpaca_prompt (str): A prompt designed to guide the generation service.
        url (str, optional): The URL of the synthesis service. Defaults to "http://10.32.2.5:82/call".
        max_attemps (int, optional): The maximum number of times to retry the request. Defaults to 3.
        **kwargs: Additional parameters to be sent to the synthesis service.
    
    Returns:
        dict or str or None: A dictionary containing the generated data if the request 
            is successful and the response is valid JSON. Returns an error message string 
            if the response status code is not 200 or if JSON parsing fails. 
            Returns None if all retry attempts fail.
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
                    return f"Exception occured during json packing: {e}"
            else:
                return f"Response status code is {resp.status_code}"

        except requests.ConnectionError as e:
            # logger.exception(f"Attempt 'call_for_generation' {attempt + 1}/{max_attemps}: Connection failed with error: {e}")
            print(
                f"Attempt {attempt + 1}/{max_attemps}: Connection failed with error: {e}"
            )
            time.sleep(1.05**attempt)

        except requests.RequestException as e:
            # logger.exception(f"Attempt 'call_for_generation' {attempt + 1}: Failed with error: {e}")
            print(f"Attempt {attempt + 1}: Failed with error: {e}")
            break  # Other request-related errors are not retried

    return None


@tool
def synthesis_generation(description: str, config: RunnableConfig) -> str:
    """
    Generates a detailed synthesis procedure based on a given description of the nanoparticles. This allows users to quickly obtain potential methods for creating specific nanomaterials.
    
    Args:
        description (str): A textual description of the desired nanoparticles, including composition, size, and any specific properties.
        config (RunnableConfig): Configuration object containing the language model to be used.
    
    Returns:
        synthesis_text (str): A text string detailing a possible synthesis procedure for the described nanoparticles.  Returns an error message if synthesis generation fails.
    """
    try:
        llm: BaseChatModel = config["configurable"]["model"]
        predictor = synt_prompt | llm
        resp = predictor.invoke(description).content
        return resp
    except Exception as e:
        # logger.exception(f"'synthesis_generation' failed with error: {e}")
        return f"I couldn't generate synthesis right now"


@tool
def predict_nanoparticle_entrapment_eff(
    description: str, config: RunnableConfig
) -> str:
    """
    Predicts the entrapment efficiency of a nanomaterial based on its descriptive text.
    
    This method utilizes a language model to interpret the provided nanomaterial description and estimate its entrapment efficiency.  It translates textual information about the material's properties and creation process into a quantifiable prediction.
    
    Args:
        description (str): A textual description of the nanomaterial, e.g., "nanoparticles obtained from the dissolution of calcium carbonate in HCl".
        config (RunnableConfig): Configuration object containing the language model to be used for prediction.
    
    Returns:
        ent_eff (str): The predicted entrapment efficiency of the nanomaterial, as a string.  May return an error message if prediction fails.
    """
    try:
        llm: BaseChatModel = config["configurable"]["model"]
        predictor = entr_eff_prompt | llm
        res = predictor.invoke(description)
        entr_eff = res.content
        return entr_eff
    except Exception as e:
        # logger.exception(f"'predict_nanoparticle_entrapment_eff' failed with error: {e}")
        return f"I couldn't predict entrapment efficiency right now"


@tool
def predict_nanoparticle_shape(description: str, config: RunnableConfig) -> str:
    """
    Predicts the shape of a nanomaterial from its descriptive text.
    
    Args:
        description (str): A text describing the nanomaterial, e.g., "nanoparticles obtained from the dissolution of calcium carbonate in HCl".
        config (RunnableConfig): Configuration object containing the language model to use.
    
    Returns:
        str: The predicted shape(s) of the nanomaterial, as determined by the language model. Returns "I couldn't predict shapes" if prediction fails.
    """
    try:
        llm: BaseChatModel = config["configurable"]["llm"]
        prompt = properties_prediction_prompt + description
        res = llm.invoke(prompt)
        predicted_shapes = res.content
        return predicted_shapes
    except Exception as e:
        # logger.exception(f"'predict_nanoparticle_shape' failed with error: {e}")
        return f"I couldn't predict shapes"


@tool
def generate_nanoparticle_images(shape: str) -> str:
    """
    Generates an image representation of a nanoparticle based on its specified shape. This helps visualize nanomaterials for research and analysis.
    
    Args:
        shape (str): The desired shape of the nanoparticle. Valid options are: 'cube', 'sphere', 'stick', 'flat', 'amorphous'.
    
    Returns:
        str: A message indicating success or failure of the image generation process, including the nanoparticle shape or the error encountered.
    """
    try:
        inference(shape)
        shape = f"I've successfully generated images of {shape} nanoparticles"
        return shape
    except Exception as e:
        # logger.exception(f"'generate_nanoparticle_images' failed with error: {e}")
        return f"I've couldn't generate images because of: {str(e)}, I should move to the next task if any"


@tool
def analyse_nanoparticle_images(config: RunnableConfig) -> str:
    """
    Analyzes images to determine the shape of nanoparticles present. This method processes image data to provide insights into nanomaterial characteristics.
    
    Args:
        config (RunnableConfig): Configuration object containing the visual model and image paths.
            - `configurable` (dict): A dictionary within the config containing:
                - `visual_model` (BaseChatModel): The language model used for image analysis.
                - `img_path` (list): A list of base64 encoded image strings.
    
    Returns:
        str: A string describing the predicted shape of the nanoparticles in each image, or an error message if no image is provided. 
             The output includes the image number and corresponding shape prediction for each image processed.
    """
    llm: BaseChatModel = config["configurable"].get("visual_model")

    if llm is None:
        raise ValueError("Visual model is not set")

    base64_images = config["configurable"].get("img_path")

    if base64_images is None:
        return "There is no image to process"  # TODO: implement human-in-the loop here

    results = ""
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
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                ),
            ]
        )
        results += f"image {idx+1} has {output_message.content} shape\n"
    cleaned_results = re.sub(r"\.", "", results)
    return cleaned_results


nanoparticle_tools = [
    synthesis_generation,
    predict_nanoparticle_shape,
    generate_nanoparticle_images,
    analyse_nanoparticle_images,
    predict_nanoparticle_entrapment_eff,
]

nano_tools_rendered = render_text_description(nanoparticle_tools)
