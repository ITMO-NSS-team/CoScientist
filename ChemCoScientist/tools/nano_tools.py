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
    """Generates the text of the synthesis of nanoparticles. Use it ONLY when you are asked to generate synthesis next

    Args:
        description (int): Description of nanoparticles: any string description is suitable

    Returns:
        synthesis_text (str): Text of the synthesis of nanoparticles
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
    """Predicts the entrapment efficiency of nanomaterial based on it's description.

    Args:
        description (str): description of nanomaterial, for example: nanoparticles obtained from the dissolution of calcium carbonate in HCl

    Returns:
        ent_eff (str): predicted entrapment efficiency of nanomaterial
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
    """Predicts the shape of nanomaterial based on it's description.

    Args:
        description (str): description of nanomaterial, for example: nanoparticles obtained from the dissolution of calcium carbonate in HCl

    Returns:
        predicted_shapes (str): predicted shapes of nanomaterial
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
    """Generates the image of nanoparticle based on it's shape. Use it when you are asked to generate nanoparticles image

    Args:
        shape (str): shape of nanomaterial: 'cube', 'sphere', 'stick', 'flat', 'amorphous'

    Returns:
        predicted_shapes (str): predicted shapes of nanomaterial
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
    """Predicts shape of nanoparticles based on their image. Use it when you need to process submitted image.

    Returns:
        predicted_shapes (str): predicted shapes of nanomaterial
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
