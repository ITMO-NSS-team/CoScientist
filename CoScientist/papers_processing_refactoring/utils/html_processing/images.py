import base64
from io import BytesIO
import re

from PIL import Image

from ..general_utils import prompt_func
from ..prompts import cls_prompt, table_extraction_prompt, image_captioning_prompt
from ...domain.entities import ImageInfo


def pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def check_image_relevance(image_b64: str, llm) -> bool:
    query = [prompt_func({"text": cls_prompt, "image": [image_b64]})]
    decision = llm.invoke(query).content.strip()
    
    return decision != "False"


def try_extract_table(image_b64: str, llm) -> str | None:
    table_query = [prompt_func({"text": table_extraction_prompt, "image": [image_b64]})]
    res = llm.invoke(table_query).content.strip()
    
    if res != "No table":
        pattern = r'<table\b[^>]*>.*?</table>'
        match = re.search(pattern, res, re.DOTALL)
        if match:
            return match.group(0)
    return None


def caption_image(
        image_info: ImageInfo,
        pil_image: Image.Image,
        llm
) -> ImageInfo:
    if not image_info.is_kept:
        return image_info
    
    image_b64 = pil_to_base64(pil_image)
    query = [prompt_func({"text": image_captioning_prompt, "image": [image_b64]})]
    caption = llm.invoke(query).content.strip()
    image_info.caption = caption
    
    return image_info
