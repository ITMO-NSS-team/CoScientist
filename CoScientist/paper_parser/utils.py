import base64
from io import BytesIO
from urllib.parse import urlparse

from langchain_core.messages import HumanMessage
from PIL import Image

from CoScientist.paper_parser.s3_connection import s3_service


def convert_to_base64(file_path):
    """
    Convert an image file to a Base64 encoded string.
    
    This method reads an image from the specified file path, encodes it as a JPEG image in memory, then converts it into a Base64 string representation. 
    
    Args:
        file_path (str): The path to the image file.
    
    Returns:
        str: A Base64 encoded string representing the JPEG image.
    """
    if file_path.startswith("http://"):
        s3_key, bucket_name = extract_s3_bucket_and_key(file_path)
        pil_image = Image.open(BytesIO(s3_service.get_image_bytes_from_s3(s3_key, bucket_name)))
    else:
        pil_image = Image.open(file_path)
    buffered = BytesIO()
    pil_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str


def prompt_func(data):
    """
    Creates a structured message containing text and images for use in a conversational context.
    
    Args:
        data (dict): A dictionary containing the message content.
            - "text" (str): The text content of the message.
            - "image" (list): A list of base64 encoded JPEG images to include in the message.
    
    Returns:
        HumanMessage: A HumanMessage object with a structured 'content' list.
            The 'content' list contains dictionaries representing each part of the message,
            with "type" keys indicating whether it's "text" or "image_url".  Image URLs
            are formatted as data URIs.
    
    This method prepares the input data into a format suitable for presenting information in a multi-modal interface,
    by converting images to data URIs that can be directly embedded in a message and combining them with the provided text.
    """
    text = data["text"]
    imgs = data["image"]
    content_parts = []

    for img in imgs:
        image_part = {
            "type": "image_url",
            "image_url": f"data:image/jpeg;base64,{img}",
        }
        content_parts.append(image_part)

    text_part = {"type": "text", "text": text}
    content_parts.append(text_part)

    return HumanMessage(content=content_parts)


def extract_s3_bucket_and_key(s3_url: str):
    o = urlparse(s3_url)
    bucket, key = o.path.split('/', 2)[1:]
    return key, bucket
