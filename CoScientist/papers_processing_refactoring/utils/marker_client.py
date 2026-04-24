import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image


def load_bytes(uri: str) -> bytes:
    """If uri is local path, read the file, if uri is http url, download it and return bytes
    
    Args
        uri: local path or http url
    Returns
        content of the file
    """
    if uri.startswith("http://") or uri.startswith("https://"):
        response = requests.get(uri)
        content = response.content
    else:
        content = Path(uri).read_bytes()
    return content


@dataclass
class ConvertedContent:
    images: dict[str, Image.Image]
    text: str


class MarkerClient:
    def __init__(self, base_url: str):
        self.base_url = base_url

    def convert(self, pdf_uri: str) -> ConvertedContent:
        content = load_bytes(pdf_uri)
        files = {"pdf_file": ("file.pdf", content, "application/pdf")}
        response = requests.post(self.base_url, files=files)

        res = response.json()
        text, images, elapsed_time = res["html"], res["images"], res["processing_time"]

        decoded_images = {}
        for name, image in images.items():
            bio = BytesIO()
            bio.write(base64.b64decode(image))
            pil_image = Image.open(bio)
            decoded_images[name] = pil_image
        return ConvertedContent(images=decoded_images, text=text)
    