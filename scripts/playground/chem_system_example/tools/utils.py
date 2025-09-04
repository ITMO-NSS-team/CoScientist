import base64
from PIL import Image
import os


def convert_to_base64(image_file_path):
    """
    Converts an image file to a Base64 encoded string.
    
        This process facilitates the storage and transmission of image data as text,
        allowing it to be embedded within documents or exchanged more easily across
        systems that may not directly support binary image formats. The image is temporarily
        saved to a file, encoded, and then the temporary file is removed.
    
        :param image_file_path: Path to the image file.
        :type image_file_path: str
        :return: Base64 encoded string representation of the image.
        :rtype: str
    """
    pil_image = Image.open(image_file_path)
    pil_image.save('tmp.png', format="png")

    with open('tmp.png', "rb") as image_file:
        result = base64.b64encode(image_file.read()).decode("utf-8")
        os.remove('tmp.png')
        return result
    
def convert_to_html(img_base64):
    """
    Converts a base64 encoded image string into an HTML img tag for display.
    
        This allows embedding images directly within web-based outputs without needing separate image files.
        The method constructs an HTML image tag with the base64 string as the image source, ensuring the image is 
        displayed responsively by limiting its maximum width to 100%.
    
        :param img_base64: The base64 encoded string representing the image data.
        :type img_base64: str
        :return: An HTML string representing an img tag with the base64 data as the source.
        :rtype: str
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = f'<img src="data:image/jpeg;base64,{img_base64}" style="max-width: 100%;"/>'
    return image_html