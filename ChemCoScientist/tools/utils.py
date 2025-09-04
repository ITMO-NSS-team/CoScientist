import base64
import os
import pandas as pd

from PIL import Image


def convert_to_base64(image_file_path):
    """
    Convert an image file to a Base64 encoded string.
    
        :param image_file_path: Path to the image file.
        :return: Base64 encoded string of the image.
    
        This method facilitates image processing by converting them into a string representation 
        suitable for storage or transmission, ensuring compatibility across different systems 
        and avoiding potential data loss during transfer. It achieves this by temporarily 
        saving the image to a file, encoding it, and then removing the temporary file.
    """
    pil_image = Image.open(image_file_path)
    pil_image.save("tmp.png", format="png")

    with open("tmp.png", "rb") as image_file:
        result = base64.b64encode(image_file.read()).decode("utf-8")
        os.remove("tmp.png")
        return result


def convert_to_html(img_base64):
    """
    Converts a base64 encoded string into an HTML image tag for display.
    
    This allows embedding images directly within a web page or rich text environment 
    without requiring separate image files. The method constructs an `<img>` tag 
    with the base64 string embedded as the image source.
    
    Args:
        img_base64 (str): The base64 encoded string representing the image data.
    
    Returns:
        str: An HTML string representing the image tag with the base64 data as the source.
    """
    # Create an HTML img tag with the base64 string as the source
    image_html = (
        f'<img src="data:image/jpeg;base64,{img_base64}" style="max-width: 100%;"/>'
    )
    return image_html


def filter_valid_strings(
    df: pd.DataFrame, column_name: str, max_length: int = 200
) -> pd.DataFrame:
    """
    Filters a DataFrame to include only rows where the specified column contains strings of valid length.
    
    This function ensures data quality by removing entries that do not conform to length requirements,
    preventing potential issues during downstream analysis or processing. It checks if the column exists,
    verifies that the values are strings, and then filters based on a maximum length.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to filter.
        max_length (int, optional): The maximum allowed length for strings in the column. Defaults to 200.
    
    Returns:
        pd.DataFrame: A new DataFrame containing only the rows where the specified column 
                      contains strings with a length less than or equal to `max_length`.
                      Returns a copy of the filtered dataframe.
    
    Example:
    -------
    >>> df = pd.DataFrame({'text': ['abc', 'def'*100, 123]})
    >>> filtered_df = filter_valid_strings(df, 'text')
    >>> print(filtered_df)
    """
    try:
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found")

        is_string = df[column_name].apply(lambda x: isinstance(x, str))
        valid_length = df[column_name].str.len() <= max_length

        filtered_df = df[is_string & valid_length].copy()

        return filtered_df

    except Exception as e:
        raise ValueError(e)
