from streamlit_pills import pills
from typing import Iterable, Union, Callable
from pathlib import Path
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
import pandas as pd
import uuid
import os

#BASE_DATA_DIR = Path("./user_data")
BASE_DATA_DIR = 'datasets'


def get_user_data_dir(id):
    """
    Get the directory for a given user or create it if it doesn't exist.
    
    Args:
        id (str): A unique identifier for the user.
    
    Returns:
        str: The path to the user's data directory.
    """

    path =  os.path.join(BASE_DATA_DIR, id)
    os.makedirs(path, exist_ok=True)
    return path


def get_user_session_id():
    """
    Get a unique identifier to associate with a user's interactions.
    
        This ensures each interaction can be tracked and distinguished,
        facilitating a personalized and consistent experience.
    
        Returns:
            str: A universally unique identifier (UUID) representing the user session.
    """

    id = str(uuid.uuid4())
    return id


def clear_directory(directory: Path):
    """
    Recursively removes all files and subdirectories within a given directory.
    This ensures a clean state for operations requiring a pristine environment.
    
    Args:
        directory (Path): The path to the directory to be cleared.
    
    Returns:
        None
    """
    if os.path.exists(directory):
        for item in os.listdir(directory):
            try:
                file_path = os.path.join(directory, item)
                if os.path.isfile(file_path):
                    #item.unlink()
                    os.remove(file_path)
                else:
                    clear_directory(item)
                    os.remove(os.path.join(directory, item))
                    #item.rmdir()
            except Exception as e:
                print(f"Failed to delete {item}. Reason: {e}")


def save_uploaded_file(file: UploadedFile, directory: Path):
    """
    Save an uploaded file to a specified location on the filesystem.
    
    This ensures that user-provided files are stored securely and are accessible for subsequent processing or analysis. 
    
    Args:
        file (UploadedFile): The file object representing the uploaded file.
        directory (Path): The destination directory where the file will be saved.
    
    Returns:
        None
    """
    file_path = os.path.join(directory, file.name)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())


def save_all_files(user_data_dir: Path):
    """
    Saves user-uploaded files to a designated directory, preparing them for analysis. 
    This ensures that any documents provided by the user are readily accessible for processing and integration with the existing knowledge base.
    
    Args:
        user_data_dir (str): The directory path where user's files will be saved.
    
    Returns:
        None
    """
    clear_directory(user_data_dir)
    for _, file_data in st.session_state.uploaded_files.items():
        save_uploaded_file(file_data["file"], user_data_dir)


def file_uploader(uploaded_files):
    """
    Process uploaded files and store them in session state for subsequent analysis.
    
    This function handles file uploads, specifically CSV and Excel formats, converting them into pandas DataFrames.
    The original file objects and their corresponding DataFrames are stored in the session state,
    making them readily available for further processing and analysis within the application.
    
    Args:
        uploaded_files: A list of uploaded file objects, typically originating from a Streamlit file uploader.
    
    Returns:
        dict: A dictionary where keys are filenames and values are dictionaries containing the original file object 
              and its corresponding pandas DataFrame (if applicable).  If a file isn't CSV or Excel, the DataFrame 
              value will be None.
    """
    st.session_state.uploaded_files = {}
    for file in uploaded_files:
        df = None
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
        st.session_state.uploaded_files[file.name] = {"file": file, "df": df}
    return  st.session_state.uploaded_files

def custom_pills(label: str, options: Iterable[str], icons: Iterable[str] = None, index: Union[int, None] = 0,
                 format_func: Callable = None, label_visibility: str = "visible", clearable: bool = None,
                 key: str = None, reset_key: str = None):
    """
    Displays clickable pills for selecting an option from a predefined list.
    
    Args:
        label (str): The label displayed above the pills.
        options (iterable of str): A list of strings representing the possible options to choose from.
        icons (iterable of str, optional): A list of emoji icons to display alongside each pill. Defaults to None.
        index (int or None, optional): The index of the pill to be initially selected. Defaults to 0.
        format_func (callable, optional): A function to format the text displayed on each pill. Defaults to None.
        label_visibility ("visible" or "hidden" or "collapsed", optional): Controls the visibility of the label for accessibility. Defaults to "visible".
        clearable (bool, optional):  Enables the user to deselect a currently selected pill. Defaults to None.
        key (str, optional): A unique key for the component, used for state management. Defaults to None.
        reset_key (str, optional): A key used to trigger a reset of the selection. Defaults to None.
    
    Returns:
        str: The text of the selected pill from the `options` list.
    """

    # Create a unique key for the component to force update when necessary
    unique_key = f"{key}-{reset_key}" if key and reset_key else key

    # Pass the arguments to the pills function
    selected = pills(label=label, options=options, icons=icons, index=index, format_func=format_func,
                     label_visibility=label_visibility, clearable=clearable, key=unique_key)

    return selected

