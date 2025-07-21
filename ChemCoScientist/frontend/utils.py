import os
import uuid
from pathlib import Path
from typing import Callable, Iterable, Union

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_pills import pills

# BASE_DATA_DIR = Path("./user_data")
BASE_DATA_DIR = "datasets"

import os
import shutil

def clean_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        return

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)  
                print(f"Deleted file: {file_path}")
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  
                print(f"Deleted folder: {file_path}")
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")



def get_user_data_dir(id):
    """
    Get or create a unique directory for the current user session.

    Returns:
        str: The path to the user's data directory.
    """

    path = os.path.join(BASE_DATA_DIR, id)
    os.makedirs(path, exist_ok=True)
    return path


def get_user_session_id():
    """
    Get or generate a unique user session ID.

    Returns:
        str: A unique identifier for the current user session.
    """

    id = str(uuid.uuid4())
    return id


def clear_directory(directory: Path):
    """
    Clear the contents of a directory.

    Args:
        directory (Path): The directory to clear.
    """
    if os.path.exists(directory):
        for item in os.listdir(directory):
            try:
                file_path = os.path.join(directory, item)
                if os.path.isfile(file_path):
                    # item.unlink()
                    os.remove(file_path)
                else:
                    clear_directory(item)
                    os.remove(os.path.join(directory, item))
                    # item.rmdir()
            except Exception as e:
                print(f"Failed to delete {item}. Reason: {e}")


def save_uploaded_file(file: UploadedFile, directory: Path):
    """
    Save an uploaded file to the specified directory.

    Args:
        file (UploadedFile): The file uploaded by the user.
        directory (str): The directory to save the file in.
    """
    file_path = os.path.join(directory, file.name)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())


def save_all_files(user_data_dir: Path):
    """
    When the task starts to run, save all the user's uploaded files to user's directory

    Args:
        user_data_dir (str): The directory path where user's files will be saved.
    """
    clear_directory(user_data_dir)
    for _, file_data in st.session_state.uploaded_files.items():
        save_uploaded_file(file_data["file"], user_data_dir)


def file_uploader(uploaded_files):
    """
    Process uploaded files and store them in session state.

    This function takes uploaded files, reads CSV and Excel files into pandas DataFrames,
    and stores both the original file and DataFrame in the session state dictionary.

    Args:
        uploaded_files: List of uploaded file objects from Streamlit's file_uploader widget
    """
    st.session_state.uploaded_files = {}
    print(uploaded_files)

    for file in uploaded_files:
        print(file)
        suffix = file.name.lower().split(".")[-1]
        df = None

        if suffix == "csv":
            df = pd.read_csv(file)
            df.to_csv(
                os.environ["DS_STORAGE_PATH"] + "/" + "users_dataset_" + file.name,
                index=False,
            )

        elif suffix in ["xls", "xlsx"]:
            df = pd.read_excel(file)
            df.to_excel(
                os.environ["DS_STORAGE_PATH"] + "/" + "users_dataset_" + file.name,
                index=False,
            )

        else:
            st.warning(f"Unsupported file type: {suffix}")
            continue

        st.session_state.uploaded_files[file.name] = {"file": file, "df": df}
    return st.session_state.uploaded_files


def papers_uploader(uploaded_files):
    """
    Process uploaded papers and save them.
    """
    st.session_state.uploaded_files = {}
    print(uploaded_files)

    for file in uploaded_files:
        print(file)
        suffix = file.name.lower().split(".")[-1]

        if suffix == "pdf":
            with open(os.environ["PAPERS_STORAGE_PATH"] + "/" + file.name, "wb") as f:
                f.write(file.getbuffer())
        else:
            st.warning(f"Unsupported papers type: {suffix}")
            continue

        st.session_state.uploaded_files[file.name] = {"file": file}
    return st.session_state.uploaded_files


def custom_pills(
    label: str,
    options: Iterable[str],
    icons: Iterable[str] = None,
    index: Union[int, None] = 0,
    format_func: Callable = None,
    label_visibility: str = "visible",
    clearable: bool = None,
    key: str = None,
    reset_key: str = None,
):
    """
    Displays clickable pills with an option to reset the selection.

    Args:
        label (str): The label shown above the pills.
        options (iterable of str): The texts shown inside the pills.
        icons (iterable of str, optional): The emoji icons shown on the left side of the pills. Each item must be a single emoji. Default is None.
        index (int or None, optional): The index of the pill that is selected by default. If None, no pill is selected. Defaults to 0.
        format_func (callable, optional): A function applied to the pill text before rendering. Defaults to None.
        label_visibility ("visible" or "hidden" or "collapsed", optional): The visibility of the label. Use this instead of `label=""` for accessibility. Defaults to "visible".
        clearable (bool, optional): Whether the user can unselect the selected pill by clicking on it. Default is None.
        key (str, optional): The key of the component. Defaults to None.
        reset_key (str, optional): The key used to reset the selection. Defaults to None.

    Returns:
        (any): The text of the pill selected by the user (same value as in `options`).
    """

    # Create a unique key for the component to force update when necessary
    unique_key = f"{key}-{reset_key}" if key and reset_key else key

    # Pass the arguments to the pills function
    selected = pills(
        label=label,
        options=options,
        icons=icons,
        index=index,
        format_func=format_func,
        label_visibility=label_visibility,
        clearable=clearable,
        key=unique_key,
    )

    return selected
