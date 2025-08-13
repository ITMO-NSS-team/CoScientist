import os
import uuid
import threading
import time

from dotenv import load_dotenv
from pathlib import Path
from typing import Callable, Iterable, Union

import pandas as pd
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit_pills import pills

from definitions import ROOT_DIR, CONFIG_PATH

load_dotenv(CONFIG_PATH)

BASE_DATA_DIR = "datasets"
PATH_TO_TEMP_FILES = os.environ["PATH_TO_TEMP_FILES"]
INACTIVITY_WINDOW_SECONDS = 24 * 60 * 60  # 24 hours

import os
import shutil

def clean_folder(folder_path):
    """
    Deletes all files and subdirectories within a specified folder to ensure a clean environment for processing new data or experiments.
    
    Args:
        folder_path (str): The path to the folder to be cleaned.
    
    Returns:
        None. Prints messages indicating which files/folders were deleted
        or if deletion failed.
    """
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
    Get the directory associated with a user ID. If the directory doesn't exist, it will be created.
    
    Args:
        id (str): The unique identifier for the user.
    
    Returns:
        str: The path to the user's data directory.
    """

    path = os.path.join(BASE_DATA_DIR, id)
    os.makedirs(path, exist_ok=True)
    return path


def get_user_session_id():
    """
    Get a unique identifier for the current user's interaction.
    
        This ID helps to track and manage the context of a user's requests 
        across multiple interactions, allowing the system to maintain continuity 
        and provide more relevant responses.
    
        Returns:
            str: A universally unique identifier (UUID) as a string, representing
                 the user's session.
    """

    id = str(uuid.uuid4())
    return id


def clear_directory(directory: Path):
    """
    Recursively removes all files and subdirectories within a given directory. 
    
    This ensures that the directory is empty before new content is added or processed, 
    which is crucial for maintaining data integrity during updates and analyses.
    It handles potential errors during deletion and reports them to the console.
    
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
    Save an uploaded file to a specified directory. This ensures that user-provided documents are stored for potential analysis and integration with existing scientific literature.
    
    Args:
        file (UploadedFile): The file uploaded by the user.
        directory (Path): The directory to save the file in.
    
    Returns:
        None
    """
    file_path = os.path.join(directory, file.name)
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())


def save_all_files(user_data_dir: Path):
    """
    Clears the user's directory and saves all uploaded files to it, ensuring only the current task's files are present.
    
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
    Process uploaded files, convert supported formats to DataFrames, and store them for further analysis.
    
    This function handles file uploads, specifically CSV and Excel files, converting them into pandas DataFrames.
    The DataFrames and original files are stored in session state for subsequent use, enabling users to 
    work with their data within the application. The converted data are also saved to the backend storage.
    
    Args:
        uploaded_files: A list of uploaded file objects from Streamlit's file_uploader widget.
    
    Returns:
        dict: A dictionary where keys are file names and values are dictionaries containing the original file object and its corresponding DataFrame. 
              Returns an empty dictionary if no files were successfully processed.
    """
    st.session_state.uploaded_files = {}

    for file in uploaded_files:
        suffix = file.name.lower().split(".")[-1]
        df = None
        clean_folder(os.environ['DS_STORAGE_PATH'])
        if suffix == "csv":
            df = pd.read_csv(file)

            df.to_csv(
                os.environ["DS_STORAGE_PATH"] + "/" + "users_dataset.csv",
                index=False,
            )

        elif suffix in ["xls", "xlsx"]:
            df = pd.read_excel(file)
            print(df)
            df.to_csv(
                os.environ["DS_STORAGE_PATH"] + "/" + "users_dataset.csv",
                index=False,
            )

        else:
            st.warning(f"Unsupported file type: {suffix}")
            continue

        st.session_state.uploaded_files[file.name] = {"file": file, "df": df}
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
    Displays clickable pills for selecting an option from a predefined list.
    
    Args:
        label (str): The label displayed above the pills.
        options (iterable of str): The text options for each pill.
        icons (iterable of str, optional): Optional emoji icons to display alongside each pill. Defaults to None.
        index (int or None, optional): The index of the pill to be initially selected. Defaults to 0.
        format_func (callable, optional): A function to format the pill text before display. Defaults to None.
        label_visibility (str, optional): Controls the visibility of the label ("visible", "hidden", "collapsed"). Defaults to "visible".
        clearable (bool, optional): Enables or disables the ability to deselect the chosen pill. Defaults to None.
        key (str, optional): A unique key for the component. Defaults to None.
        reset_key (str, optional): A key used to trigger a reset of the pill selection. Defaults to None.
    
    Returns:
        str: The text of the selected pill from the `options` list.
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


def update_activity(session_folder: str) -> None:
    """
    Updates the last activity timestamp for a session.
    
    This method records the time of the latest interaction with a session,
    allowing the system to track session usage and potentially manage resources
    or provide context-aware features. It writes the current timestamp to a 
    hidden file within the session directory.
    
    Args:
        session_folder: The path to the session folder.
    
    Returns:
        None
    """
    with open(os.path.join(session_folder, ".last_activity"), "w") as f:
        f.write(str(time.time()))


def cleanup_expired_sessions(base_dir: str = None) -> None:
    """
    Deletes expired session folders to manage storage and maintain a clean environment.
    
    Args:
        base_dir (str, optional): The base directory where session folders are stored. 
                                  Defaults to the project's temporary files directory.
    
    Returns:
        None
    """
    if base_dir is None:
        base_dir = os.path.join(ROOT_DIR, PATH_TO_TEMP_FILES)
    if not os.path.exists(base_dir):
        return
    now = time.time()
    for session_id in os.listdir(base_dir):
        session_folder = os.path.join(base_dir, session_id)
        last_activity_file = os.path.join(session_folder, ".last_activity")
        if os.path.isdir(session_folder) and os.path.exists(last_activity_file):
            with open(last_activity_file, "r") as f:
                last_activity = float(f.read().strip())
            if now - last_activity > INACTIVITY_WINDOW_SECONDS:
                shutil.rmtree(session_folder)


def start_cleanup_thread(interval: int = 60 * 60) -> threading.Thread:
    """
    Starts a background thread to periodically remove outdated or unused session data.
    
    Args:
        interval (int, optional): The time in seconds between cleanup cycles. Defaults to 3600 (1 hour).
    
    Returns:
        threading.Thread: The thread object that was started.
    
    This method ensures that the system remains efficient by regularly removing inactive sessions, 
    preventing the accumulation of unnecessary data and optimizing resource usage.
    """
    def cleanup_loop():
        while True:
            cleanup_expired_sessions()
            time.sleep(interval)
    thread = threading.Thread(target=cleanup_loop, daemon=True)
    thread.start()
    return thread


