import os
import sys
import streamlit as st

from dotenv import load_dotenv
from pathlib import Path

from ChemCoScientist.paper_analysis.question_processing import simple_query_llm
from ChemCoScientist.frontend.memory import SELECTED_PAPERS
from ChemCoScientist.frontend.utils import update_activity
from definitions import CONFIG_PATH, ROOT_DIR

load_dotenv(CONFIG_PATH)
PATH_TO_TEMP_FILES = os.path.join(ROOT_DIR, os.environ["PATH_TO_TEMP_FILES"])

logger = st.logger.get_logger(__name__)
logger.info(str(Path(__file__).resolve().parent.parent))

sys.path.append(str(Path(__file__).resolve().parent.parent))

VISION_LLM_URL = os.environ["VISION_LLM_URL"]


def process_uploaded_paper(uploaded_file) -> dict:
    """
    Processes an uploaded paper file, saving it for subsequent analysis.
    
    This method securely stores the uploaded file in a temporary directory, preparing it for extraction of relevant information. It includes error handling to ensure robustness during the file saving process.
    
    Args:
        uploaded_file: The file object representing the uploaded scientific paper.
    
    Returns:
        dict: A dictionary indicating the success or failure of the file processing.
              'success' is True if the file was saved successfully, False otherwise.
              'msg' contains an error message if saving failed, or an empty string if successful.
    """
    print('inside process_uploaded_paper')
    res = {'success': False, 'msg': ''}

    files_path = get_session_temp_folder(st.session_state.session_id)

    uploaded_file_path = Path(files_path, uploaded_file.name)
    print(f'uploaded_file_path: {uploaded_file_path}')
    try:
        save_file_to_temp_dir(uploaded_file, uploaded_file_path)
        logger.info('file uploaded')
    except Exception as e:
        Path(uploaded_file_path).unlink(missing_ok=True)
        logger.error(f'Could not process file: {e}')
        res['msg'] = res['msg'] + f' Could not upload file: {uploaded_file_path}'

    res['success'] = True
    logger.info('finished processing')
    return res


def get_session_temp_folder(session_id: str) -> str:
    """
    Creates a temporary folder for a given session to store intermediate files and data.
    If the folder does not exist, it is created.  Updates a timestamp to track recent activity within the session folder.
    
    Args:
        session_id (str): A unique identifier for the session.
    
    Returns:
        str: The path to the session's temporary folder.
    """
    session_folder = os.path.join(PATH_TO_TEMP_FILES, session_id)
    os.makedirs(session_folder, exist_ok=True)
    # Update last activity timestamp
    update_activity(session_folder)
    return session_folder


def save_file_to_temp_dir(uploaded_file, uploaded_file_path: str) -> None:
    """
    Saves the content of an uploaded file to a temporary directory for processing.
    
    Args:
        uploaded_file: The file object containing the file content.
        uploaded_file_path: The full path, including filename, where the file will be stored.
    
    Returns:
        None
    """
    with open(uploaded_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    update_activity(os.path.dirname(uploaded_file_path))


def delete_temp_papers(file_names: list) -> None:
    """
    Deletes temporary paper files associated with the current session.
    
    This method removes files uploaded by the user from temporary storage. It constructs the full file paths based on the session ID and file names,
    then deletes each file.  Deleting a file triggers logging of the event and updates the application state to reflect the removal.
    
    Args:
        file_names (list): A list of dictionaries, where each dictionary contains the 'name' of the file to delete.
    
    Returns:
        None
    """
    session_id = st.session_state.session_id
    file_paths = [os.path.join(ROOT_DIR, os.environ["PATH_TO_TEMP_FILES"], session_id, file_name['name'])
                  for file_name in file_names]
    logger.info(f'delete files: {file_paths}')
    for file_path in file_paths:
        Path(file_path).unlink(missing_ok=True)
        logger.info(f'paper deleted: {file_path}')
        update_activity(os.path.dirname(file_path))
        deselect_file(file_path)


def select_file(file_name: str) -> None:
    """
    Adds a file path to the list of papers associated with the current session.
    
    This method ensures that the system keeps track of which papers a user has selected
    for analysis within their current session. It constructs the full file path, 
    and adds it to the session's list of selected papers if it's not already present,
    allowing the system to focus on the user's chosen documents.
    
    Args:
        file_name (str): The name of the file to select.
    
    Returns:
        None
    """
    session_id = st.session_state.session_id
    file_path = os.path.join(ROOT_DIR, os.environ["PATH_TO_TEMP_FILES"], session_id, file_name)
    if file_path not in SELECTED_PAPERS.get(session_id, []):
        SELECTED_PAPERS.get(session_id, []).append(file_path)
    logger.info(f'SELECTED_PAPERS: {SELECTED_PAPERS}')


def deselect_file(file_name: str) -> None:
    """
    Removes a file from the list of papers currently being considered in the session.
    
    Args:
        file_name (str): The name of the file to deselect.
    
    Returns:
        None
    """
    session_id = st.session_state.session_id
    file_path = os.path.join(ROOT_DIR, os.environ["PATH_TO_TEMP_FILES"], session_id, file_name)
    if file_path in SELECTED_PAPERS.get(session_id, []):
        SELECTED_PAPERS.get(session_id, []).remove(file_path)
    logger.info(f'SELECTED_PAPERS: {SELECTED_PAPERS}')


def explore_my_papers(task: str) -> dict:
    '''
    Answers a question using the content of uploaded papers.
    
    Args:
        task (str): The question or task to be answered, formulated as a text prompt.
    
    Returns:
        dict: The response from the Large Language Model (LLM) query, containing the answer and related metadata.
    '''
    logger.info(f'answering question based on uploaded papers')
    papers = SELECTED_PAPERS[st.session_state.session_id]
    logger.info(f'using papers: {papers}')
    logger.info(f'using model: {VISION_LLM_URL}')
    return simple_query_llm(VISION_LLM_URL, task, papers)
