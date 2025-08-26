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
    Creates (if not exists) and returns the path to a temp folder for the given session_id.
    Updates the last activity timestamp.
    """
    session_folder = os.path.join(PATH_TO_TEMP_FILES, session_id)
    os.makedirs(session_folder, exist_ok=True)
    # Update last activity timestamp
    update_activity(session_folder)
    return session_folder


def save_file_to_temp_dir(uploaded_file, uploaded_file_path: str) -> None:
    with open(uploaded_file_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    update_activity(os.path.dirname(uploaded_file_path))


def delete_temp_papers(file_names: list) -> None:
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
    session_id = st.session_state.session_id
    file_path = os.path.join(ROOT_DIR, os.environ["PATH_TO_TEMP_FILES"], session_id, file_name)
    if file_path not in SELECTED_PAPERS.get(session_id, []):
        SELECTED_PAPERS.get(session_id, []).append(file_path)
    logger.info(f'SELECTED_PAPERS: {SELECTED_PAPERS}')


def deselect_file(file_name: str) -> None:
    session_id = st.session_state.session_id
    file_path = os.path.join(ROOT_DIR, os.environ["PATH_TO_TEMP_FILES"], session_id, file_name)
    if file_path in SELECTED_PAPERS.get(session_id, []):
        SELECTED_PAPERS.get(session_id, []).remove(file_path)
    logger.info(f'SELECTED_PAPERS: {SELECTED_PAPERS}')


def explore_my_papers(task: str) -> dict:
    logger.info(f'answering question based on uploaded papers')
    papers = SELECTED_PAPERS[st.session_state.session_id]
    logger.info(f'using papers: {papers}')
    logger.info(f'using model: {VISION_LLM_URL}')
    return simple_query_llm(VISION_LLM_URL, task, papers)
