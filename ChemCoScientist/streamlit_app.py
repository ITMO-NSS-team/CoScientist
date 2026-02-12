import streamlit as st
from dotenv import load_dotenv

from ChemCoScientist.frontend import chat, init_page, side_bar
from ChemCoScientist.frontend.paper_management import paper_management
from ChemCoScientist.frontend.dataset_management import dataset_management
from ChemCoScientist.frontend.utils import start_cleanup_thread
from ChemCoScientist.memory.json_db import JSONFileDB
from definitions import ROOT_DIR, CONFIG_PATH

load_dotenv(CONFIG_PATH)

import os

if __name__ == "__main__":

    path = f'{ROOT_DIR}/app/ChemCoScientist/data_store'
    os.makedirs(os.path.join(path, 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(path, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(path, 'another'), exist_ok=True)

    

    db = JSONFileDB(os.environ.get('MEMORY_DB_PATH', 'ChemCoScientist/data_store/files_db.json'))

    start_cleanup_thread()
    init_page()

    backend_type = os.environ.get('BACKEND_TYPE', "ChemCoScientist")
    side_bar(backend=backend_type)
    tab_chat, tab_files, tab_datasets = st.tabs(["💬 Chat", "📁 File Management", "Available Datasets"])

    with tab_chat:
        chat()
    with tab_files:
        paper_management()
    with tab_datasets:
        dataset_management(db)
