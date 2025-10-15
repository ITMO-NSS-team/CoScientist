import streamlit as st
from dotenv import load_dotenv

from ChemCoScientist.frontend import chat, init_page, side_bar
from ChemCoScientist.frontend.paper_management import paper_management
from ChemCoScientist.frontend.utils import start_cleanup_thread
from definitions import CONFIG_PATH

load_dotenv(CONFIG_PATH)

import os

if __name__ == "__main__":

    path = '/app/ChemCoScientist/data_store'
    os.makedirs(os.path.join(path, 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(path, 'imgs'), exist_ok=True)
    os.makedirs(os.path.join(path, 'another'), exist_ok=True)

    start_cleanup_thread()
    init_page()
    side_bar()
    tab_chat, tab_files = st.tabs(["💬 Chat", "📁 File Management"])
    with tab_chat:
        chat()
    with tab_files:
        paper_management()

