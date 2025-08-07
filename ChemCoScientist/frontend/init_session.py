import os
import streamlit as st

from dotenv import load_dotenv
from ChemCoScientist.frontend.streamlit_endpoints import SELECTED_PAPERS
from ChemCoScientist.frontend.utils import get_user_data_dir, get_user_session_id

from definitions import ROOT_DIR

load_dotenv()


def init_page():
    st.set_page_config(
        page_title="🧪 Chemistry Chatbot", initial_sidebar_state="expanded"
    )
    st.title("🧪 Chemistry Chatbot")
    st.sidebar.image(
        os.path.join(ROOT_DIR, "ChemCoScientist/frontend/logo_na_plashke_russkiy_belyy.png"), width=150
    )

    init_session_state()

    """
    with st.container(border=False, height = 250):
        col1, mid, col2 = st.columns([1, 15, 1], vertical_alignment="bottom")
        with mid:
            selected = custom_pills("Examples to copy", ["Write smiles of acetone and calculate it's qed and molecular mass",
                                                         "Visualize (2S)-N-methyl-1-phenylpropan-2-amine and write its smiles",
                                                         "What is the iupac of hexenal? visualize it",

                                                         "What are the main ways to synthesize nanoparticles?",
                                                         "Generate an image of cube nanoparticles",
                                                         "what is the shape of nanoparticles on the submitted image?"
                                                         "Can you predict the nanoparticle shape for nanoparticles obtained from dissolution of calcium carbonate in hcl?"],
                                                         index = None, clearable=True, key="pill", reset_key=str(st.session_state.reset_key))
            if selected != None:
                st.session_state.selected_option = selected
                pyperclip.copy(st.session_state.selected_option)
                with st.empty():
                    st.write('Example copied!')
                    time.sleep(1)
                    st.write('')
                reset_selection()
    """


def reset_selection():
    """
    util funcion for custom pills
    """
    st.session_state.selected_option = None
    st.session_state.reset_key += 1  # Increment the key to force update


def init_session_state():
    if "language" not in st.session_state:
        st.session_state.language = "English"

    if "main_model_input" not in st.session_state:
        st.session_state.main_model_input = None

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! Pick models and tell me what would you like to do",
                "image_urls": None,
                "molecules_vis": None,
                "images_generated": None,
                "steps": None,
                "automl_results": None,
            }
        ]

    if "backend" not in st.session_state:
        st.session_state.backend = None

    if "base_url" not in st.session_state:
        st.session_state.base_url = None

    if "api_key" not in st.session_state:
        st.session_state.api_key = None

    if "tavily_api_key" not in st.session_state:
        st.session_state.api_key = None

    if "images" not in st.session_state:
        st.session_state.images = None

    if "images_b64" not in st.session_state:
        st.session_state.images_b64 = None

    if "selected_option" not in st.session_state:
        st.session_state.selected_option = None
    if "reset_key" not in st.session_state:
        st.session_state.reset_key = 0
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = {}
    if "uploaded_papers" not in st.session_state:
        st.session_state.uploaded_papers = []
    if "user_data_dir" not in st.session_state:
        st.session_state.user_data_dir = None
    if "session_id" not in st.session_state:
        st.session_state.session_id = get_user_session_id()
    if "user_data_dir" not in st.session_state:
        st.session_state.user_data_dir = get_user_data_dir(
            st.session_state.session_id
        )
    if st.session_state.session_id not in SELECTED_PAPERS:
        SELECTED_PAPERS[st.session_state.session_id] = []
    if "explore_mode" not in st.session_state:
        st.session_state.explore_mode = False

