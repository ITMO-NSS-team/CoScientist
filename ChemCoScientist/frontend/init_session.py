import os
import streamlit as st

from dotenv import load_dotenv
from ChemCoScientist.frontend.streamlit_endpoints import SELECTED_PAPERS
from ChemCoScientist.frontend.utils import get_user_data_dir, get_user_session_id

from definitions import ROOT_DIR

load_dotenv()


def init_page():
    """
    Displays a set of pre-defined example prompts for users to quickly input common chemistry-related queries.

    Args:
        None

    Returns:
        None
    """
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
    Resets the currently selected option and triggers a re-render of components
    that depend on it. This ensures the UI reflects a clean, unselected state.

    Args:
        None

    Returns:
        None
    """
    st.session_state.selected_option = None
    st.session_state.reset_key += 1  # Increment the key to force update


def init_session_state():
    """
    Initializes the session state with default values.

    This method sets up the initial state for the Streamlit session, ensuring that all necessary variables are present for a consistent user experience. It prepares the environment for interacting with the application, handling user preferences, data, and API keys.

    Args:
        None

    Initializes the following session state variables:
        language: The current language setting (default: "English").
        main_model_input: The input for the main model (default: None).
        messages: A list of messages representing the conversation history, starting with an initial assistant message (default: a list containing an initial assistant message).
        backend: The specified backend being used (default: None).
        base_url: The base URL for API calls (default: None).
        api_key: The API key for accessing external services (default: None).
        tavily_api_key: The Tavily API key (default: None).
        images: Stores generated images (default: None).
        images_b64: Stores base64 encoded images (default: None).
        selected_option: The currently selected option (default: None).
        reset_key: A key used to trigger a reset (default: 0).
        uploaded_files: A dictionary containing uploaded files (default: {}).
        uploaded_papers: A list of uploaded papers (default: []).
        user_data_dir: The directory for storing user-specific data (default: None).
        session_id: A unique identifier for the user's session, obtained using `get_user_session_id()` (default: generated session ID).
        explore_mode: Indicates if the exploration mode is used (default: False).

    Returns:
        None
    """
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

