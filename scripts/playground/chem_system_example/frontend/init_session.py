import streamlit as st
from dotenv import load_dotenv
from frontend.utils import custom_pills
import pyperclip
import time

load_dotenv()

def init_page():
    '''
    Displays a set of pre-defined example prompts as interactive pills, allowing users to quickly copy them to the input field.
    
    Args:
        None
    
    Returns:
        None
    
    This section provides users with convenient starting points for interacting with the chatbot, 
    covering both chemical compound analysis and nanoparticle-related queries. 
    Selecting an example copies it to the clipboard and displays a brief confirmation message, 
    streamlining the process of formulating questions.
    '''

    st.set_page_config(
        page_title="🧪 Chemistry Chatbot",
        initial_sidebar_state="expanded"
    )
    st.title("🧪 Chemistry Chatbot")
    st.sidebar.image("frontend/logo_na_plashke_russkiy_belyy.png", width=150)

    init_session_state()

    '''
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
    '''
def reset_selection():
    """
    Resets the selected option and triggers a state update in the Streamlit application.
    
    This ensures that changes to available options are reflected in the user interface 
    by forcing a re-render when the selection needs to be cleared.
    
    Args:
        None
    
    Returns:
        None
    """
    st.session_state.selected_option = None
    st.session_state.reset_key += 1  # Increment the key to force update

def init_session_state():
    """
    Initializes the session state with default values, ensuring a consistent starting point for user interactions.
    
    This method populates the Streamlit session state with essential variables used throughout the application. It checks if each necessary key exists; if not, it initializes the key with a predefined default value. This setup guarantees that the application functions correctly regardless of whether the session is new or has been refreshed.
    
    Args:
        None
    
    Initializes the following session state variables:
        language (str): The user's preferred language, defaulting to 'English'.
        main_model_input (Any): The primary input for the main model, initialized to None.
        messages (list): Conversation history, starting with an initial assistant message.
        backend (Any): The selected backend, defaulting to None.
        base_url (str): The base URL for API requests, defaulting to None.
        api_key (str): The API key for accessing external services, defaulting to None.
        tavily_api_key (str): The Tavily API key for accessing external services, defaulting to None.
        images (Any): Generated images, initialized to None.
        images_b64 (Any): Base64 encoded images, initialized to None.
        selected_option (Any): The currently selected option, defaulting to None.
        reset_key (int): A key to trigger resets, initialized to 0.
        user_session (Any): Data about the current user session, defaulting to None.
        uploaded_files (dict): A dictionary to store uploaded files, initialized as empty.
        user_data_dir (str): The path to the user's data directory, defaulting to None.
    
    Returns:
        None
    """
    if 'language' not in st.session_state:
        st.session_state.language = 'English'
        
    if 'main_model_input' not in st.session_state:
        st.session_state.main_model_input = None

    if 'messages' not in st.session_state:
        st.session_state.messages = [{
            'role': 'assistant',
            'content': 'Hello! Pick models and tell me what would you like to do',
            'image_urls': None,
            'molecules_vis': None,
            'images_generated': None,
            'steps': None,
            'automl_results' : None
        }]
    
    if 'backend' not in st.session_state:
        st.session_state.backend = None

    if 'base_url' not in st.session_state:
        st.session_state.base_url = None
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = None

    if 'tavily_api_key' not in st.session_state:
        st.session_state.api_key = None

    if 'images' not in st.session_state:
        st.session_state.images = None

    if 'images_b64' not in st.session_state:
        st.session_state.images_b64 = None

    if 'selected_option' not in st.session_state:
        st.session_state.selected_option = None
    if 'reset_key' not in st.session_state:
        st.session_state.reset_key = 0
    if 'user_session_id' not in st.session_state:
        st.session_state.user_session = None
    if 'uploaded_files' not in st.session_state:
        st.session_state.uploaded_files = {}
    if 'user_data_dir' not in st.session_state:
        st.session_state.user_data_dir = None

