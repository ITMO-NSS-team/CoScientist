import os
import time
import io
import zipfile

import streamlit as st
import base64
import streamlit.components.v1 as components
#from protollm.agents.builder import GraphBuilder
from streamlit_extras.grid import GridDeltaGenerator, grid
from ChemCoScientist.tools.utils import convert_to_base64
from ChemCoScientist.logger import logger
from ChemCoScientist.frontend.utils import file_uploader, clean_folder
from ChemCoScientist.frontend.streamlit_endpoints import process_uploaded_paper
from ChemCoScientist.memory.json_db import JSONFileDB
from ChemCoScientist.memory.memory_manager import MemoryGraph
from definitions import ROOT_DIR


def init_language():
    """
    Initializes the language selection interface.
    
    This method creates a Streamlit container with a header and a selectbox
    allowing the user to choose a language. This enables the application to 
    present information and interact with the user in their preferred language.
    The selected language is stored using a Streamlit session state key
    called "language".
    
    Args:
        None
    
    Returns:
        None
    """
    with st.container(border=True):
        st.header("Select language")

        on_lang = st.selectbox(
            "Select language",
            placeholder="Русский",
            key="language",
            options=["English", "Русский"],
        )


def init_models():
    """
    Initializes Large Language Models (LLMs) based on user preferences and selected backend.
    
    This method presents a user interface to select a base URL for the LLM provider (e.g., OpenRouter, Groq). 
    Once a provider is chosen and submitted, the backend is initialized using the `init_backend()` function. 
    If the backend is already initialized, it displays a success message. 
    
    Args:
        None
    
    Returns:
        None
    """
    with st.container(border=True):
        match st.session_state.language:

            case "Русский":
                st.header("Модели")

                if not st.session_state.backend:
                    on_provider = st.selectbox(
                        "Выберите провайдера",
                        placeholder="base url",
                        key="api_base_url",
                        options=[
                            "https://openrouter.ai/api/v1",
                            "https://api.groq.com/openai/v1",
                        ],
                    )
                    form_grid = grid(1, 1, 1, 1, 1, vertical_align="bottom")

                    if on_provider:
                        on_provider_selected_rus(form_grid)

                    submit = st.button(
                        label="Submit",
                        use_container_width=True,
                        disabled=bool(st.session_state.backend),
                    )
                    if submit:
                        init_backend(backend='ChemCoScientist')
                else:
                    st.write(f"Система успешно инициализированна!")

            case "English":
                st.header("Models")

                if not st.session_state.backend:
                    on_provider = st.selectbox(
                        "Select base url",
                        placeholder="base url",
                        key="api_base_url",
                        options=[
                            "https://openrouter.ai/api/v1",
                            "https://api.groq.com/openai/v1",
                        ],
                    )
                    form_grid = grid(1, 1, 1, 1, 1, vertical_align="bottom")

                    if on_provider:
                        on_provider_selected_eng(form_grid)

                    submit = st.button(
                        label="Submit",
                        use_container_width=True,
                        disabled=bool(st.session_state.backend),
                    )
                    if submit:
                        init_backend(backend='ChemCoScientist')
                else:
                    st.write(f"The system has been initialized successfully!")


def on_provider_selected_eng(grid: GridDeltaGenerator):
    """
    Presents a form to configure the large language model (LLM) provider and model options based on the selected provider.
    
    Args:
        grid (GridDeltaGenerator): A Streamlit grid object used to layout the form elements.
    
    Returns:
        None
    """
    provider = st.session_state.api_base_url

    grid.text_input(
        "API key",
        placeholder="Your API key",
        key="api_key",
        disabled=bool(st.session_state.backend),
        type="password",
    )

    # used DuckDuckGo by default

    # grid.text_input("tavily API key (optional)", placeholder="Your API key",
    #             key="tavily_api_key", disabled=bool(st.session_state.backend),
    #             type='password')

    match provider:
        case "https://api.groq.com/openai/v1":
            grid.selectbox(
                "Select main model",
                options=[
                    "groq/deepseek-r1-distill-llama-70b",
                    "llama-3.3-70b-versatile",
                ],
                key="main_model_input",
                placeholder="llama-3.3-70b-versatile",
            )
            grid.selectbox(
                "Select visual model",
                options=["llama-3.2-90b-vision-preview"],
                key="visual_model_input",
                placeholder="llama-3.2-90b-vision-preview",
            )

            grid.selectbox(
                "Select model for scenarion agent",
                options=[
                    "groq/deepseek-r1-distill-llama-70b",
                    "groq/llama-3.3-70b-versatile",
                ],
                key="sc_model_input",
                placeholder="groq/deepseek-r1-distill-llama-70b",
            )

        case "https://openrouter.ai/api/v1":
            grid.selectbox(
                "Select main model",
                options=[
                    "deepseek/deepseek-r1-distill-llama-70b",
                    "meta-llama/llama-3.3-70b-instruct",
                ],
                key="main_model_input",
            )

            grid.selectbox(
                "Select visual model",
                options=["google/gemini-2.5-pro"],
                key="visual_model_input",
                placeholder="google/gemini-2.5-pro",
            )
            grid.selectbox(
                "Select model for scenarion agent",
                options=[
                    "google/gemini-2.5-pro",
                    "deepseek/deepseek-r1-distill-llama-70b",
                    "meta-llama/llama-3.3-70b-instruct",
                    "openai/o1"
                ],
                key="sc_model_input",
            )


def on_provider_selected_rus(grid: GridDeltaGenerator):
    """
    Accepts provider parameters from the expander to configure the language model settings.
    
    Args:
        grid (GridDeltaGenerator): A Streamlit grid object used for layout and input elements.
    
    Returns:
        None
    """
    provider = st.session_state.api_base_url

    grid.text_input(
        "API ключ",
        placeholder="Ваш API ключ",
        key="api_key",
        disabled=bool(st.session_state.backend),
        type="password",
    )

    # grid.text_input("API ключ для tavily (веб поиск - опционально)", placeholder="Ваш API ключ",
    #             key="tavily_api_key", disabled=bool(st.session_state.backend),
    #             type='password')

    match provider:
        case "https://api.groq.com/openai/v1":
            grid.selectbox(
                "Выберите главную модель",
                options=[
                    "groq/deepseek-r1-distill-llama-70b",
                    "llama-3.3-70b-versatile",
                ],
                key="main_model_input",
                placeholder="llama-3.3-70b-versatile",
            )
            grid.selectbox(
                "Выберите модель для картинок",
                options=["llama-3.2-90b-vision-preview"],
                key="visual_model_input",
                placeholder="llama-3.2-90b-vision-preview",
            )

            grid.selectbox(
                "Выберите модель для сценарных агентов",
                options=[
                    "groq/deepseek-r1-distill-llama-70b",
                    "groq/llama-3.3-70b-versatile",
                ],
                key="sc_model_input",
                placeholder="groq/deepseek-r1-distill-llama-70b",
            )

        case "https://openrouter.ai/api/v1":
            grid.selectbox(
                "Выберите главную модель",
                options=[
                    "deepseek/deepseek-r1-distill-llama-70b",
                    "meta-llama/llama-3.3-70b-instruct",
                ],
                key="main_model_input",
                placeholder="deepseek/deepseek-r1-distill-llama-70b",
            )

            grid.selectbox(
                "Выберите модель для картинок",
                options=["vis-meta-llama/llama-3.2-90b-vision-instruct"],
                key="visual_model_input",
                placeholder="vis-meta-llama/llama-3.2-90b-vision-instruct",
            )
            grid.selectbox(
                "Выберите модель для сценарных агентов",
                options=[
                    "google/gemini-2.5-pro",
                    "deepseek/deepseek-r1-distill-llama-70b",
                    "meta-llama/llama-3.3-70b-instruct",
                    "openai/o1"
                ],
                key="sc_model_input",
            )


def init_backend(backend):
    """
    Initializes the backend for the application, configuring it with API keys and models from session state.
    
    This method retrieves API keys, base URLs, and model inputs from the session state and sets them as environment variables,
    allowing the application to connect to and leverage Large Language Models (LLMs) and vision models. It then initializes the 
    GraphBuilder with a configuration object for processing scientific information and prepares storage locations 
    for new analyses.
    
    Args:
        None
    
    Initializes the following class fields:
        backend: An instance of GraphBuilder, initialized with the application configuration.
    
    Returns:
        None
    """
    # by deafault in ChemCoSc duckduckgo without key

    # tavily_api_key = st.session_state.get('tavily_api_key')
    # if tavily_api_key:
    #     os.environ['TAVILY_API_KEY'] = tavily_api_key

    api_key = st.session_state.get("api_key")
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    base_url = st.session_state.get("api_base_url")
    if base_url:
        os.environ["MAIN_LLM_URL"] = base_url
        os.environ["SCENARIO_LLM_URL"] = base_url

    sc_model_input = st.session_state.get("sc_model_input")
    if sc_model_input:
        os.environ["SCENARIO_LLM_MODEL"] = sc_model_input

    main_model_input = st.session_state.get("main_model_input")
    if main_model_input:
        os.environ["MAIN_LLM_MODEL"] = main_model_input

    visual_model_input = st.session_state.get("visual_model_input")
    if visual_model_input:
        # TODO: add model from user input
        os.environ["VISION_LLM_URL"] = os.environ["VISION_LLM_URL"]

    # it must be here !!!
    if backend == 'ChemCoScientist':
        from ChemCoScientist.conf.create_conf import conf
    elif backend == 'MedCoScientist':
        from MedCoScientist.conf.create_conf import conf
    else:
        raise ValueError(f'Wrong backend value: {backend}')
        
    conf['configurable']['logger'] = logger
    conf['files_db'] = JSONFileDB(os.environ.get('MEMORY_DB_PATH', 'ChemCoScientist/data_store/files_db.json'))
    conf['configurable']['session_id'] = st.session_state.session_id
    #st.session_state.backend = GraphBuilder(conf)
    if "backend" not in st.session_state:
        st.session_state.backend = MemoryGraph(config=conf, llm=conf['configurable']['llm'], logger=logger)


def init_dataset():
    """
    Initializes the dataset section of the application, displaying a header and file uploader.
    
    Args:
        None
    
    Returns:
        None
    """
    dataset_files_container = st.container(border=True)
    with dataset_files_container:
        if st.session_state.language == "English":
            st.header("Dataset Files")
        else:
            st.header("Датасет")

        _render_file_uploader()


def _render_paper_uploader():
    """
    Renders a file uploader for PDF papers.
    
    This method displays a Streamlit expander containing a form with a file uploader
    specifically for PDF files, allowing multiple files to be selected. 
    The labels and help text dynamically adjust based on the user's selected language 
    (English or Russian) to provide a localized user experience.
    Submitting the form triggers the `load_papers` function to process the uploaded files.
    
    Args:
        None
    
    Returns:
        None
    """
    match st.session_state.language:

        case "English":
            with st.expander("Choose paper PDF files"):
                with st.form(key="papers_files_form", border=False):
                    st.file_uploader(
                        "Choose paper files",
                        accept_multiple_files=True,
                        key="papers_file_uploader",
                        label_visibility="collapsed",
                        type=['pdf'],
                        help="Supported formats: PDF",
                    )
                    st.form_submit_button(
                        "Submit", use_container_width=True, on_click=load_papers
                    )

        case "Русский":
            with st.expander("Выбери статьи в PDF"):
                with st.form(key="papers_files_form", border=False):
                    st.file_uploader(
                        "Выберите статьи в PDF",
                        accept_multiple_files=True,
                        key="papers_file_uploader",
                        label_visibility="collapsed",
                        type=['pdf'],
                        help="Поддерживаемые форматы: PDF",
                    )
                    st.form_submit_button(
                        "Submit", use_container_width=True, on_click=load_papers
                    )


def _render_file_uploader():
    """
    Renders a file uploader component in the Streamlit application, allowing users to upload dataset files. 
    
    The uploader adapts to the user's selected language (English or Russian), presenting the interface in the corresponding language.  Upon file selection and submission, the `load_dataset` function is triggered to process the uploaded data. This enables users to provide their own data for analysis alongside existing resources.
    
    Args:
        None
    
    Returns:
        None
    """
    match st.session_state.language:
        case "English":
            with st.expander("Choose dataset files"):
                with st.form(key="dataset_files_form", border=False):
                    st.file_uploader(
                        "Choose dataset files",
                        accept_multiple_files=True,
                        key="file_uploader",
                        label_visibility="collapsed",
                    )
                    st.form_submit_button(
                        "Submit", use_container_width=True, on_click=load_dataset
                    )

        case "Русский":
            with st.expander("Выберите файлы"):
                with st.form(key="dataset_files_form", border=False):
                    st.file_uploader(
                        "Выберите файлы",
                        accept_multiple_files=True,
                        key="file_uploader",
                        label_visibility="collapsed",
                    )
                    st.form_submit_button(
                        "Submit", use_container_width=True, on_click=load_dataset
                    )


def load_dataset():
    """
    Loads datasets uploaded by the user into the session state.
    
    Args:
        None
    
    Returns:
        None
    """
    files = st.session_state.file_uploader
    uploaded_files = file_uploader(files)
    if uploaded_files:
        # st.session_state.dataset, st.session_state.dataset_name = StreamlitDatasetLoader.load(files=[file])
        # st.toast(f"Successfully loaded dataset:\n {st.session_state.dataset_name}", icon="✅")
        st.toast(f"Successfully loaded datasets", icon="✅")


def load_papers():  # TODO: add russian version
    """
    Loads and processes uploaded scientific papers, storing their metadata in the session state.

    Args:
        None

    Returns:
        None
    """
    uploaded_papers = st.session_state.papers_file_uploader
    if uploaded_papers is not None:
        # Process file here
        st.write("File uploaded and processed")

        if uploaded_papers:
            new_files_processed = False

            with st.spinner("Processing uploaded files..."):
                for uploaded_file in uploaded_papers:
                    if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_papers]:
                        try:
                            # Process the uploaded file
                            result = process_uploaded_paper(uploaded_file)

                            if result["success"]:
                                st.session_state.uploaded_papers.append({
                                    "name": uploaded_file.name,
                                    "size": uploaded_file.size,
                                    "type": uploaded_file.type
                                })
                                # st.success(f"✅ Successfully processed: {uploaded_file.name}")
                                new_files_processed = True
                            else:
                                st.error(f"❌ Error processing file: {result['error']}")
                        except Exception as e:
                            st.error(f"❌ Unexpected error processing {uploaded_file.name}: {str(e)}")


def init_images():
    """
    Initializes the image upload section of the application.
    
    This section provides a user interface for uploading images relevant to the analysis. The header displayed adapts to the user's selected language.
    
    Args:
        None
    
    Returns:
        None
    """
    images_files_container = st.container(border=True)
    with images_files_container:
        if st.session_state.language == "English":
            st.header("Images Files")
        else:
            st.header("Изображения")
        _render_image_uploader()


def init_papers():
    """
    Initializes the paper file uploader interface.
    
    This method creates a container to display a header and render a file uploader for scientific papers. The header text adapts to the user's selected language (English or Russian).
    
    Args:
        None
    
    Returns:
        None
    """
    images_files_container = st.container(border=True)
    with images_files_container:
        if st.session_state.language == "English":
            st.header("Paper files")
        else:
            st.header("Статьи")
        _render_paper_uploader()


def init_downloaded_papers():
    """
    Initializes the Downloaded Papers section in the sidebar and renders
    the list of papers the agent downloaded.
    """
    downloads_container = st.container(border=True)
    with downloads_container:
        if st.session_state.language == "English":
            st.header("Papers Search Results")
        else:
            st.header("Результаты поиска статей")
        _render_downloaded_papers()


def _render_downloaded_papers():
    """
    Renders a list of papers downloaded by the papers-search agent and allows importing them
    into the session (copies into the temp session folder and processes them the same way
    as uploaded files).
    """
    downloads_dir = os.path.join(ROOT_DIR, "downloaded_papers")
    os.makedirs(downloads_dir, exist_ok=True)
    files = [f for f in os.listdir(downloads_dir) if f.lower().endswith(".pdf")]

    match st.session_state.language:
        case "English":
            title = "Papers Search Results"
            import_label = "Import"
        case _:
            title = "Результаты поиска статей"
            import_label = "Импортировать"

    with st.expander(title):
        st.markdown(
            "<div style='margin:0 0 0 0; font-size:14px; padding-bottom:2px;'>Papers downloaded by the agent will appear here.</div>",
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <style>
            /* Make Streamlit buttons full-width and center the text; reduce vertical margin */
            .stButton>button {
                width: 100% !important;
                text-align: center !important;
                margin: 4px 0 !important;
                padding: 8px 0 !important;
            }
            /* Slightly tighten the container spacing around the buttons */
            .stExpander .stButton {
                margin-bottom: 4px !important;
            }
            /* Reduce top margin for multiselect inside expander to tighten vertical gap */
            .stMultiSelect, .stMultiSelect > div {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
            /* Reduce expander internal padding */
            .stExpander>div {
                padding-top: 4px !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
        selected = st.multiselect(
            "Select papers",
            options=files,
            key="downloaded_papers_selection",
            help="Select papers to import for processing",
            label_visibility="collapsed",
        )

        if st.button(import_label, key="import_downloaded_papers", use_container_width=True):
            load_downloaded_papers()

        selection = st.session_state.get("downloaded_papers_selection", []) or []
        if selection:
            # Single-button download: create an in-memory ZIP of selected PDFs and show one download button
            match st.session_state.language:
                case "English":
                    dl_label = "Download selected"
                case _:
                    dl_label = "Скачать выбранные"

            # Build ZIP immediately (small numbers of files expected). If files are large, consider streaming.
            buf = io.BytesIO()
            added = 0
            with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
                for fname in selection:
                    path = os.path.join(downloads_dir, fname)
                    try:
                        if not os.path.exists(path):
                            logger.error(f"File not found: {path}")
                            continue
                        with open(path, "rb") as f:
                            data = f.read()
                            if not data.startswith(b"%PDF"):
                                logger.error(f"Invalid PDF format: {path}")
                                continue
                            zf.writestr(fname, data)
                            added += 1
                    except Exception as e:
                        logger.exception(f"Error adding {fname} to zip: {e}")

            if added == 0:
                st.warning("No valid PDFs selected to download")
            else:
                buf.seek(0)
                zip_name = f"papers_{time.strftime('%Y%m%d_%H%M%S')}.zip"
                st.download_button(
                    label=dl_label,
                    data=buf.getvalue(),
                    file_name=zip_name,
                    mime="application/zip",
                    use_container_width=True,
                    key="download_selected_zip"
                )


def load_downloaded_papers():
    """
    Imports selected files from the `downloaded_papers/` folder and processes them
    using the existing `process_uploaded_paper` endpoint (by wrapping file bytes
    in a small object compatible with Streamlit's UploadedFile `.getvalue()` API).
    """
    selection = st.session_state.get("downloaded_papers_selection", []) or []
    if not selection:
        st.toast("No papers selected", icon="⚠️")
        return

    downloads_dir = os.path.join(ROOT_DIR, "downloaded_papers")
    new_files_processed = False
    with st.spinner("Importing selected downloaded papers..."):
        for fname in selection:
            # avoid duplicating already-processed uploads
            if fname in [f["name"] for f in st.session_state.uploaded_papers]:
                continue
            path = os.path.join(downloads_dir, fname)
            try:
                class LocalFile:
                    def __init__(self, path, name):
                        self.name = name
                        with open(path, "rb") as _f:
                            self._data = _f.read()

                    def getvalue(self):
                        return self._data

                lf = LocalFile(path, fname)
                result = process_uploaded_paper(lf)
                if result.get("success"):
                    st.session_state.uploaded_papers.append({
                        "name": fname,
                        "size": os.path.getsize(path),
                        "type": "application/pdf",
                    })
                    new_files_processed = True
                else:
                    st.error(f"❌ Error processing file: {fname} - {result.get('msg','')}")
            except Exception as e:
                st.error(f"❌ Unexpected error processing {fname}: {e}")

    if new_files_processed:
        st.toast("Imported downloaded papers", icon="✅")


def _render_image_uploader():
    """
    Renders an interface for uploading images of nanomaterials for analysis.
    
    Args:
        None
    
    Returns:
        None
    """
    match st.session_state.language:
        case "English":
            with st.expander("Choose image files"):
                with st.form(key="image_files_form", border=False):
                    st.file_uploader(
                        "Upload an image of nanomaterial for analysis",
                        type=["png", "jpg", "jpeg", "tiff"],
                        accept_multiple_files=True,
                        key="images_file_uploader",
                        label_visibility="collapsed",
                    )

                    st.form_submit_button(
                        "Submit images", use_container_width=True, on_click=load_images
                    )

        case "Русский":
            with st.expander("Выберите файлы"):
                with st.form(key="image_files_form", border=False):
                    st.file_uploader(
                        "Загрузите изображения наноматериалов для анализа",
                        type=["png", "jpg", "jpeg", "tiff"],
                        accept_multiple_files=True,
                        key="images_file_uploader",
                        label_visibility="collapsed",
                    )

                    st.form_submit_button(
                        "Submit images", use_container_width=True, on_click=load_images
                    )


def load_images():
    """
    Loads images uploaded by the user and stores them for further processing.
    
    Args:
        None
    
    Returns:
        None
    """
    files = st.session_state.images_file_uploader
    # assert max number of images, e.g. 7
    assert len(files) <= 7, (st.error("Please upload at most 7 images"), st.stop())

    if files:
        images_b64 = []
        os.makedirs(os.path.join(ROOT_DIR, os.environ["IMG_STORAGE_PATH"]), exist_ok=True)

        for image in files:
            # save the original file to dir
            file_path = os.path.join(ROOT_DIR, os.environ["IMG_STORAGE_PATH"], image.name)
            print(f'IMAGE PATH: {file_path}')

            with open(file_path, "wb") as f:
                f.write(image.getbuffer())

            print('IMAGE SAVED')
            image_b64 = convert_to_base64(image)
            images_b64.append(image_b64)

        # st.session_state.images = files
        st.session_state.images_b64 = images_b64
        st.session_state.images = file_path
        st.toast(f"Successfully loaded images", icon="✅")


def side_bar(backend='ChemCoScientis'):
    """
    Initializes the Streamlit sidebar with options for configuring the analysis environment 
    and provides example queries to guide user interaction.
    
    This method sets up the sidebar with controls for selecting the language, model, dataset,
    uploading images, and specifying research papers. It then displays tailored instructions 
    and example queries based on the chosen language to facilitate effective interaction
    with the application.
    
    Args:
        None
    
    Returns:
        None
    """
    # Display static examples at the top
    # st.session_state.language = 'Русский'

    # uncomment for start without pass model, key, etc (from gui)
    init_backend(backend)

    with st.sidebar:
        init_language()
        init_models()
        init_dataset()
        init_images()
        init_papers()
        init_downloaded_papers()

    match st.session_state.language:
        case "English":
            # Показываем описание отдельно
            st.markdown(
                "**Be sure to fill in the fields (on the left) for your model selection before starting work!**\n\n"
                "**You can ask questions or make requests in the chat.**\n\n"
                "If you want to attach an image, figure, or article and ask to process it:\n"
                "1. Upload it using the windows on the left\n"
                "2. Then ask your question or make a request in the chat"
            )

            with st.expander("Query examples:", expanded=True):
                examples = [
                    "What can you do?",
                    "Download data for SARS-CoV-2 from BindingDb with IC50 values.",
                    "Download data for GSK from ChemBL with IC50 values.",
                    "Run the ML model training on the attached data to predict Ki. Name the case 'MEK4_Ki'.",
                    "Generate an image of spherical nanoparticles.",
                    "What trained generative models do you have available?",
                    "Obtain Ki data for Glycogen synthase kinase-3 beta and MEK1 proteins from all available databases.",
                    "What is the IUPAC name of hexanal?",
                    "Generate a drug molecule for the Alzheimer case by generatime model.",
                    "Find the most interesting articles on leukemia treatment on the Internet and provide links.",
                ]
                for example in examples:
                    st.markdown(f"- {example}")

        case "Русский":
            st.markdown(
                "**Обязательно заполни поля (слева) по настройке модели перед началом работы!**\n\n"
                "**Ты можешь задавать вопросы и писать просьбы в чат.**\n\n"
                "Если хочешь прикрепить изображение, картинку или статью и попросить что-то сделать:\n"
                "1. Загрузи файл с помощью окон слева\n"
                "2. После загрузки задай вопрос или напиши просьбу в чат"
            )

            with st.expander("Примеры запросов:", expanded=True):
                examples = [
                    "Что ты умеешь?",
                    "Скачай данные для белка SARS-CoV-2 из BindingDb с рассчитанным IC50.",
                    "Скачай данные для белка GSK из ChemBL с рассчитанным IC50.",
                    "Запусти обучение ML-модели на прикрепленных мною данных для предсказания Ki. Назови кейс 'MEK4_Ki'.",
                    "Предскажи форму наноматериала, получаемого с помощью данного синтеза.",
                    "Сгенерируй изображение сферических наночастиц.",
                    "Какие обученные генеративные модели у тебя есть в наличии?",
                    "Получи данные по Ki для белков Glycogen synthase kinase-3 beta и MAP2K1 из всех доступных баз данных.",
                    "Какой IUPAC у гексеналя?",
                    "Сгенерируй лекарственную молекулу c помощью генеративной модели по кейсу Альцгеймер.",
                    "Найди в интернете самые интересные статьи по лечению лейкемии и предоставь ссылки."
                ]
                for example in examples:
                    st.markdown(f"- {example}")
