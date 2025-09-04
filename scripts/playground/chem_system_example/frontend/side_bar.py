import streamlit as st
from streamlit_extras.grid import grid, GridDeltaGenerator
from graph import App
from tools.utils import convert_to_base64
from .utils import file_uploader, BASE_DATA_DIR
import os
import uuid



def init_language():
    """
    Initializes the language selection interface.
    
    This method creates a Streamlit container with a header and a selectbox
    allowing the user to choose the application's language for interacting with the application.
    The selected language is stored using the 'language' key for later use.
    
    Args:
        None
    
    Returns:
        None
    """
    with st.container(border=True):
        st.header("Select language")

        on_lang = st.selectbox("Select language", placeholder="English", key="language",
                            options=['English',
                                        "Русский"]
        )

def init_models():
    """
    Initializes Large Language Models (LLMs) based on user preferences and selected backend.
    
    This method presents a user interface to select a base URL for the LLM provider (e.g., Vsegpt, Groq). 
    Upon selection and submission, it configures the chosen backend and prepares it for use. 
    If a backend is already initialized, it displays the name of the currently active model.
    
    Args:
        None
    
    Returns:
        None
    """
    with st.container(border=True):
        match st.session_state.language: 

            case 'Русский':
                st.header("Модели")

                if not st.session_state.backend:
                    on_provider = st.selectbox("Выберите провайдера", placeholder="base url", key="api_base_url",
                                        options=["https://api.vsegpt.ru/v1",
                                            'https://api.groq.com/openai/v1',
                                                ]
                                            )
                    form_grid = grid(1, 1, 1, 1, 1, vertical_align='bottom')


                    if on_provider:
                        on_provider_selected_rus(form_grid)

                    submit = st.button(label="Submit", use_container_width=True, disabled=bool(st.session_state.backend))
                    if submit:
                        init_backend()
                else:
                    st.write(f"Name: {st.session_state.main_model_input}") 

            case 'English':   
                st.header("Models")

                if not st.session_state.backend:
                    on_provider = st.selectbox("Select base url", placeholder="base url", key="api_base_url",
                                        options=["https://api.vsegpt.ru/v1",
                                            'https://api.groq.com/openai/v1',
                                                ]
                                            )
                    form_grid = grid(1, 1, 1, 1, 1, vertical_align='bottom')


                    if on_provider:
                        on_provider_selected_eng(form_grid)

                    submit = st.button(label="Submit", use_container_width=True, disabled=bool(st.session_state.backend))
                    if submit:
                        init_backend()
                else:
                    st.write(f"Name: {st.session_state.main_model_input}")
             

def on_provider_selected_eng(grid: GridDeltaGenerator):
    """
    Presents a form to collect API keys and model selections based on the selected provider.
    
    Args:
        grid (GridDeltaGenerator): A Streamlit grid object used for layout.
    
    Returns:
        None
    """
    provider = st.session_state.api_base_url

    grid.text_input("API key", placeholder="Your API key",
                key="api_key", disabled=bool(st.session_state.backend),
                type='password')
    grid.text_input("tavily API key (optional)", placeholder="Your API key",
                key="tavily_api_key", disabled=bool(st.session_state.backend),
                type='password')

    match provider:

        case 'https://api.groq.com/openai/v1':
            grid.selectbox("Select main model", options=['llama3-70b-8192',
                                                            'llama-3.3-70b-versatile',
                                                            'llama-3.3-8b-instant',
                                                            'mixtral-8x7b-32768'],
                                        key='main_model_input')



            grid.selectbox("Select visual model", options=['llama-3.2-90b-vision-preview'],
                                        key='visual_model_input',
                                        placeholder='llama-3.2-90b-vision-preview')

        case 'https://api.vsegpt.ru/v1':
            grid.selectbox("Select main model", options=["meta-llama/llama-3.3-70b-instruct",
                                                         'meta-llama/llama-3.1-405b-instruct',
                                                                    "meta-llama/llama-3.1-8b-instruct",
                                                                    "openai/gpt-4-32k",
                                                                    "openai/gpt-4o"],
                                        key='main_model_input')



            grid.selectbox("Select visual model", options=["vis-meta-llama/llama-3.2-90b-vision-instruct"],
                                        key='visual_model_input',
                                        placeholder="vis-meta-llama/llama-3.2-90b-vision-instruct")


def on_provider_selected_rus(grid: GridDeltaGenerator):
    """
    Collects provider-specific parameters from the user via input fields.
    
    Args:
        grid (GridDeltaGenerator): A Streamlit grid object used to layout the input fields.
    
    Returns:
        None
    
    The method dynamically displays input fields for API keys (main and Tavily) and model selection
    based on the selected provider. This allows the user to configure the connection to a specific
    LLM provider and choose the desired models for text and image processing.
    """
    provider = st.session_state.api_base_url

    grid.text_input("API ключ", placeholder="Ваш API ключ",
                key="api_key", disabled=bool(st.session_state.backend),
                type='password')

    grid.text_input("API ключ для tavily (веб поиск - опционально)", placeholder="Ваш API ключ",
                key="tavily_api_key", disabled=bool(st.session_state.backend),
                type='password')

    match provider:

        case 'https://api.groq.com/openai/v1':
            grid.selectbox("Выберите главную модель", options=['llama-3-70b-8192',
                                                                    'llama-3.3-70b-versatile',
                                                                     'llama-3.3-8b-instant',
                                                                     'mixtral-8x7b-32768'],
                                        key='main_model_input')



            grid.selectbox("Выберите модель для картинок", options=['llama-3.2-90b-vision-preview'],
                                        key='visual_model_input',
                                        placeholder='llama-3.2-90b-vision-preview')

        case 'https://api.vsegpt.ru/v1':
            grid.selectbox("Выберите главную модель", options=["meta-llama/llama-3.3-70b-instruct",
                                                                'meta-llama/llama-3.1-405b-instruct',
                                                                    "meta-llama/llama-3.1-8b-instruct",
                                                                    "openai/gpt-4-32k",
                                                                    "openai/gpt-4o"],
                                        key='main_model_input')



            grid.selectbox("Выберите модель для картинок", options=["vis-meta-llama/llama-3.2-90b-vision-instruct"],
                                        key='visual_model_input',
                                        placeholder="vis-meta-llama/llama-3.2-90b-vision-instruct")

def init_backend():
    """
    Initializes the backend for the application, configuring it with necessary API keys, base URLs, and model names.
    
    This method retrieves configuration values from Streamlit's session state and environment variables, ensuring the application has the necessary credentials and settings to interact with various AI models and APIs. It dynamically sets default model inputs based on the selected base URL, providing a sensible starting point for users.
    
    Args:
        None
    
    Returns:
        None
    
    Initializes the following object properties:
        backend: An instance of the App class, configured with the retrieved API keys, base URL, and model names.
    """
    tavily_api_key = st.session_state.get('tavily_api_key')
    if tavily_api_key:
        os.environ['TAVILY_API_KEY'] = tavily_api_key
    
    api_key = st.session_state.get('api_key')
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')

    base_url = st.session_state.get('api_base_url')
    if not base_url:
        base_url = 'https://api.vsegpt.ru/v1'

    main_model_input = st.session_state.get('main_model_input')
    if not main_model_input:
        match base_url:
            case 'https://api.groq.com/openai/v1':
                main_model_input = 'llama-3.3-70b-versatile'
            case 'https://api.vsegpt.ru/v1':
                main_model_input = "meta-llama/llama-3.3-70b-instruct"

    visual_model_input = st.session_state.get('visual_model_input')
    if not visual_model_input:
        match base_url:
            case 'https://api.groq.com/openai/v1':
                visual_model_input = 'llama-3.2-90b-vision-preview'
            case 'https://api.vsegpt.ru/v1':
                visual_model_input = "vis-meta-llama/llama-3.2-90b-vision-instruct"

    st.session_state.backend = App(main_model_name=main_model_input,
                                    visual_model_name=visual_model_input,
                                    fedot_model_name='openai/gpt-4o-mini', #TODO add window in ui to choose model here
                                    base_url=base_url,
                                    api_key=api_key,
                                    tavily_api_key=tavily_api_key
                                    )

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
        if st.session_state.language == 'English':
            st.header("Dataset Files")
        else:
            st.header("Датасет")

        _render_file_uploader()

def _render_file_uploader():
    """
    Renders a file uploader component allowing users to submit dataset files for processing.
    
    Args:
        None
    
    Returns:
        None
    """
    match st.session_state.language:

        case 'English':
            with st.expander("Choose dataset files"):
                with st.form(key="dataset_files_form", border=False):
                    st.file_uploader(
                        "Choose dataset files",
                        accept_multiple_files=True,
                        key="file_uploader",
                        label_visibility='collapsed'
                    )
                    st.form_submit_button("Submit", use_container_width=True, on_click=load_dataset)
        
        case 'Русский':
             with st.expander("Выберите файлы"):
                with st.form(key="dataset_files_form", border=False):
                    st.file_uploader(
                        "Выберите файлы",
                        accept_multiple_files=True,
                        key="file_uploader",
                        label_visibility='collapsed'
                    )
                    st.form_submit_button("Submit", use_container_width=True, on_click=load_dataset)           

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
        #st.session_state.dataset, st.session_state.dataset_name = StreamlitDatasetLoader.load(files=[file])
        #st.toast(f"Successfully loaded dataset:\n {st.session_state.dataset_name}", icon="✅")
        st.toast(f"Successfully loaded dataset", icon="✅")

def init_images():
    """
    Initializes the image upload section within the Streamlit application.
    
    This section provides a header and a file uploader for users to submit image files, which are then processed as part of the document analysis workflow. The header text adapts to the user's selected language.
    
    Args:
        None
    
    Returns:
        None
    """
    images_files_container = st.container(border=True)
    with images_files_container:
        if st.session_state.language == 'English':
            st.header("Images Files")
        else:
            st.header("Изображения")
        _render_image_uploader()


def _render_image_uploader():
    """
    Presents a file uploader interface to the user, allowing them to select image files for analysis.
    
    Args:
        None
    
    Returns:
        None
    """
    match st.session_state.language:
        case 'English':
            with st.expander("Choose image files"):
                with st.form(key="image_files_form", border=False):
                    st.file_uploader(
                        "Upload an image of nanomaterial for analysis",
                        type=["png", "jpg", "jpeg", "tiff"],
                        accept_multiple_files=True,
                        key="images_file_uploader",
                        label_visibility='collapsed'
                        )

                    st.form_submit_button("Submit images", use_container_width=True, on_click=load_images)

        case "Русский":
            with st.expander("Выберите файлы"):
                with st.form(key="image_files_form", border=False):
                    st.file_uploader(
                        "Загрузите изображения наноматериалов для анализа",
                        type=["png", "jpg", "jpeg", "tiff"],
                        accept_multiple_files=True,
                        key="images_file_uploader",
                        label_visibility='collapsed'
                        )

                    st.form_submit_button("Submit images", use_container_width=True, on_click=load_images)          

def load_images():
    """
    Loads images uploaded by the user into the session state for processing.
    
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
        for image in files:
            image_b64 = convert_to_base64(image)
            images_b64.append(image_b64)


        #st.session_state.images = files
        st.session_state.images_b64 = images_b64
        st.toast(f"Successfully loaded images", icon="✅")

def side_bar():
    """
    Displays example queries in the sidebar to guide user interaction.
    
    This method initializes the backend and presents a sidebar containing example 
    queries tailored to the current language setting. These examples demonstrate 
    the types of questions the application can handle, aiding users in formulating 
    their own inquiries.
    
    Args:
        None
    
    Returns:
        None
    """
    # Display static examples at the top
    st.session_state.language = 'Русский'
    init_backend()

    with st.sidebar:
        #init_language()
        #init_models()
        init_dataset()
        init_images()
        #st.write(st.session_state.uploaded_files)

    match st.session_state.language: 
        case 'English':
            with st.expander(label="Query examples:", expanded=True):
                expander_placeholder = st.empty()
            examples = [
                "What are the main methods of nanoparticle synthesis? What methods are most suitable for drug delivery systems?",
                "For coprecipitation synthesis of drug delivery nanoparticles what nanoparticle shape is optimal?"
                "Generate a synthesis for sphere nanoparticles without toxic solvents and with numeric values for each reagent",
                "Predict shape of nanoparticles obtained by such synthesis: ...",
                "Generate an image of sphere nanoparticles",
                "What is the shape of nanoparticles in the submitted image?",
                "Generate SMILES of drug molecule of JAK1 that can be delivered by nanoparticles obtained below and predict its QED and molecular weight",
                "Calculate entrapment efficiency for such nanomaterial",
                "Write smiles of acetone and calculate its QED and molecular mass",
                "What is the IUPAC name of hexanal? Visualize it.",               
            ]
            with expander_placeholder.container(height=400):
                for example in examples:
                    st.write(f"- {example}")

        case 'Русский':
            with st.expander(label="Примеры запросов:", expanded=True):
                expander_placeholder = st.empty()

            examples = [
                "Какие существуют основные методы синтеза наноматериалов? Какой из них является наиболее подходящим для синтеза частиц, используемых для создания систем доставки лекарств?",
                "Если мы выберем синтез методом соосаждения то какая наиболее предпочтительная форма наноматериалов если мы хотим создать системы доставки лекарств на их основе?",
                "Сгенерируй синтез наноматериалов сферической формы методом соосаждения без использования токсичных растворителей с численными значениями каждого реагента.",
                "Предскажи форму наноматериала получаемого с помощью данного синтеза",
                "Сгенерируй изображение сферических наночастиц",
                "Какая форма у наночастиц на загруженном изображении?",
                "Сгенерируй SMILES лекарственной молекулы являющейся ингибитором JAK1 которую можно было бы доставлять наночастицами с приведенным ниже синтезом и предскажи ее QED и молекулярную массу.",
                "Посчитай entrapment efficiency для наноматериала с таким синтезом",
                "Напиши smiles ацетона и посчитай его QED и молекулярную массу",
                "Какой IUPAC у гексеналя? Визуализируй его.",
                "Пример синтеза: Синтез золотых наночастиц диаметром примерно 10 нанометров можно осуществить следующим образом: смешайте 0,01М хлорид золота(III) тригидрат (HAuCl4·3H2O) с 0,01М цитратом натрия (C6H5Na3O7) в воде, затем нагрейте смесь до 100°C в течение 30 минут в условиях обратного хода.",
                "Необходимо построить модель, способную предсказывать форму наноматериалов по условиям их синтеза. Форма наноматериалов закодирована в формате one hot в файле labeled_dataset.csv в колонках от cube до amorphous. При этом при одном синтезе могут получаться наноматериалы разных форм. В качестве параметров на которых нужно строить предсказания нужно использовать параметры от 'Ca ion, mM' до 'PVP' включительно."
            ]
            with expander_placeholder.container(height=400):
                for example in examples:
                    st.write(f"- {example}")
