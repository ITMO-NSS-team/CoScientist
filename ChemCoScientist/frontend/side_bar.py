import os

import streamlit as st
from protollm.agents.builder import GraphBuilder
from streamlit_extras.grid import GridDeltaGenerator, grid
from tools.utils import convert_to_base64
from frontend.utils import clean_folder

from .utils import file_uploader, papers_uploader


def init_language():
    with st.container(border=True):
        st.header("Select language")

        on_lang = st.selectbox(
            "Select language",
            placeholder="English",
            key="language",
            options=["English", "Русский"],
        )


def init_models():
    """
    accepts data from user and initializes llm models
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
                        init_backend()
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
                        init_backend()
                else:
                    st.write(f"The system has been initialized successfully!")


def on_provider_selected_eng(grid: GridDeltaGenerator):
    """
    accepts provider parameters from expander
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
                options=["google/gemini-2.5-pro"],
                key="visual_model_input",
                placeholder="google/gemini-2.5-pro",
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
                    "deepseek/deepseek-r1-distill-llama-70b",
                    "meta-llama/llama-3.3-70b-instruct",
                ],
                key="sc_model_input",
            )


def on_provider_selected_rus(grid: GridDeltaGenerator):
    """
    accepts provider parameters from expander
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
                    "deepseek/deepseek-r1-distill-llama-70b",
                    "meta-llama/llama-3.3-70b-instruct",
                ],
                key="sc_model_input",
            )


def init_backend():
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
    from ChemCoScientist.conf.create_conf import conf

    print(conf)
    st.session_state.backend = GraphBuilder(conf)
    # clean folder for new job here
    clean_folder(os.environ['DS_STORAGE_PATH'])
    clean_folder(os.environ['IMG_STORAGE_PATH'])
    clean_folder(os.environ['ANOTHER_STORAGE_PATH'])

def init_dataset():
    """
    Initializes dataset
    """
    dataset_files_container = st.container(border=True)
    with dataset_files_container:
        if st.session_state.language == "English":
            st.header("Dataset Files")
        else:
            st.header("Датасет")

        _render_file_uploader()


def _render_paper_uploader():
    match st.session_state.language:

        case "English":
            with st.expander("Choose paper PDF files"):
                with st.form(key="papers_files_form", border=False):
                    st.file_uploader(
                        "Choose paper files",
                        accept_multiple_files=True,
                        key="papers_file_uploader",
                        label_visibility="collapsed",
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
                    )
                    st.form_submit_button(
                        "Submit", use_container_width=True, on_click=load_papers
                    )


def _render_file_uploader():
    """
    Renders file uploader
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
    loads submited datasets to the session state on button click
    """
    files = st.session_state.file_uploader
    print(files)
    uploaded_files = file_uploader(files)
    if uploaded_files:
        # st.session_state.dataset, st.session_state.dataset_name = StreamlitDatasetLoader.load(files=[file])
        # st.toast(f"Successfully loaded dataset:\n {st.session_state.dataset_name}", icon="✅")
        st.toast(f"Successfully loaded datasets", icon="✅")


def load_papers():
    """
    loads submited papers to the session state on button click
    """
    files = st.session_state.papers_file_uploader
    print(files)
    uploaded_files = papers_uploader(files)
    if uploaded_files:
        # st.session_state.dataset, st.session_state.dataset_name = StreamlitDatasetLoader.load(files=[file])
        # st.toast(f"Successfully loaded dataset:\n {st.session_state.dataset_name}", icon="✅")
        st.toast(f"Successfully loaded papers", icon="✅")


def init_images():
    """
    initializes images
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
    initializes images
    """
    images_files_container = st.container(border=True)
    with images_files_container:
        if st.session_state.language == "English":
            st.header("Paper files")
        else:
            st.header("Статьи")
        _render_paper_uploader()


def _render_image_uploader():
    """
    renders images uploader
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
    loads submitted images to the session state on button click
    """
    files = st.session_state.images_file_uploader
    # assert max number of images, e.g. 7
    assert len(files) <= 7, (st.error("Please upload at most 7 images"), st.stop())

    if files:
        images_b64 = []
        os.makedirs(os.environ["IMG_STORAGE_PATH"], exist_ok=True)

        for image in files:
            # save the original file to dir
            file_path = os.path.join(os.environ["IMG_STORAGE_PATH"], image.name)

            with open(file_path, "wb") as f:
                f.write(image.getbuffer())

            image_b64 = convert_to_base64(image)
            images_b64.append(image_b64)

        # st.session_state.images = files
        st.session_state.images_b64 = images_b64
        st.toast(f"Successfully loaded images", icon="✅")


def side_bar():
    with st.sidebar:
        init_language()
        init_models()
        init_dataset()
        init_images()
        init_papers()

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
                    "Create a dataset for NRAS from BindingDB with IC50.",
                    "Run the ML model training on the attached data to predict Ki. Name the case 'MEK4_Ki'.",
                    "Generate an image of spherical nanoparticles.",
                    "What trained generative models do you have available?",
                    "Obtain Ki data for HRAS and NRAS proteins from all available databases.",
                    "What is the IUPAC name of hexanal? Visualize it.",
                    "Generate a drug molecule for the Alzheimer case.",
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
                    "Создай датасет для NRAS из BindingDB с IC50.",
                    "Запусти обучение ML-модели на прикрепленных мною данных для предсказания Ki. Назови кейс 'MEK4_Ki'.",
                    "Предскажи форму наноматериала, получаемого с помощью данного синтеза.",
                    "Сгенерируй изображение сферических наночастиц.",
                    "Какие обученные генеративные модели у тебя есть в наличии?",
                    "Получи данные по Ki для белков HRAS и NRAS из всех доступных баз данных.",
                    "Какой IUPAC у гексеналя? Визуализируй его.",
                    "Сгенерируй лекарственную молекулу по кейсу Альцгеймер.",
                    "Найди в интернете самые интересные статьи по лечению лейкемии и предоставь ссылки."
                ]
                for example in examples:
                    st.markdown(f"- {example}")
