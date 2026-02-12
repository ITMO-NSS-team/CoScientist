import asyncio
import glob
import logging
import os
import streamlit as st
import threading

from io import BytesIO
from langgraph.errors import GraphRecursionError
from pathlib import Path
from PIL import Image
from queue import Queue, Empty
from urllib.parse import urlparse

from definitions import ROOT_DIR
from ChemCoScientist.frontend.utils import get_user_data_dir, get_user_session_id, save_all_files
from ChemCoScientist.tools.utils import convert_to_base64, convert_to_html
from CoScientist.paper_parser.s3_connection import s3_service

# Create a separate logger for chat.py
logger = logging.getLogger("chat_logger")
logger.setLevel(logging.INFO)

# Configure a file handler for the chat logger
file_handler = logging.FileHandler("chat.log")
file_handler.setLevel(logging.INFO)

# Set a formatter for the chat logger
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

# Add the handler to the chat logger
logger.addHandler(file_handler)


os.environ.get("ML_TOOLS_IP")

def chat():
    """
    Displays the chat interface and handles user interactions for querying scientific papers.

    This method manages the display of chat messages, handles user input, and interacts with other functions to process and display responses.
    It allows users to interact with the assistant, optionally choosing to analyze uploaded papers directly instead of relying on a pre-existing database.  The interface also supports displaying intermediate reasoning steps, AutoML results, and visualizations of generated images and molecules.

    Args:
        None

    Returns:
        None
    """
    st.session_state.explore_mode = st.checkbox(
        "🔍 Explore My Papers",
        help="When checked, assistant will search through your uploaded papers instead of using the database",
        key="explore_papers_mode"
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and message.get("steps"):
                with st.expander(
                    f"🔍 Intermediate Thoughts (click to expand)", expanded=False
                ):
                    for step in message["steps"]:
                        st.markdown(step)
            # st.markdown(message['content'])

            if message.get("automl_results"):
                st.markdown(message["content"])
                st.markdown(message["automl_results"])
            else:
                st.markdown(message["content"])

            if message.get('found_pubmed_papers'):
                papers = message.get('found_pubmed_papers')
                st.subheader("Список найденных статей:")

                for idx, paper in enumerate(papers):
                    with st.expander(
                        f"{idx+1}. {paper.title}", expanded=False
                    ):

                        authors = ', '.join(paper.authors)
                        st.markdown(f"**Авторы:** {authors}")
                        st.markdown(f"**Год:** {paper.year}")
                        st.markdown(f"**Журнал:** {paper.journal}")
                        st.markdown(f"**Аннотация:** {paper.abstract}")
                        
                        if paper.link:
                            st.markdown(f"[Перейти к статье]({paper.link})")


            gen_imgs = message.get("images_generated")

            # if imgs := message.get("image_urls"):  # render previously submitted images
            #     for img in imgs:
            #         st.components.v1.html(convert_to_html(img), height=400)

            if mols := message.get(
                "molecules_vis"
            ):  # render previously visualized molecules
                for mol in mols:
                    st.components.v1.html(mol, height=400)

            if gen_imgs := message.get(
                "images_generated"
            ):  # render previously generated images
                for img in gen_imgs:
                    st.components.v1.html(convert_to_html(img), height=200)

            # Display paper analysis metadata if present (for previously stored messages)
            if message.get("paper_analysis") and message["role"] == "assistant":
                message_index = st.session_state.messages.index(message)
                display_paper_analysis_metadata(message, message_index)

            if message.get("chem_ocr") and message["role"] == "assistant":
                display_chem_ocr_metadata(message)

            if message.get("docking") and message["role"] == "assistant":
                display_docking_metadata(message)

    streaming_placeholder = st.empty()

    user_text = st.chat_input("Enter a prompt here...", key="chat_input")
    if user_text:
        st.session_state.messages.append({"role": "user", "content": user_text})
        message_handler(user_text, streaming_placeholder)
        # When finished, force rerun so chat history + input re-render in correct order
        st.rerun()


def message_handler(user_query: str, placeholder: st.delta_generator.DeltaGenerator):
    """
    Processes a user's message through the backend and displays the response.

    This method handles the complete flow of a user interaction: it adds the user's message
    to the chat history, presents it in the interface, sends it to the backend for processing,
    manages potential errors, and displays the backend's reply. This includes intermediate reasoning steps,
    generated visualizations (like molecule structures or images), and handles file uploads.

    Args:
        None

    Returns:
        None
    """
    user_query = st.session_state.chat_input

    try:
        images = st.session_state.images_b64

        config = {
            "recursion_limit": 30,
            "configurable": {
                "img_path": images,
            },
        }

        if st.session_state.uploaded_files:
            save_all_files(st.session_state.user_data_dir)
            config["configurable"]["user_data_dir"] = st.session_state.user_data_dir

        inputs = {"input": user_query}
        # add path to users image from gui
        if st.session_state.images:
            inputs["attached_img"] = st.session_state.images

        with placeholder.container():
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.spinner("Give me a moment..."):
                st.session_state.messages.append(
                    {"role": "assistant", "content": "", "steps": []}
                )
                expander = st.expander(
                        "🔍 Intermediate Thoughts (click to expand)", expanded=False
                    )
                expander_placeholder = expander.empty()

                if "steps" not in st.session_state.messages[-1]:
                        st.session_state.messages[-1]["steps"] = []


                with expander:
                    steps_container = st.container()

                if st.session_state.explore_mode:
                    inputs['input'] = inputs.get('input', '') + " Only use the files provided by the user to answer " \
                                                                "the question. You can only use one of these tools:" \
                                                                "create_dataset_from_papers or " \
                                                                "explore_my_papers tools from paper_analysis_agent. " \
                                                                "use create_dataset_from_papers when the user asks " \
                                                                "to create/collect a dataset," \
                                                                "use explore_my_papers when the user has a question " \
                                                                "about chemistry or about uploaded files/papers"
                try:
                    print('In main graph section')
                    for result in async_to_sync(st.session_state.backend.stream(inputs, "1", user_id="1")):
                        print("=================new step=================")

                        if result.get("plan"):
                            plan = result["plan"]

                            if not isinstance(plan, list):
                                plan = [plan]

                            for step in plan:
                                raw_text = ""
                                for i, task in enumerate(step):
                                    raw_text += f"({i}) " + task + ' '

                                if len(step) > 1:
                                    formatted_text = (
                                        f"📝 {raw_text}" if "Step" in raw_text else f"**📝 Step with parallel launch:** {raw_text}"
                                    )
                                else:
                                    formatted_text = (
                                        f"📝 {raw_text}" if "Step" in raw_text else f"**📝 Step:** {raw_text}"
                                    )

                                existing_steps = set(st.session_state.messages[-1]["steps"])
                                if formatted_text not in existing_steps:
                                    st.session_state.messages[-1]["steps"].append(formatted_text)
                                    existing_steps.add(formatted_text)
                                    steps_container.markdown(formatted_text)

                            with expander_placeholder.container():
                                if st.session_state.messages[-1]['steps']:  # Only render if steps exist
                                    for step in st.session_state.messages[-1]['steps']:
                                        st.markdown(step)

                        if result.get("past_steps") and not result.get("automl_results"):
                            past_steps = result.get('past_steps')
                            first_step = list(past_steps)[-1]
                            text = f"**✅ Result of last step:** {first_step[1]}"

                            if text not in st.session_state.messages[-1]["steps"]:
                                st.session_state.messages[-1]["steps"].append(text)
                                with expander_placeholder.container():
                                    if st.session_state.messages[-1][
                                        "steps"
                                    ]:  # Only render if steps exist
                                        for step in st.session_state.messages[-1]["steps"]:
                                            st.markdown(step)

                        elif result.get("automl_results"):
                            text = f"**✅ Result of last step:** Automl is done"
                            st.session_state.messages[-1]["steps"].append(text)
                            with expander_placeholder.container():
                                if st.session_state.messages[-1][
                                    "steps"
                                ]:  # Only render if steps exist
                                    for step in st.session_state.messages[-1]["steps"]:
                                        st.markdown(step)
                                else:
                                    st.write(" ")  # Ensures blank space instead of None

                            st.session_state.messages[-1]["automl_results"] = result.get(
                                "automl_results"
                            )
                except GraphRecursionError:
                    result["response"] = (
                        "Ooops.. It seems that I've caught a recursion limit. Could you simlify your question and try once more?"
                    )

                except AttributeError as e:
                    print(f"ERROR: {e}")
                    result = dict()
                    result["response"] = (
                        "Something went wrong. Please reload the page, initialize models and try again. If this happens again, check your base url and api key"
                    )

                # st.session_state.messages.append({'role': 'assistant', "content": result['response']})
                st.session_state.messages[-1]["content"] = result["response"]
                if result.get('found_pubmed_papers'):
                    st.session_state.messages[-1]["found_pubmed_papers"] = result.get('found_pubmed_papers')

                # clean_folder(os.path.join(ROOT_DIR, os.environ["DS_STORAGE_PATH"]))
                # clean_folder(os.path.join(ROOT_DIR, os.environ["IMG_STORAGE_PATH"]))
                # clean_folder(os.path.join(ROOT_DIR, os.environ["ANOTHER_STORAGE_PATH"]))

            if st.session_state.images_b64:  # get user's submitted images
                st.session_state.messages[-1][
                    "image_urls"
                ] = st.session_state.images_b64
                st.session_state.images_b64 = None

            path_to_molecules = os.path.join(
                os.environ.get("PATH_TO_RESULTS"), "vis_mols"
            )
            if not os.path.exists(path_to_molecules):
                os.makedirs(path_to_molecules, exist_ok=True)

            if molecules := os.listdir(
                os.path.join(os.getenv("PATH_TO_RESULTS"), "vis_mols")
            ):  # get generated molecules
                mols = []
                for file in molecules:
                    file_path = os.path.join(
                        os.getenv("PATH_TO_RESULTS"), "vis_mols", file
                    )
                    with open(os.path.join(file_path), "r", encoding="utf-8") as f:
                        mol = f.read()
                    mols.append(mol)
                    os.remove(file_path)
                st.session_state.messages[-1]["molecules_vis"] = mols

            path_to_results = os.path.join(os.environ.get("PATH_TO_RESULTS"), "cvae")
            if not os.path.exists(path_to_results):
                os.makedirs(path_to_results, exist_ok=True)

            if files := os.listdir(
                os.path.join(os.getenv("PATH_TO_RESULTS"), "cvae")
            ):  # get generated images
                imgs = []
                # Cleaning here
                for file in files:
                    file_path = os.path.join(os.getenv("PATH_TO_RESULTS"), "cvae", file)
                    imgs.append(convert_to_base64(file_path))
                    os.remove(file_path)
                st.session_state.messages[-1]["images_generated"] = [
                    imgs[0]
                ]  # use only first 5 images

            with st.chat_message("assistant"):
                msg = st.session_state.messages[-1]
                if 'dataset' in msg:
                    display_dataset(msg)
                elif msg.get("automl_results"):
                    st.markdown(msg["content"])
                    st.markdown(msg["automl_results"])
                else:
                    st.markdown(msg["content"])

                if result.get('metadata'):
                    
                    papers = result['metadata']
                    print(papers)

                    paper_key = 'found_papers_' + user_query

                    if papers.get(paper_key):
                        st.subheader("Список найденных статей:")

                    st.session_state.messages[-1]['found_pubmed_papers'] = []

                    for idx, paper in enumerate(papers.get(paper_key, [])):
                        with st.expander(
                            f"{idx+1}. {paper.title}", expanded=False
                        ):

                            authors = ', '.join(paper.authors)
                            st.markdown(f"**Авторы:** {authors}")
                            st.markdown(f"**Год:** {paper.year}")
                            st.markdown(f"**Журнал:** {paper.journal}")
                            st.markdown(f"**Аннотация:** {paper.abstract}")

                            if paper.link:
                                st.markdown(f"[Перейти к статье]({paper.link})")

                            st.session_state.messages[-1]['found_pubmed_papers'].append(paper)

                # ATTENTION: RENDER IMG FOR USER
                if imgs := msg.get("image_urls"):
                    for img in imgs:
                        st.components.v1.html(convert_to_html(img), height=400)
                if "metadata" in result.keys():
                    if "dataset_builder_agent" in result["metadata"].keys():
                        st.markdown("### Dataset Builder Agent Results")
                        for file in result["metadata"]["dataset_builder_agent"]:
                            file_name = os.path.basename(file)
                            st.markdown(f"- {file_name}")

                            # show content from file
                            if file.endswith(".csv"):
                                import pandas as pd

                                df = pd.read_csv(file)
                                st.dataframe(df)
                            elif file.endswith(".xlsx"):
                                import pandas as pd

                                df = pd.read_excel(file)
                                st.dataframe(df)

                            # button for download dataset
                            with open(file, "rb") as f:
                                st.download_button(
                                    label=f"Download {file_name}",
                                    data=f,
                                    file_name=file_name,
                                    key=f"download_{file_name}",
                                )

                            #os.remove(file)

                    # Store metadata in the message for later display
                    if "paper_analysis" in result["metadata"].keys():
                        st.session_state.messages[-1]["paper_analysis"] = result["metadata"]["paper_analysis"]
                        # Display the metadata immediately after storing it
                        message_index = len(st.session_state.messages) - 1
                        display_paper_analysis_metadata(st.session_state.messages[-1], message_index)

                    if "chem_ocr" in result["metadata"].keys():
                        st.session_state.messages[-1]["chem_ocr"] = result["metadata"]["chem_ocr"]
                        # Display the metadata immediately after storing it
                        display_chem_ocr_metadata(st.session_state.messages[-1])
                    
                    if "docking" in result["metadata"].keys():
                        st.session_state.messages[-1]["docking"] = result["metadata"]["docking"]
                        display_docking_metadata(st.session_state.messages[-1])

                if mols := msg.get("molecules_vis"):
                    for mol in mols:
                        st.components.v1.html(mol, height=400)

                if gen_imgs := msg.get("images_generated"):
                    for img in gen_imgs:
                        st.components.v1.html(convert_to_html(img), height=200)

                storage_path = os.path.join(ROOT_DIR, os.environ["DS_STORAGE_PATH"])

                # search all files with 'users_dataset_'
                pattern = os.path.join(storage_path, "users_dataset_*")
                matching_files = glob.glob(pattern)

                # delete all users datasets
                for file_path in matching_files:
                    try:
                        os.remove(file_path)
                    except Exception as e:
                        print(f"Cannot delete {file_path}: {e}")

    except Exception as e:
        logger.exception(
            f"Chat failed with error: {str(e)}\t State: {st.session_state}"
        )


def display_chem_ocr_metadata(message):
    images = message["chem_ocr"]["annotated_images"]
    try:
        cols = st.columns(2)
        for i, img_path in enumerate(images):
            if 'annotated' in img_path:
                img = Image.open(img_path)
                with cols[i % 2]:
                    st.image(img, width=400)
    except Exception as e:
        print(f'Could not display image: {img_path}, ERROR: {e}')


def display_docking_metadata(message):
    html_file = message["docking"]["html_file"]
    if not os.path.isabs(html_file):
        html_file = os.path.join(ROOT_DIR, html_file)
    try:
        with open(html_file, "r", encoding="utf-8") as f:
            html_content = f.read()
    except FileNotFoundError:
        st.warning(f"Файл результата докинга не найден: {html_file}")
        return
    except Exception as e:
        logger.exception("Ошибка при чтении файла докинга: %s", e)
        st.warning("Не удалось загрузить визуализацию докинга.")
        return
    if "<head>" in html_content:
        html_content = html_content.replace("<head>", "<head><base href=\"about:blank\" target=\"_blank\">", 1)
    st.components.v1.html(html_content, height=400)
    st.download_button(
        label="Скачать HTML с результатом докинга",
        data=html_content,
        file_name=os.path.basename(html_file),
        key=f"download_docking",
    )
    
@st.dialog("Extracted compounds", width="large")
def pdf_viewer(folder: str):
    folder_path = Path(folder)

    exts = ("*.png", "*.jpg", "*.jpeg", "*.webp")
    image_paths = []
    for ext in exts:
        image_paths.extend(folder_path.glob(ext))

    image_paths = sorted(image_paths)

    # st.write(f"Folder: {folder_path} ({len(image_paths)} pages)")
    with st.container(height=500, border=True):
        for i, p in enumerate(image_paths):
            img = Image.open(p)
            st.image(img, width=800)

            if i < len(image_paths) - 1:
                st.divider()


def display_paper_analysis_metadata(message, message_index):
    """
    Display analysis details extracted from scientific papers, allowing users to selectively view text, images, and metadata.

    Args:
        message (dict): A dictionary containing the paper analysis data.  It is expected to have a "paper_analysis" key.
        message_index (int): A unique index for the message, used to create unique keys for the checkboxes to maintain state.

    Returns:
        None: This function displays content using Streamlit and does not return a value.  It updates the Streamlit session state with the checkbox values.
    """
    if "paper_analysis" not in message:
        return

    paper_analysis = message["paper_analysis"]

    if "dataset" in paper_analysis.keys():
        display_dataset(paper_analysis.get("dataset"))

    if "images_path" in paper_analysis.keys():
        if st.button("Check extracted compounds"):
            pdf_viewer(paper_analysis.get("images_path"))

    if "text_context" in paper_analysis.keys():
        text_context = paper_analysis.get("text_context")
        images_context = paper_analysis.get("image_context")

        # Create unique keys for this message's checkboxes
        text_key = f"text_context_{message_index}"
        images_key = f"image_context_{message_index}"
        meta_key = f"metadata_{message_index}"

        # Initialize checkbox states in session_state if not present
        if text_key not in st.session_state:
            st.session_state[text_key] = False
        if images_key not in st.session_state:
            st.session_state[images_key] = False
        if meta_key not in st.session_state:
            st.session_state[meta_key] = False

        # Create toggles for context display
        col1, col2, col3 = st.columns(3)

        with col1:
            show_text = st.checkbox(
                "📄 Text Context",
                value=st.session_state[text_key],
                key=text_key
            )

        with col2:
            show_images = st.checkbox(
                "🖼️ Image Context",
                value=st.session_state[images_key],
                key=images_key
            )

        # Display text context if selected
        if show_text:
            with st.expander("📄 Text Context", expanded=True):
                for text in text_context:
                    st.text_area(f'Text Context:', value=text['chunk'], height=350, disabled=True)
                    description_text = f'{text["Paper"]}, {text["Year"]}'
                    st.markdown(f"""<span style="color:black;"><b>Source: </b> {description_text}</span>""",
                                unsafe_allow_html=True)

        # Display image context if selected
        if show_images:
            with st.expander("🖼️ Image Context", expanded=True):
                if images_context:
                    for i, image_item in enumerate(images_context):
                        bucket_name, s3_key = urlparse(image_item['path']).path.split('/', 2)[1:]

                        # Display image if selected
                        try:
                            pil_image = Image.open(BytesIO(s3_service.get_image_bytes_from_s3(s3_key, bucket_name)))
                            st.image(pil_image, caption='', width=800)
                            description_text = f'{image_item["Paper"]}, {image_item["Year"]}'
                            st.markdown(f"""<span style="color:black;"><b>Source: </b> {description_text}</span>""",
                                        unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Could not display image: {image_item}. Error: {str(e)}")
                else:
                    st.write("No image context available")


def display_dataset(dataset: str):
    import pandas as pd
    df = pd.read_csv(dataset, sep="\t")
    st.dataframe(df)


def async_to_sync(async_gen):
    queue = Queue()

    async def produce():
        try:
            async for item in async_gen:
                queue.put(item)
        finally:
            queue.put(None)  # signal completion

    # Background thread to run async loop
    def runner():
        asyncio.run(produce())

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()

    # Yield results synchronously as they arrive
    while True:
        try:
            item = queue.get(timeout=0.1)
        except Empty:
            continue

        if item is None:
            break
        yield item
