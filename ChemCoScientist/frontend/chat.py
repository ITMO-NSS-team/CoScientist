import glob
import logging
import os
import time

import streamlit as st
from ChemCoScientist.frontend.utils import get_user_data_dir, get_user_session_id, save_all_files
from langgraph.errors import GraphRecursionError
from ChemCoScientist.tools.utils import convert_to_base64, convert_to_html
from ChemCoScientist.frontend.streamlit_endpoints import explore_my_papers
from definitions import ROOT_DIR

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


def chat():
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

            gen_imgs = message.get("images_generated")

            if imgs := message.get("image_urls"):  # render previously submitted images
                for img in imgs:
                    st.components.v1.html(convert_to_html(img), height=400)

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

    on_submit = st.chat_input(
        "Enter a prompt here...", key="chat_input", disabled=False
    )

    if on_submit:
        # chat_logger.info(f'Submitted message: {st.session_state.chat_input}')
        message_handler()


def message_handler():
    user_query = st.session_state.chat_input
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.chat_message("user"):
        st.markdown(user_query)

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

    try:
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

            existing_steps = set(st.session_state.messages[-1]["steps"])

            with expander:
                steps_container = st.container()

            if st.session_state.explore_mode:
                # Use explore_my_papers function instead of general AI assistant
                result = explore_my_papers(inputs)

                st.markdown(result["answer"])

                st.session_state.messages[-1]["content"] = (result["answer"])
            else:
                # result = st.session_state.backend.invoke(input=inputs, config=config)
                try:
                    for result in st.session_state.backend.stream(inputs, "1"):
                        print("=================new step=================")
                        print(result)

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
                                        f"📝 {raw_text}" if "Step" in raw_text else f"**�� Step with parallel launch:** {raw_text}"
                                    )
                                else:
                                    formatted_text = (
                                        f"📝 {raw_text}" if "Step" in raw_text else f"**📝 Step:** {raw_text}"
                                    )

                                if formatted_text not in existing_steps:
                                    st.session_state.messages[-1]["steps"].append(formatted_text)
                                    existing_steps.add(formatted_text)

                                    # Показываем только новые шаги
                                    steps_container.markdown(formatted_text)

                        elif result.get("past_steps") and not result.get("automl_results"):
                            text = f"**✅ Result of last step:** {result.get('past_steps')[0][1]}"
                            st.session_state.messages[-1]["steps"].append(text)

                            with expander_placeholder.container():
                                if st.session_state.messages[-1][
                                    "steps"
                                ]:  # Only render if steps exist
                                    for step in st.session_state.messages[-1]["steps"]:
                                        st.markdown(step)
                                else:
                                    st.write(" ")  # Ensures blank space instead of None

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
                if msg.get("automl_results"):
                    st.markdown(msg["content"])
                    st.markdown(msg["automl_results"])
                else:
                    st.markdown(msg["content"])

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

                            os.remove(file)

                    # Store metadata in the message for later display
                    if "paper_analysis" in result["metadata"].keys():
                        st.session_state.messages[-1]["paper_analysis"] = result["metadata"]["paper_analysis"]
                        # Display the metadata immediately after storing it
                        message_index = len(st.session_state.messages) - 1
                        display_paper_analysis_metadata(st.session_state.messages[-1], message_index)

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


def display_paper_analysis_metadata(message, message_index):
    """Display paper analysis metadata with persistent checkboxes"""
    if "paper_analysis" not in message:
        return

    paper_analysis = message["paper_analysis"]
    text_context = paper_analysis.get("text_context")
    images_context = paper_analysis.get("image_context")
    metadata = paper_analysis.get("metadata")

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

    with col3:
        show_meta = st.checkbox(
            "ℹ️ Metadata",
            value=st.session_state[meta_key],
            key=meta_key
        )

    # Display text context if selected
    if show_text:
        with st.expander("📄 Text Context", expanded=True):
            st.text_area("Text Context:", value=text_context, height=200, disabled=True)

    # Display image context if selected
    if show_images:
        with st.expander("🖼️ Image Context", expanded=True):
            if images_context:
                for i, image_item in enumerate(images_context):
                    img_key = f"img_checkbox_{message_index}_{i}"

                    # Initialize image checkbox state if not present
                    if img_key not in st.session_state:
                        st.session_state[img_key] = False

                    show_img = st.checkbox(
                        f"{i + 1}. {image_item}",
                        value=st.session_state[img_key],
                        key=img_key
                    )

                    # Display image if selected
                    if show_img:
                        try:
                            st.image(image_item, caption=image_item, use_container_width=True)
                        except Exception as e:
                            st.error(f"Could not display image: {image_item}. Error: {str(e)}")
            else:
                st.write("No image context available")

    # Display metadata if selected
    if show_meta:
        with st.expander("ℹ️ Metadata", expanded=True):
            if metadata:
                for key, value in metadata.items():
                    st.write(f"**{key}:** {value}")
            else:
                st.write("No metadata available")