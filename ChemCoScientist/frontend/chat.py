import glob
import logging
import os

import streamlit as st
from frontend.utils import get_user_data_dir, get_user_session_id, save_all_files
from langgraph.errors import GraphRecursionError
from tools.utils import convert_to_base64, convert_to_html

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
        if st.session_state.get("user_session_id") is None:
            st.session_state.user_session_id = get_user_session_id()

        if st.session_state.get("user_data_dir") is None:
            st.session_state.user_data_dir = get_user_data_dir(
                st.session_state.user_session_id
            )

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
            expander_placeholder = expander.empty()  # Container for intermediate output

            # result = st.session_state.backend.invoke(input=inputs, config=config)
            try:
                # test = [{'plan':['use automl']}, {'past_steps': [['', 'automl here']]}, {'automl_results': '🚀 **ML Model Report** 🚀\n\n**Task:** Regression on `train.csv` with feature `x` and label `y`.\n\n### Model Characteristics:\n- **Metrics:** RMSE (Root Mean Square Error)\n- **Pipeline Structure:**\n  - Depth: 2\n  - Length: 2\n  - Nodes:\n    - **Random Forest Regressor (RFR):** \n      - `n_jobs`: 1\n    - **Scaling:** \n      - No parameters\n\n### Performance Metrics:\n| Metric        | Value       |\n|---------------|-------------|\n| RMSE          | [Insert Value] |\n\n### Code Overview:\n- **Data Loading:** Utilizes `pandas` to read `train.csv` and `val.csv`.\n- **Model Training:** \n  - AutoML framework (`Fedot`) for regression.\n  - 5-fold cross-validation with a timeout of 1.0 seconds.\n- **Evaluation:** \n  - Metrics obtained from validation set.\n\n### Output:\n- Predictions saved to `submission.csv`.\n\n📊 **Model Performance on Test Set:** [Insert Performance Metrics] \n\n#ML #MachineLearning #Regression #DataScience'}, {'response': 'done'}]
                for result in st.session_state.backend.stream(inputs, "1"):
                    print("=================new step=================")
                    print(result)
                    # for result in test:
                    if result.get("plan"):
                        text = result.get("plan")[0]
                        if "Step" not in text:
                            text = f"**📝 Step:** {text}"
                        else:
                            text = f"📝 {text}"

                        st.session_state.messages[-1]["steps"].append((text))

                        with expander_placeholder.container():
                            if st.session_state.messages[-1][
                                "steps"
                            ]:  # Only render if steps exist
                                for step in st.session_state.messages[-1]["steps"]:
                                    st.markdown(step)
                            else:
                                st.write(" ")  # Ensures blank space instead of None

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

            except AttributeError:
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

                if mols := msg.get("molecules_vis"):
                    for mol in mols:
                        st.components.v1.html(mol, height=400)

                if gen_imgs := msg.get("images_generated"):
                    for img in gen_imgs:
                        st.components.v1.html(convert_to_html(img), height=200)

                storage_path = os.environ.get("DS_STORAGE_PATH")

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
