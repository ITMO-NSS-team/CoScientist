import streamlit as st
import time
import uuid
from utils.chat_functions import get_answer_from_assistant
from utils.file_processor import process_uploaded_file

# Global dictionary to store selected papers by session
SELECTED_PAPERS = {}

# Page configuration
st.set_page_config(
    page_title="ChatGPT-like Interface",
    page_icon="💬",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Initialize session ID for tracking selected papers
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Initialize selected papers list for this session
if st.session_state.session_id not in SELECTED_PAPERS:
    SELECTED_PAPERS[st.session_state.session_id] = []

# Main title
st.title("💬 ChatGPT-like Interface")

# Create tabs
tab_chat, tab_files = st.tabs(["💬 Chat", "📁 File Management"])

with tab_chat:
    # Sidebar for file uploads and chat management
    with st.sidebar:
        st.header("📁 File Upload")

        # File uploader - allow multiple files
        uploaded_files = st.file_uploader(
            "Upload documents to add to knowledge base",
            type=['txt', 'pdf', 'docx', 'md', 'csv'],
            help="Supported formats: TXT, PDF, DOCX, MD, CSV",
            accept_multiple_files=True
        )

        if uploaded_files:
            new_files_processed = False

            with st.spinner("Processing uploaded files..."):
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in [f["name"] for f in st.session_state.uploaded_files]:
                        try:
                            # Process the uploaded file
                            result = process_uploaded_file(uploaded_file)

                            if result["success"]:
                                st.session_state.uploaded_files.append({
                                    "name": uploaded_file.name,
                                    "size": uploaded_file.size,
                                    "type": uploaded_file.type
                                })
                                st.success(f"✅ Successfully processed: {uploaded_file.name}")
                                new_files_processed = True
                            else:
                                st.error(f"❌ Error processing file: {result['error']}")
                        except Exception as e:
                            st.error(f"❌ Unexpected error processing {uploaded_file.name}: {str(e)}")

            # Rerun only if new files were processed
            if new_files_processed:
                time.sleep(1)
                st.rerun()

        # Display uploaded files
        if st.session_state.uploaded_files:
            st.subheader("📚 Uploaded Files")
            for file_info in st.session_state.uploaded_files:
                with st.expander(f"📄 {file_info['name']}", expanded=False):
                    st.write(f"**Size:** {file_info['size']:,} bytes")
                    st.write(f"**Type:** {file_info['type']}")

        st.divider()

        # Chat management
        st.header("🔧 Chat Management")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.success("Chat history cleared!")
            time.sleep(1)
            st.rerun()

        # Display chat statistics
        if st.session_state.messages:
            st.metric("Messages", len(st.session_state.messages))
            user_messages = len([msg for msg in st.session_state.messages if msg["role"] == "user"])
            ai_messages = len([msg for msg in st.session_state.messages if msg["role"] == "assistant"])
            st.metric("User Messages", user_messages)
            st.metric("AI Responses", ai_messages)

    # Main chat interface
    st.header("💭 Chat")

    # Display chat messages
    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            st.info("👋 Welcome! Start a conversation by typing a message below.")
        else:
            for i, message in enumerate(st.session_state.messages):
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if "timestamp" in message:
                        st.caption(f"*{message['timestamp']}*")

                    # Show context buttons for assistant messages that have additional data
                    if (message["role"] == "assistant" and
                            any(key in message for key in ["text_context", "images_context", "metadata"])):

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            show_hist_text = st.checkbox("📄 Text Context", key=f"hist_text_{i}")

                        with col2:
                            show_hist_images = st.checkbox("🖼️ Image Context", key=f"hist_image_{i}")

                        with col3:
                            show_hist_meta = st.checkbox("ℹ️ Metadata", key=f"hist_meta_{i}")

                        # Display contexts if selected
                        if show_hist_text:
                            with st.expander("📄 Text Context", expanded=True):
                                st.text_area("Text Context:", value=message.get("text_context", ""), height=200,
                                             disabled=True, key=f"text_area_{i}")

                        if show_hist_images:
                            with st.expander("🖼️ Image Context", expanded=True):
                                images_context = message.get("images_context", [])
                                if images_context:
                                    for j, image_item in enumerate(images_context):
                                        show_hist_img = st.checkbox(f"{j + 1}. {image_item}",
                                                                    key=f"hist_img_checkbox_{i}_{j}")

                                        # Display image if selected
                                        if show_hist_img:
                                            try:
                                                st.image(image_item, caption=image_item, use_container_width=True)
                                            except Exception as e:
                                                st.error(f"Could not display image: {image_item}. Error: {str(e)}")
                                else:
                                    st.write("No image context available")

                        if show_hist_meta:
                            with st.expander("ℹ️ Metadata", expanded=True):
                                metadata = message.get("metadata", {})
                                if metadata:
                                    for key, value in metadata.items():
                                        st.write(f"**{key}:** {value}")
                                else:
                                    st.write("No metadata available")

    # Chat input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        # Add user message to chat history
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        user_message = {
            "role": "user",
            "content": user_input,
            "timestamp": timestamp
        }
        st.session_state.messages.append(user_message)

        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)
            st.caption(f"*{timestamp}*")

        # Generate AI response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Call the AI response function
                    text_context, images_context, answer, metadata = get_answer_from_assistant(user_input)

                    # Display the answer
                    st.markdown(answer)

                    # Create toggles for context display
                    col1, col2, col3 = st.columns(3)

                    msg_idx = len(st.session_state.messages)

                    with col1:
                        show_text = st.checkbox("📄 Text Context", key=f"text_context_{msg_idx}")

                    with col2:
                        show_images = st.checkbox("🖼️ Image Context", key=f"image_context_{msg_idx}")

                    with col3:
                        show_meta = st.checkbox("ℹ️ Metadata", key=f"metadata_{msg_idx}")

                    # Display text context if selected
                    if show_text:
                        with st.expander("📄 Text Context", expanded=True):
                            st.text_area("Text Context:", value=text_context, height=200, disabled=True)

                    # Display image context if selected
                    if show_images:
                        with st.expander("🖼️ Image Context", expanded=True):
                            if images_context:
                                for i, image_item in enumerate(images_context):
                                    show_img = st.checkbox(f"{i + 1}. {image_item}", key=f"img_checkbox_{msg_idx}_{i}")

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

                    # Add AI response to chat history
                    ai_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    ai_message = {
                        "role": "assistant",
                        "content": answer,
                        "text_context": text_context,
                        "images_context": images_context,
                        "metadata": metadata,
                        "timestamp": ai_timestamp
                    }
                    st.session_state.messages.append(ai_message)
                    st.caption(f"*{ai_timestamp}*")

                except Exception as e:
                    error_message = f"❌ Sorry, I encountered an error: {str(e)}"
                    st.error(error_message)

                    # Add error message to chat history
                    error_timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                    error_msg = {
                        "role": "assistant",
                        "content": error_message,
                        "timestamp": error_timestamp
                    }
                    st.session_state.messages.append(error_msg)

    # Footer information
    st.divider()
    st.caption("💡 **Tip:** Upload documents to enhance the AI's knowledge base and get more accurate responses!")

    # Display system status
    with st.expander("ℹ️ System Information", expanded=False):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Chat Messages", len(st.session_state.messages))

        with col2:
            st.metric("Uploaded Files", len(st.session_state.uploaded_files))

        with col3:
            total_file_size = sum(file["size"] for file in st.session_state.uploaded_files)
            st.metric("Total File Size", f"{total_file_size:,} bytes")

# File Management Tab
with tab_files:
    st.header("📁 File Management")

    # Initialize session state for showing papers
    if "show_papers" not in st.session_state:
        st.session_state.show_papers = False

    # File type selection - small button on the left
    col1, col2, col3 = st.columns([2, 6, 1])

    with col1:
        if st.button("📄 My Papers", key="my_papers_btn"):
            st.session_state.show_papers = not st.session_state.show_papers
            st.rerun()

    st.divider()

    # Display files when papers are selected
    if st.session_state.show_papers:
        # Filter files by type
        scientific_papers = [f for f in st.session_state.uploaded_files if
                             f.get("type") in ["application/pdf", "text/plain",
                                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]]

        if scientific_papers:
            # Sync backend state with frontend checkboxes on page load
            session_id = st.session_state.session_id
            selected_papers = SELECTED_PAPERS.get(session_id, [])

            # Debug: Show currently selected papers
            if selected_papers:
                st.success(f"📊 Currently selected for analysis: {', '.join(selected_papers)}")

            # Top row with master checkboxes and delete button
            col1, col2, col3, col4 = st.columns([1, 6, 2, 1])

            with col1:
                # Master checkbox for deletion
                master_delete = st.checkbox(
                    "Delete All",
                    key="master_delete",
                    help="Select/deselect all papers for deletion",
                    label_visibility="hidden"
                )

                # Update individual checkboxes based on master checkbox
                if master_delete != st.session_state.get("prev_master_delete", False):
                    for i in range(len(scientific_papers)):
                        st.session_state[f"delete_paper_{i}"] = master_delete
                    st.session_state["prev_master_delete"] = master_delete
                    st.rerun()

            with col2:
                st.write("**Paper Name**")

            with col3:
                # Master checkbox for analysis
                master_analysis = st.checkbox(
                    "Select All for Analysis",
                    key="master_analysis",
                    help="Select/deselect all papers for analysis",
                    label_visibility="hidden"
                )

                # Update individual checkboxes based on master checkbox
                if master_analysis != st.session_state.get("prev_master_analysis", False):
                    for i, paper in enumerate(scientific_papers):
                        st.session_state[f"process_paper_{i}"] = master_analysis
                        file_path = paper["name"]  # Using filename as file_path

                        # Call backend functions for each paper
                        if master_analysis:
                            select_file(file_path)
                        else:
                            deselect_file(file_path)

                    st.session_state["prev_master_analysis"] = master_analysis
                    st.rerun()

            with col4:
                # Small red delete button in upper right
                st.markdown(
                    """
                    <style>
                    .stButton > button[data-testid="baseButton-secondary"] {
                        background-color: #ff4444;
                        color: white;
                        border: none;
                        border-radius: 4px;
                        padding: 0.25rem 0.5rem;
                        font-size: 0.8rem;
                        height: 2rem;
                    }
                    .stButton > button[data-testid="baseButton-secondary"]:hover {
                        background-color: #cc0000;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                if st.button("🗑️", help="Delete Selected Papers", type="secondary"):
                    papers_to_delete = []
                    for i, paper in enumerate(scientific_papers):
                        if st.session_state.get(f"delete_paper_{i}", False):
                            papers_to_delete.append(paper)

                    if papers_to_delete:
                        # Placeholder for backend deletion function
                        # In your implementation, replace this with actual backend call
                        delete_papers_from_database(papers_to_delete)

                        # Remove from session state
                        for paper in papers_to_delete:
                            st.session_state.uploaded_files = [
                                f for f in st.session_state.uploaded_files
                                if f["name"] != paper["name"]
                            ]

                        st.success(f"✅ Successfully deleted {len(papers_to_delete)} papers!")
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.warning("⚠️ Please select at least one paper to delete.")

            st.divider()

            # Display papers with checkboxes
            for i, paper in enumerate(scientific_papers):
                col1, col2, col3 = st.columns([1, 6, 2])

                with col1:
                    st.checkbox(
                        "Delete",
                        key=f"delete_paper_{i}",
                        help="Select to delete this paper",
                        label_visibility="hidden"
                    )

                with col2:
                    st.write(f"**{paper['name']}**")

                with col3:
                    # Check current state and handle changes
                    current_state = st.session_state.get(f"process_paper_{i}", False)
                    is_selected = st.checkbox(
                        "Select for analysis",
                        key=f"process_paper_{i}",
                        help="Select to process this paper for analysis"
                    )

                    # Call backend functions when checkbox state changes
                    if is_selected != current_state:
                        file_path = paper["name"]  # Using filename as file_path
                        if is_selected:
                            select_file(file_path)
                        else:
                            deselect_file(file_path)

        else:
            st.info(
                "📄 No scientific papers uploaded yet. Upload some PDF, TXT, or DOCX files in the Chat tab to get started!")


# Backend functions for managing selected papers
def select_file(file_path: str) -> None:
    """Add a paper to the selected papers list for processing"""
    session_id = st.session_state.session_id
    if session_id not in SELECTED_PAPERS:
        SELECTED_PAPERS[session_id] = []
    if file_path not in SELECTED_PAPERS[session_id]:
        SELECTED_PAPERS[session_id].append(file_path)


def deselect_file(file_path: str) -> None:
    """Remove a paper from the selected papers list"""
    session_id = st.session_state.session_id
    if session_id in SELECTED_PAPERS and file_path in SELECTED_PAPERS[session_id]:
        SELECTED_PAPERS[session_id].remove(file_path)


# Placeholder backend functions (to be replaced with actual implementation)
def delete_papers_from_database(papers):
    """
    Placeholder function for deleting papers from database.
    Replace this with your actual backend implementation.

    Args:
        papers: List of paper dictionaries to delete
    """
    # Your backend deletion logic here
    pass
