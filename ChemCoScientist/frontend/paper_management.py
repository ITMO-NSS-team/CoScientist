import streamlit as st
from ChemCoScientist.frontend.streamlit_endpoints import delete_temp_papers, SELECTED_PAPERS, select_file, deselect_file


logger = st.logger.get_logger(__name__)


def paper_management():
    """
    Manages the display, selection, and deletion of uploaded scientific papers.
    
    This method provides a user interface for managing uploaded papers, allowing users to select papers for analysis and remove unwanted files. It leverages Streamlit's session state to maintain UI state and track user interactions.
    
    Args:
        None
    
    Returns:
        None
    """
    st.header("📁 File Management")

    # Initialize session state for showing papers
    if "show_papers" not in st.session_state:
        st.session_state.show_papers = False

    # File type selection - small button on the left
    col1, col2, col3 = st.columns([2, 6, 1])

    with col1:
        if st.button("📄 My Papers", key="my_papers_btn"):
            st.session_state.show_papers = not st.session_state.show_papers
            # st.rerun()

    st.divider()

    # Display files when papers are selected
    if st.session_state.show_papers:
        # Filter files by type
        logger.info(f'uploaded papers: {st.session_state.uploaded_papers}')
        scientific_papers = [f for f in st.session_state.uploaded_papers if
                             f.get("type") in ["application/pdf", "text/plain",
                                               "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]]

        if scientific_papers:
            # Sync backend state with existing files on page load
            # sync_selected_papers_with_existing_files()
            session_id = st.session_state.session_id
            selected_papers = SELECTED_PAPERS.get(session_id, [])

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
                        st.session_state[f"prev_process_paper_{i}"] = master_analysis  # Update previous state tracking
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
                        logger.info(f'DELETE PAPERS: {papers_to_delete}')
                        delete_temp_papers(papers_to_delete)

                        # Clear all checkbox states since indices will change after deletion
                        keys_to_clear = [key for key in st.session_state.keys()
                                         if key.startswith(f"delete_paper_") or
                                         key.startswith(f"process_paper_") or
                                         key.startswith(f"prev_process_paper_")]

                        logger.info(f'KEYS TO CLEAR: {keys_to_clear}')

                        # Reset master checkbox states
                        if "prev_master_analysis" in st.session_state:
                            del st.session_state["prev_master_analysis"]

                        if "prev_master_delete" in st.session_state:
                            del st.session_state["prev_master_delete"]

                        for key in keys_to_clear:
                            if key in st.session_state:
                                del st.session_state[key]

                        # Remove from session state
                        for paper in papers_to_delete:
                            temp_list = [
                                f for f in st.session_state.uploaded_papers
                                if f["name"] != paper["name"]
                            ]
                            st.session_state.uploaded_papers = temp_list
                            logger.info(f'UPLOADED PAPERS: {st.session_state.uploaded_papers}')
                            # Remove from selected papers backend
                            # deselect_file(paper["name"])

                        st.rerun()
                        # logger.info(f'UPLOADED FILES AFTER RERUN: {st.session_state.uploaded_files}')
                    else:
                        st.warning("⚠️ Please select at least one paper to delete.")

            # st.divider()

            # Display papers with checkboxes
            logger.info(f'scientific_papers: {scientific_papers}')
            for i, paper in enumerate(scientific_papers):
                col1, col2, col3 = st.columns([1, 6, 2])

                with col1:
                    st.checkbox(
                        "Delete",
                        key=f"delete_paper_{i}",
                        help="Select to delete this paper",
                        label_visibility="hidden",
                        value=st.session_state.get(f"delete_paper_{i}", False),
                    )

                with col2:
                    st.write(f"**{paper['name']}**")

                with col3:
                    # Store previous state in a separate key to track changes
                    prev_state_key = f"prev_process_paper_{i}"
                    previous_state = st.session_state.get(prev_state_key, False)

                    is_selected = st.checkbox(
                        "Select for analysis",
                        key=f"process_paper_{i}",
                        help="Select to process this paper for analysis",
                        value=st.session_state.get(f"process_paper_{i}", False),
                    )

                    logger.info(f'is file selected: {is_selected}')
                    # Call backend functions when checkbox state changes
                    if is_selected != previous_state:
                        file_path = paper["name"]  # Using filename as file_path
                        if is_selected:
                            logger.info('select_file called')
                            select_file(file_path)
                        else:
                            deselect_file(file_path)
                            logger.info('deselect_file called')

                        # Update the previous state
                        st.session_state[prev_state_key] = is_selected
        else:
            st.info(
                "📄 No scientific papers uploaded yet. Upload some PDF files to get started!")
