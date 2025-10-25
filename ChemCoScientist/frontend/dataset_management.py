import streamlit as st
import os
import pandas as pd
import plotly.express as px
import uuid


def dataset_management(db, user_id='user'):
    """
    Manages the display and downloading of uploaded files for a specific user.
    
    This method provides a user interface for managing uploaded files, allowing users to 
    preview files and download them. It leverages Streamlit's session state to maintain 
    UI state and track user interactions.
    
    Args:
        user_id (str): The user ID to filter files by. Defaults to 'user'.
    
    Returns:
        None
    """
    st.markdown("""
    <style>
        .stButton button {
            margin-top: 20px;
            float: right;
        }
    </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 1, 1])
    with col1:
        st.header("📁 File Management")
    with col3:  # Button in the rightmost column
        if st.button("🔄 Refresh", key="refresh_files"):
            st.rerun()
    
    st.divider()

    # Display files when button is clicked
    if True:
        # Get all files for the specific user from the database
        try:
            user_files = db.get_files_by_user(user_id)
            #logger.info(f'Found {len(user_files)} files for user {user_id}')
            
            if not user_files:
                st.info("📄 No files uploaded yet. Upload some files to get started!")
                return

            # Display files in an organized manner
            display_user_files(user_files, user_id)

        except Exception as e:
            #logger.error(f"Error retrieving files for user {user_id}: {e}")
            st.error(f"❌ Error loading files: {str(e)}")

def display_user_files(user_files, user_id):
    """
    Display user files with download and preview options.
    
    Args:
        user_files (list): List of file dictionaries from the database
        user_id (str): The user ID for session tracking
    """
    # Header row
    col1, col2 = st.columns([7, 2])
    
    # with col1:
    #     st.write("**File Name & Details**")
    
    # with col2:
    #     st.write("**Download**")

    # Display files with preview and download options
    #logger.info(f'Displaying {len(user_files)} files for user {user_id}')
    
    for i, file_data in enumerate(user_files):
        display_file_row(file_data, i, user_id)

def display_file_row(file_data, index, user_id):
    """
    Display a single file row with metadata, preview, and download options.
    
    Args:
        file_data (dict): File data from database
        index (int): Index of the file in the list
        user_id (str): User ID for session tracking
    """
    col1, col2 = st.columns([7, 2])

    with col1:
        # File name as clickable for preview
        file_key = f"file_preview_{index}"

        file_type = file_data.get('file_type', 'Unknown').upper()
        file_size = format_file_size(file_data.get('file_size', 0))
        upload_date = format_date(file_data.get('upload_date', ''))
        
        metadata_text = f"📄 {file_type} • 📏 {file_size} • 📅 {upload_date}"
    
        # Create an expander for file preview
        with st.expander(f"**{file_data['filename']}**", expanded=False):
            display_file_preview(file_data)
            
            # File metadata
            display_file_metadata(file_data)
        st.caption(metadata_text)
        
    with col2:
        # Download button
        display_download_button(file_data)

def display_file_preview(file_data):
    """
    Display file preview based on file type.
    
    Args:
        file_data (dict): File data from database
    """
    file_type = file_data.get('file_type', '').lower()
    file_path = file_data.get('file_path', '')
    
    try:
        if file_type == 'csv':
            display_csv_preview(file_data, file_path)
        elif file_type == 'pdf':
            display_pdf_preview(file_data)
        elif file_type in ['txt', 'log']:
            display_text_preview(file_data, file_path)
        else:
            st.info(f"📄 Preview not available for {file_type.upper()} files")
            
    except Exception as e:
        #logger.error(f"Error previewing file {file_data['filename']}: {e}")
        st.error(f"❌ Error loading preview: {str(e)}")


def display_csv_preview(file_data, file_path):
    """
    Display CSV file preview using pandas.
    
    Args:
        file_data (dict): File data from database
        file_path (str): Path to the CSV file
    """
    try:
        # Try to read the CSV file
        df = pd.read_csv(file_path)
        
        st.write("**Data Preview:**")
        
        # Show basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Size", format_file_size(file_data.get('file_size', 0)))
        
        # Show dataframe head
        st.write("**First 5 rows:**")
        st.dataframe(df.head(), use_container_width=True)
        
        # Show data types
        st.write("**Data Types:**")
        dtype_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count().values
        })
        st.dataframe(dtype_info, use_container_width=True, hide_index=True)
        
        # Show basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            st.write("**Numeric Columns Statistics:**")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)

        if len(numeric_cols) > 0:
            st.write("**📈 Distribution Plots:**")
            
            # Let user select column for histogram
            selected_col = st.selectbox("Select column for distribution:", numeric_cols)
            
            col1, col2 = st.columns(2)
            with col1:
                # Histogram
                fig_hist = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig_hist, use_container_width=True, key=f"hist_{selected_col}_{uuid.uuid4().hex[:8]}")
            
            with col2:
                # Box plot
                fig_box = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig_box, use_container_width=True, key=f"box_{selected_col}_{uuid.uuid4().hex[:8]}")

        if len(numeric_cols) > 1:
            st.write("**🔥 Correlation Matrix:**")
            corr_matrix = df[numeric_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Correlation Heatmap",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True, key=f"heatmap_{uuid.uuid4().hex[:8]}")

    except Exception as e:
        st.error(f"❌ Could not read CSV file: {str(e)}")
        # Fallback to metadata if available
        if 'metadata' in file_data and 'csv_metadata' in file_data['metadata']:
            display_csv_metadata_fallback(file_data['metadata']['csv_metadata'])

def display_csv_metadata_fallback(csv_metadata):
    """
    Display CSV metadata when file cannot be read directly.
    
    Args:
        csv_metadata (dict): CSV metadata from database
    """
    st.write("**File Information:**")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"Rows: {csv_metadata.get('row_count', 'N/A')}")
        st.write(f"Columns: {csv_metadata.get('column_count', 'N/A')}")
    with col2:
        st.write(f"Headers: {csv_metadata.get('has_headers', 'N/A')}")
        st.write(f"Delimiter: {csv_metadata.get('delimiter', 'N/A')}")
    
    if 'headers' in csv_metadata:
        st.write("**Columns:**")
        st.write(", ".join(csv_metadata['headers']))
    
    if 'sample_data' in csv_metadata and csv_metadata['sample_data']:
        st.write("**Sample Data:**")
        for i, row in enumerate(csv_metadata['sample_data'][:3]):  # Show first 3 sample rows
            st.write(f"Row {i+1}: {row}")

def display_pdf_preview(file_data):
    """
    Display PDF file information.
    
    Args:
        file_data (dict): File data from database
    """
    st.info("📊 PDF preview requires specialized libraries. You can download the file to view it.")
    
    # Show basic PDF info
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**File Size:** {format_file_size(file_data.get('file_size', 0))}")
    with col2:
        st.write(f"**Uploaded:** {format_date(file_data.get('upload_date', ''))}")

def display_text_preview(file_data, file_path):
    """
    Display text file preview.
    
    Args:
        file_data (dict): File data from database
        file_path (str): Path to the text file
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        st.write("**File Content Preview:**")
        
        # Show first 500 characters
        preview_content = content[:500] + "..." if len(content) > 500 else content
        st.text_area("Content", preview_content, height=150, key=f"text_preview_{file_data['filename']}")
        
        # Show file stats
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Size:** {format_file_size(file_data.get('file_size', 0))}")
            st.write(f"**Characters:** {len(content)}")
        with col2:
            st.write(f"**Lines:** {len(content.splitlines())}")
            st.write(f"**Encoding:** UTF-8")
            
    except Exception as e:
        st.error(f"❌ Could not read text file: {str(e)}")

def display_file_metadata(file_data):
    """
    Display file metadata information.
    
    Args:
        file_data (dict): File data from database
    """
    st.divider()
    st.write("**File Information**")
    
    # Basic metadata
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Type:** {file_data.get('file_type', 'Unknown').upper()}")
        st.write(f"**Size:** {format_file_size(file_data.get('file_size', 0))}")
        st.write(f"**Uploaded:** {format_date(file_data.get('upload_date', ''))}")
    
    with col2:
        st.write(f"**Original Name:** {file_data.get('original_filename', 'N/A')}")
        st.write(f"**Location:** {file_data.get('file_path', 'N/A')}")
    
    # User context if available
    user_context = file_data.get('user_context')
    if user_context:
        st.write(f"**Upload Context:** {user_context}")
    
    # Tags if available
    tags = file_data.get('tags', [])
    if tags:
        tag_names = [tag['name'] for tag in tags]
        st.write(f"**Tags:** {', '.join(tag_names)}")
    
    # Access history if available
    access_logs = file_data.get('access_logs', [])
    if access_logs:
        last_access = max([log.get('access_timestamp', '') for log in access_logs if log.get('access_timestamp')])
        if last_access:
            st.write(f"**Last Accessed:** {format_date(last_access)}")

def display_download_button(file_data):
    """
    Display download button for a file.
    
    Args:
        file_data (dict): File data from database
    """
    file_path = file_data.get('file_path', '')
    filename = file_data.get('filename', 'download')
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as file:
                file_bytes = file.read()
                
            st.download_button(
                label="📥 Download",
                data=file_bytes,
                file_name=filename,
                mime=file_data.get('mime_type', 'application/octet-stream'),
                help=f"Download {filename}",
                key=f"download_{file_data['id']}",
                use_container_width=True
            )
        except Exception as e:
            #logger.error(f"Error reading file for download {file_path}: {e}")
            st.error("❌ Error preparing download")
    else:
        st.warning("⚠️ File not found")

def format_file_size(size_bytes):
    """
    Convert file size to human readable format.
    
    Args:
        size_bytes (int): File size in bytes
        
    Returns:
        str: Formatted file size
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
        
    return f"{size_bytes:.1f} {size_names[i]}"

def format_date(date_string):
    """
    Format date string to readable format.
    
    Args:
        date_string (str): ISO format date string
        
    Returns:
        str: Formatted date
    """
    try:
        if isinstance(date_string, str):
            from datetime import datetime
            # Handle both with and without timezone
            if 'Z' in date_string:
                dt = datetime.fromisoformat(date_string.replace('Z', '+00:00'))
            else:
                dt = datetime.fromisoformat(date_string)
            return dt.strftime("%Y-%m-%d %H:%M")
        return str(date_string)
    except:
        return str(date_string)