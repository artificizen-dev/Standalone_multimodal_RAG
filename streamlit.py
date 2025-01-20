import streamlit as st
import requests
import os

# FastAPI backend URL
BACKEND_URL = "http://localhost:8000"

# Set page config
st.set_page_config(page_title="Training Document System", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select a page",
    ["Upload Materials", "View Materials", "Chat with Documents"]
)

# Initialize session state for uploaded files
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# Display current user
st.sidebar.divider()
st.sidebar.text(f"Current User: {st.session_state.get('user_id', 'default_user')}")

# Function to render the Upload Materials page
def render_upload_page():
    st.header("Upload Materials")
    
    # File uploader for videos and PDFs
    uploaded_file = st.file_uploader("Choose a file to upload", type=["mp4", "pdf"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        file_path = os.path.join("uploads", uploaded_file.name)
        os.makedirs("uploads", exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Determine the file type and upload to the backend
        if uploaded_file.type.startswith("video"):
            endpoint = "/upload-video/"
        elif uploaded_file.type == "application/pdf":
            endpoint = "/upload-pdf/"
        else:
            st.error("Unsupported file type. Please upload a video (mp4) or PDF.")
            return
        
        # Upload the file to the backend
        with st.spinner(f"Uploading {uploaded_file.name}..."):
            files = {"file": open(file_path, "rb")}
            response = requests.post(f"{BACKEND_URL}{endpoint}", files=files)
        
        if response.status_code == 200:
            st.success(f"File uploaded successfully: {uploaded_file.name}")
            st.session_state.uploaded_files.append(uploaded_file.name)
            st.json(response.json())
        else:
            st.error(f"Failed to upload file: {response.text}")

# Function to render the View Materials page
def render_view_materials_page():
    st.header("View Uploaded Materials")
    
    if not st.session_state.uploaded_files:
        st.info("No files have been uploaded yet.")
    else:
        st.write("List of uploaded files:")
        for file_name in st.session_state.uploaded_files:
            st.write(f"- {file_name}")

# Function to render the Chat with Documents page
def render_chatbot_page():
    st.header("Chat with Documents")
    
    # Select a file to query
    if not st.session_state.uploaded_files:
        st.info("No files have been uploaded yet. Please upload a file first.")
        return
    
    selected_file = st.selectbox("Select a file to query", st.session_state.uploaded_files)
    query = st.text_input("Enter your query:")
    
    if st.button("Submit Query"):
        if selected_file and query:
            # Determine the file type and query the backend
            if selected_file.endswith(".mp4"):
                endpoint = "/query-video/"
                payload = {"video_path": selected_file, "query": query}
            elif selected_file.endswith(".pdf"):
                endpoint = "/query-pdf/"
                payload = {"file_name": selected_file, "query": query}
            else:
                st.error("Unsupported file type.")
                return
            
            # Query the backend
            with st.spinner("Processing query..."):
                response = requests.post(f"{BACKEND_URL}{endpoint}", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                st.success("Query processed successfully!")
                st.json(result)
                
                # Display relevant results
                if "answer" in result:
                    st.subheader("Answer:")
                    st.write(result["answer"])
                
                if "highlights" in result:
                    st.subheader("Highlights:")
                    for highlight in result["highlights"]:
                        st.write(f"Page {highlight['page_num']}: {highlight['text']}")
                
                if "subclip_path" in result:
                    st.subheader("Relevant Video Segment:")
                    st.video(f"{BACKEND_URL}/get-subclip/?subclip_path={result['subclip_path']}")
            else:
                st.error(f"Failed to process query: {response.text}")
        else:
            st.warning("Please provide both a file and a query.")

# Main function to create the Streamlit app
def create_streamlit_app():
    if page == "Upload Materials":
        render_upload_page()
    elif page == "View Materials":
        render_view_materials_page()
    else:  
        render_chatbot_page()

if __name__ == "__main__":
    create_streamlit_app()