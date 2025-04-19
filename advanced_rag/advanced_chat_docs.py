import os
import tempfile
import streamlit as st
from embedchain import App
import base64
from streamlit_chat import message
import uuid
import json
import time
import ssl

# Define the embedchain_bot function
def embedchain_bot(db_path):
    return App.from_config(
        config={
            "llm": {"provider": "ollama", "config": {"model": "llama3.2:latest", "max_tokens": 250, "temperature": 0.5, "stream": True, "base_url": 'http://localhost:11434'}},
            "vectordb": {"provider": "chroma", "config": {"dir": db_path}},
            "embedder": {"provider": "ollama", "config": {"model": "llama3.2:latest", "base_url": 'http://localhost:11434'}},
        }
    )

# Document tracking with persistence
def get_documents_file_path():
    """Get the path to the documents tracking file"""
    # Use the same directory as the vector database for consistency
    if 'db_path' in st.session_state:
        return os.path.join(st.session_state.db_path, "document_list.json")
    return "document_list.json"

def save_documents():
    """Save the document list to a file"""
    try:
        with open(get_documents_file_path(), 'w') as f:
            json.dump(st.session_state.documents, f)
    except Exception as e:
        st.error(f"Error saving document list: {str(e)}")

def load_documents():
    """Load the document list from file"""
    try:
        if os.path.exists(get_documents_file_path()):
            with open(get_documents_file_path(), 'r') as f:
                return json.load(f)
    except Exception as e:
        st.error(f"Error loading document list: {str(e)}")
    return []

def init_document_state():
    """Initialize document tracking state"""
    if 'documents' not in st.session_state:
        st.session_state.documents = load_documents()

def add_document(filename, file_id, doc_type):
    """Add document to tracking list"""
    st.session_state.documents.append({
        "filename": filename,
        "id": file_id,
        "type": doc_type,
        "added_time": time.time()
    })
    save_documents()

def delete_document(file_id):
    """Delete document from the knowledge base and tracking list"""
    try:
        # Try using the most common delete methods
        try:
            # Try by direct ID (most common in newer versions)
            st.session_state.app.delete(file_id)
        except Exception:
            # Try by querying metadata
            try:
                st.session_state.app.delete(where={"id": file_id})
            except Exception:
                # Last attempt with where clause on other fields
                st.session_state.app.delete(where={"metadata.id": file_id})
        
        # Update the documents list
        st.session_state.documents = [doc for doc in st.session_state.documents if doc["id"] != file_id]
        save_documents()
        return True
    except Exception as e:
        st.error(f"Error deleting document: {str(e)}")
        return False

def delete_all_documents():
    """Delete all documents from the knowledge base"""
    try:
        # Prefer reset if available
        try:
            st.session_state.app.reset()
        except Exception:
            # Fall back to individual deletion
            for doc in st.session_state.documents:
                try:
                    delete_document(doc["id"])
                except Exception:
                    pass  # Continue with others if one fails
        
        # Clear the document list
        st.session_state.documents = []
        save_documents()
        return True
    except Exception as e:
        st.error(f"Error deleting all documents: {str(e)}")
        return False

# Add a function to display files
def display_file(file, file_type=None):
    """
    Display various file types in Streamlit
    
    Parameters:
    - file: The file object opened in binary mode
    - file_type: Optional string to specify file type ('pdf', 'txt', 'md')
                If None, will try to infer from file name
    """
    if file_type is None:
        # Try to infer file type from name if not specified
        file_name = file.name if hasattr(file, 'name') else ''
        if file_name.lower().endswith('.pdf'):
            file_type = 'pdf'
        elif file_name.lower().endswith('.txt'):
            file_type = 'txt'
        elif file_name.lower().endswith(('.md', '.markdown')):
            file_type = 'md'
        else:
            st.error("Could not determine file type. Please specify file_type parameter.")
            return
    
    # Reset file position
    if hasattr(file, 'seek'):
        file.seek(0)
    
    # Handle different file types
    if file_type.lower() == 'pdf':
        # Display PDF using iframe
        base64_pdf = base64.b64encode(file.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    
    elif file_type.lower() == 'txt':
        # Display text file
        text_content = file.read().decode('utf-8')
        st.text_area("File Content", text_content, height=400)
        
        # Option to process with LLM
        if st.button("Process with LLM"):
            process_with_llm(text_content)
    
    elif file_type.lower() == 'md':
        # Display markdown file
        md_content = file.read().decode('utf-8')
        st.markdown(md_content)
        
        # Option to process with LLM
        if st.button("Process with Markdown LLM"):
            process_with_llm(md_content)
    
    else:
        st.error(f"Unsupported file type: {file_type}")

def process_with_llm(content):
    """Process content with local LLM using Ollama"""
    with st.spinner("Processing with local LLM..."):
        try:
            # Trim content to reasonable size to prevent timeouts
            trimmed_content = content[:4000] + ("..." if len(content) > 4000 else "")
            
            # Use the same App instance that's already set up with proper configurations
            if 'app' in st.session_state:
                # Create a prompt for summarization or analysis
                prompt = f"Please analyze and summarize the following text:\n\n{trimmed_content}"
                
                # Use the existing app instance to process the content
                response = st.session_state.app.chat(prompt)
                
                # Display the result
                st.subheader("LLM Analysis Results")
                st.write(response)
            else:
                st.error("LLM app not initialized. Please ensure the app is set up correctly.")
                
        except Exception as e:
            st.error(f"Error processing with LLM: {str(e)}")
            st.info("Make sure Ollama is running locally with the correct model.")

# Main app
st.title("Chat with Documents using Llama 3.2")
st.caption("This app allows you to chat with your documents using Llama 3.2 running locally with Ollama!")

# Define and store the database path
if 'db_path' not in st.session_state:
    # Use a more persistent location for the database
    db_path = os.path.join(tempfile.gettempdir(), "embedchain_kb")
    os.makedirs(db_path, exist_ok=True)
    st.session_state.db_path = db_path
else:
    db_path = st.session_state.db_path

# Create a session state to store the app instance and chat history
if 'app' not in st.session_state:
    st.session_state.app = embedchain_bot(db_path)
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Initialize document tracking
init_document_state()

# Create tabs for main interface and document management
tab1, tab2 = st.tabs(["Chat Interface", "Document Management"])

with tab1:
    # Sidebar for file upload and preview
    with st.sidebar:
        st.header("File Upload")
        
        # Create tabs for different file types
        file_type_tab = st.radio("Select file type:", ["PDF", "Text", "Markdown"])
        
        if file_type_tab == "PDF":
            uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
            file_format = "pdf"
            data_type = "pdf_file"
        elif file_type_tab == "Text":
            uploaded_file = st.file_uploader("Upload a text file", type="txt")
            file_format = "txt"
            data_type = "text"
        else:  # Markdown
            uploaded_file = st.file_uploader("Upload a markdown file", type=["md", "markdown"])
            file_format = "md"
            data_type = "text"

        if uploaded_file:
            st.subheader(f"{file_type_tab} Preview")
            # Create a file-like object for display
            file_bytes = uploaded_file.getvalue()
            file_buffer = uploaded_file
            # Reset buffer position for display
            file_buffer.seek(0)
            display_file(file_buffer, file_type=file_format)
            
            if st.button(f"Add to Knowledge Base"):
                with st.spinner(f"Adding {file_type_tab} to knowledge base..."):
                    # Generate a unique ID based on filename
                    file_id = f"{uploaded_file.name}_{uuid.uuid4().hex[:8]}"
                    
                    # Create a temporary file with appropriate extension
                    suffix = f".{file_format}"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                        f.write(file_bytes)
                        
                        try:
                            # Use consistent metadata format
                            metadata = {
                                "id": file_id,
                                "filename": uploaded_file.name,
                                "type": file_type_tab
                            }
                            
                            # Add to knowledge base
                            st.session_state.app.add(
                                f.name, 
                                data_type=data_type, 
                                metadata=metadata
                            )
                            
                            # Track the document
                            add_document(uploaded_file.name, file_id, file_type_tab)
                            
                        except Exception as e:
                            st.error(f"Error adding document: {str(e)}")
                        
                    os.remove(f.name)
                st.success(f"Added {uploaded_file.name} to knowledge base!")

    # Chat interface
    for i, msg in enumerate(st.session_state.messages):
        message(msg["content"], is_user=msg["role"] == "user", key=str(i))

    if prompt := st.chat_input("Ask a question about your documents"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        message(prompt, is_user=True)

        with st.spinner("Thinking..."):
            response = st.session_state.app.chat(prompt)
            st.session_state.messages.append({"role": "assistant", "content": response})
            message(response)

    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []

with tab2:
    st.header("Document Management")
    
    if not st.session_state.documents:
        st.info("No documents in the knowledge base yet.")
    else:
        st.write(f"You have {len(st.session_state.documents)} documents in your knowledge base.")
        
        # Display all documents with delete options
        st.subheader("Your Documents")
        
        # Create a table for better display
        columns = st.columns([3, 2, 1.5])
        columns[0].write("**Filename**")
        columns[1].write("**Type**")
        columns[2].write("**Actions**")
        
        for idx, doc in enumerate(st.session_state.documents):
            cols = st.columns([3, 2, 1.5])
            cols[0].write(doc["filename"])
            cols[1].write(doc["type"])
            
            # Use a unique key for each button based on index and id
            if cols[2].button("Delete", key=f"del_{idx}_{doc['id'][-4:]}"):
                if delete_document(doc["id"]):
                    st.success(f"Deleted {doc['filename']}")
                    st.experimental_rerun()
        
        # Option to delete all documents
        st.subheader("Delete All Documents")
        st.warning("This will remove all documents from your knowledge base. This action cannot be undone.")
        if st.button("Delete All Documents"):
            if delete_all_documents():
                st.success("All documents have been deleted from the knowledge base.")
                st.rerun()