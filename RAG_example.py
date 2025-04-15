import os
import tempfile
import gradio as gr
import tiktoken
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
from langchain_core.documents import Document
import time

# Constants
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
LLM_CONFIG = {
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "model": "llama3.2",
    "streaming": True
}
CHAT_COLLECTION = "chat_history"

# Document loaders registry
DOC_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader
}

def load_document(file_path):
    """Load documents based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if loader := DOC_LOADERS.get(ext):
        return loader(file_path).load()
    raise ValueError(f"Unsupported file type: {ext}")

def get_chroma_db():
    """Get or create Chroma database instance"""
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDINGS)

def store_documents(files):
    """Store documents in Chroma vector DB"""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    db = get_chroma_db()
    
    all_docs = []
    for file in files:
        all_docs.extend(load_document(file.name))
    
    db.add_documents(all_docs)
    return f"✅ Successfully stored {len(all_docs)} documents in Chroma DB!"

def count_tokens(text, model_name="gpt-3.5-turbo"):
    """Count tokens in text using tiktoken"""
    try:
        return len(tiktoken.encoding_for_model(model_name).encode(text))
    except Exception:
        return 0

def create_qa_chain():
    """Create preconfigured QA chain"""
    llm = OpenAI(**LLM_CONFIG)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""You are an expert assistant. Use the context to answer accurately.
        If you don't know the answer, say "I don't know".\n\n
        Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"""
    )
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=get_chroma_db().as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

def save_chat_history(history):
    """Save chat history as documents in Chroma DB (in a separate collection)."""
    chroma_dir = os.path.join(tempfile.gettempdir(), "chroma_db")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
        collection_name=CHAT_COLLECTION
    )
    docs = []
    for i, msg in enumerate(history):
        # Add a timestamp if not present
        metadata = {
            "role": msg["role"],
            "index": i,
            "timestamp": msg.get("timestamp", time.time())
        }
        docs.append(
            Document(
                page_content=msg["content"],
                metadata=metadata
            )
        )
    if docs:
        db.add_documents(docs)

def load_chat_history():
    """Load chat history from Chroma DB (from the chat_history collection)."""
    chroma_dir = os.path.join(tempfile.gettempdir(), "chroma_db")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(
        persist_directory=chroma_dir,
        embedding_function=embeddings,
        collection_name=CHAT_COLLECTION
    )
    try:
        docs = db.get()
        if not docs or not docs["documents"]:
            return []
        # Sort by timestamp (or index as fallback)
        chat = [
            {
                "role": meta.get("role", "user"),
                "content": doc,
                "timestamp": meta.get("timestamp", 0)
            }
            for doc, meta in zip(docs["documents"], docs["metadatas"])
        ]
        chat.sort(key=lambda x: (x.get("timestamp", 0), x.get("role", "")))
        return chat
    except Exception:
        return []

def process_query(message, history):
    """Process user query and generate response"""
    import time
    # Ensure message is a string
    if isinstance(message, list):
        message = message[-1] if message else ""
    if not isinstance(message, str):
        message = str(message)
    result = create_qa_chain()({"query": message})
    answer = result["result"]
    sources = "\n".join(doc.metadata.get("source", "N/A") for doc in result["source_documents"])
    user_tokens = count_tokens(message)
    answer_tokens = count_tokens(answer)
    token_info = f"🔢 Tokens: User {user_tokens}, Assistant {answer_tokens}, Total {user_tokens + answer_tokens}"
    timestamp = time.time()
    new_history = history + [
        {"role": "user", "content": message, "timestamp": timestamp},
        {"role": "assistant", "content": f"{answer}\n\n**Sources:**\n{sources}\n\n{token_info}", "timestamp": timestamp + 0.0001}
    ]
    save_chat_history(new_history)
    # Sort and format for gr.Chatbot (type="messages")
    sorted_history = sorted(new_history, key=lambda x: x.get("timestamp", 0))
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in sorted_history]
    return new_history, messages, ""

def delete_documents():
    """Delete all documents from vector DB"""
    if os.path.exists(CHROMA_DIR):
        for root, dirs, files in os.walk(CHROMA_DIR, topdown=False):
            for name in files + dirs:
                os.remove(os.path.join(root, name)) if files else os.rmdir(os.path.join(root, name))
        os.rmdir(CHROMA_DIR)
        return "🗑️ All documents deleted!"
    return "No database found!"

def delete_chat_history():
    """Delete all chat history from the chat_history collection in Chroma DB."""
    chroma_dir = os.path.join(tempfile.gettempdir(), "chroma_db")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        db = Chroma(
            persist_directory=chroma_dir,
            embedding_function=embeddings,
            collection_name=CHAT_COLLECTION
        )
        # Remove all documents in the chat_history collection
        db._collection.delete(where={})  # Use the underlying Chroma collection's delete method
        return "🗑️ Chat history deleted!"
    except Exception as e:
        return f"Error deleting chat history: {e}"

def show_history(history):
    # Return chat history as a list of dicts for gr.Chatbot with type="messages"
    if not history:
        return gr.update(value=[], visible=True)
    # Sort by timestamp to ensure correct order
    sorted_history = sorted(history, key=lambda x: x.get("timestamp", 0))
    messages = [{"role": msg["role"], "content": msg["content"]} for msg in sorted_history]
    return gr.update(value=messages, visible=True)

# Gradio UI setup
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet")) as demo:
        gr.Markdown("""<h1 style='text-align:center; color:#6C63FF;'>
            📚 RAG System with local LLM and Chroma DB</h1>""")
        
        # 1. Chat interface tab
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Chat History", type="messages")
            msg = gr.Textbox(placeholder="Ask about your documents...")
            state = gr.State(load_chat_history())

            del_chat_btn = gr.Button("Delete Conversation History")

            def submit(history, message):
                return process_query(message, history)
            msg.submit(submit, [state, msg], [state, chatbot, msg])
            gr.Button("Send").click(submit, [state, msg], [state, chatbot, msg])

            del_chat_btn.click(delete_chat_history)
        
        # 2. Document management tab 
        with gr.Tab("Upload & Store"):
            gr.Markdown("Upload PDF/TXT/MD files to vector database")
            file_input = gr.File(file_types=list(DOC_LOADERS), file_count="multiple")
            status_box = gr.Textbox(label="Status")
            upload_btn = gr.Button("Store 🚀")
            upload_btn.click(store_documents, inputs=file_input, outputs=status_box)
        
        # 3. Database management
        with gr.Tab("Manage"):
            del_status = gr.Textbox(label="Delete Status")
            gr.Button("🗑️ Delete All").click(delete_documents, outputs=del_status)
            docs_output = gr.Markdown()
            def list_docs():
                docs = get_chroma_db().get()
                if not docs or not docs["documents"]:
                    return "No documents"
                return "\n---\n".join(
                    f"**Document {i+1}:**\n{doc}\n*Metadata:* {docs['metadatas'][i] if docs['metadatas'] else {}}"
                    for i, doc in enumerate(docs["documents"])
                )
            gr.Button("📄 List Docs").click(list_docs, outputs=docs_output)
    
    return demo

if __name__ == "__main__":
    create_interface().launch()