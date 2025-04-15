import os
import tempfile
import time
import gradio as gr
import tiktoken
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader

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
DOC_LOADERS = {".pdf": PyPDFLoader, ".txt": TextLoader, ".md": UnstructuredMarkdownLoader}

class ChromaManager:
    @staticmethod
    def get_db(collection_name=None):
        """Get Chroma DB instance with optional collection name"""
        kwargs = {"persist_directory": CHROMA_DIR, "embedding_function": EMBEDDINGS}
        if collection_name:
            kwargs["collection_name"] = collection_name
        return Chroma(**kwargs)

    @staticmethod
    def delete_all(collection_name=None):
        """Delete all documents from a collection"""
        try:
            db = ChromaManager.get_db(collection_name)
            db.delete_collection()
            db = ChromaManager.get_db(collection_name)  # Recreate empty collection
            return True
        except Exception:
            return False

def load_document(file_path):
    """Load documents based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    if loader := DOC_LOADERS.get(ext):
        return loader(file_path).load()
    raise ValueError(f"Unsupported file type: {ext}")

def store_documents(files):
    """Store documents in Chroma vector DB"""
    if not files:
        return "❌ No files selected"
    try:
        os.makedirs(CHROMA_DIR, exist_ok=True)
        db = ChromaManager.get_db()
        all_docs = []
        for file in files:
            all_docs.extend(load_document(file.name))
        db.add_documents(all_docs)
        return f"✅ Stored {len(all_docs)} documents successfully!"
    except Exception as e:
        return f"❌ Error storing documents: {str(e)}"

def create_qa_chain():
    """Create QA chain with RAG"""
    llm = OpenAI(**LLM_CONFIG)
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are an expert assistant. Use the context to answer accurately.\n"
            "If you don't know the answer, say \"I don't know\".\n\n"
            "Context:\n{context}\n\nQuestion:\n{question}\nAnswer:"
        )
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=ChromaManager.get_db().as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

class ChatManager:
    @staticmethod
    def save_history(history):
        """Save chat history to Chroma DB"""
        if not history:
            return
        db = ChromaManager.get_db(CHAT_COLLECTION)
        docs = [
            Document(
                page_content=msg["content"],
                metadata={"role": msg["role"], "index": i, "timestamp": msg.get("timestamp", time.time())}
            )
            for i, msg in enumerate(history)
        ]
        db.add_documents(docs)

    @staticmethod
    def load_history():
        """Load chat history from Chroma DB"""
        try:
            db = ChromaManager.get_db(CHAT_COLLECTION)
            docs = db.get()
            if not docs or not docs["documents"]:
                return []
            chat = [
                {
                    "role": meta.get("role", "user"),
                    "content": doc,
                    "timestamp": meta.get("timestamp", 0)
                }
                for doc, meta in zip(docs["documents"], docs["metadatas"])
            ]
            return sorted(chat, key=lambda x: (x.get("timestamp", 0), x.get("role", "")))
        except Exception:
            return []

    @staticmethod
    def delete_history():
        """Delete chat history"""
        return "🗑️ Chat history deleted!" if ChromaManager.delete_all(CHAT_COLLECTION) else "❌ Error deleting chat history"

def process_query(message, history):
    """Process user query and update chat history"""
    # Ensure message is a string
    if not isinstance(message, str):
        message = str(message) if message else ""
    
    # Ensure history is a list
    if not isinstance(history, list):
        history = []
    
    result = create_qa_chain()({"query": message})
    answer = result["result"]
    sources = "\n".join(doc.metadata.get("source", "N/A") for doc in result["source_documents"])
    
    timestamp = time.time()
    new_messages = [
        {"role": "user", "content": message, "timestamp": timestamp},
        {
            "role": "assistant",
            "content": f"{answer}\n\n**Sources:**\n{sources}\n\n🔢 Tokens: "
                      f"User {len(message.split())}, Assistant {len(answer.split())}, "
                      f"Total {len(message.split()) + len(answer.split())}",
            "timestamp": timestamp + 0.0001
        }
    ]
    
    new_history = history + new_messages
    ChatManager.save_history(new_history)
    
    # Format messages for display
    messages = [{"role": msg["role"], "content": msg["content"]} 
               for msg in sorted(new_history, key=lambda x: x.get("timestamp", 0))]
    
    return new_history, messages, ""

def create_interface():
    """Create Gradio interface"""
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="violet")) as demo:
        gr.Markdown(
            """<h1 style='text-align:center; color:#6C63FF;'>
            📚 RAG System with local LLM and Chroma DB</h1>"""
        )
        
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(label="Chat History", type="messages", height=400)  # Added height
            msg = gr.Textbox(
                placeholder="Ask about your documents...",
                label="Your message",  # Added label
                interactive=True,      # Ensure textbox is interactive
                lines=2               # Make textbox bigger
            )
            state = gr.State(ChatManager.load_history())
            
            with gr.Row():
                send_btn = gr.Button("Send")
                del_chat_btn = gr.Button("🗑️ Clear History")
            
            # Update process_query outputs to clear textbox
            def on_submit(history, message):
                if message.strip() == "":  # Don't process empty messages
                    return history, [], message
                new_history, messages, _ = process_query(message, history)
                return new_history, messages, ""  # Empty string to clear textbox
            
            msg.submit(on_submit, [state, msg], [state, chatbot, msg])
            send_btn.click(on_submit, [state, msg], [state, chatbot, msg])
            
            # Update delete history to clear chat
            def clear_chat():
                ChatManager.delete_history()
                return [], [], ""  # Clear history, chatbot, and textbox
            
            del_chat_btn.click(clear_chat, outputs=[state, chatbot, msg])
        
        with gr.Tab("Upload & Store"):
            gr.Markdown("Upload PDF/TXT/MD files to vector database")
            file_input = gr.File(file_types=list(DOC_LOADERS), file_count="multiple")
            status_box = gr.Textbox(label="Status")
            gr.Button("Store 🚀").click(store_documents, inputs=file_input, outputs=status_box)
        
        with gr.Tab("Manage"):
            with gr.Column():  # Use gr.Column for vertical layout
                delete_status = gr.Textbox(label="Delete Status")
                delete_button = gr.Button("🗑️ Delete All Documents")
                list_output = gr.Markdown()
                list_button = gr.Button("📄 List Documents")

            delete_button.click(
                lambda: "🗑️ Documents deleted!" if ChromaManager.delete_all() else "❌ Error deleting documents",
                outputs=delete_status
            )
            list_button.click(
                lambda: "\n---\n".join(
                    f"**Document {i+1}:**\n{doc}\n*Metadata:* {meta}"
                    for i, (doc, meta) in enumerate(zip(
                        ChromaManager.get_db().get().get("documents", []),
                        ChromaManager.get_db().get().get("metadatas", [])
                    ))
                ) or "No documents found",
                outputs=list_output
            )
        
        gr.Markdown(
            "<div style='text-align: center; margin-top: 2em; color: #888;'>"
            "Created by Maurizio Orani"
            "</div>"
        )
    
    return demo

if __name__ == "__main__":
    create_interface().launch()