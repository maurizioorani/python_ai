import os
import tempfile
import time
from typing import List, Dict, Any, Optional
import gradio as gr
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Constants and Configuration
CHROMA_DIR = os.path.join(tempfile.gettempdir(), "chroma_db")
EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
LLM_CONFIG = {
    "base_url": "http://localhost:11434/v1",
    "api_key": "ollama",
    "model": "llama3.2",
    "streaming": True
}
CHAT_COLLECTION = "chat_history"
DOC_LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": UnstructuredMarkdownLoader
}

# Database and Document Management
class DatabaseManager:
    """Handle all database operations"""
    
    @staticmethod
    def get_db(collection_name: Optional[str] = None) -> Chroma:
        """Get or create a Chroma database instance"""
        kwargs = {"persist_directory": CHROMA_DIR, "embedding_function": EMBEDDINGS}
        if collection_name:
            kwargs["collection_name"] = collection_name
        return Chroma(**kwargs)
    
    @staticmethod
    def delete_collection(collection_name: Optional[str] = None) -> bool:
        """Delete a collection and recreate it empty"""
        try:
            db = DatabaseManager.get_db(collection_name)
            db.delete_collection()
            DatabaseManager.get_db(collection_name)  # Recreate empty
            return True
        except Exception:
            return False

class DocumentHandler:
    """Handle document operations"""
    
    @staticmethod
    def load_document(file_path: str) -> List[Document]:
        """Load document based on file extension and split into chunks"""
        ext = os.path.splitext(file_path)[1].lower()
        if loader := DOC_LOADERS.get(ext):
            try:
                # Load the document
                docs = loader(file_path).load()
                
                # Initialize text splitter
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=400,            
                    chunk_overlap=20,          
                    length_function=len,
                    separators=["\n\n", "\n", ". ", " ", ""],
                    keep_separator=False
                )
                
                # Process documents 
                processed_docs = []
                for doc in docs:
                    # Extract and clean text content
                    if hasattr(doc, 'page_content'):
                        text = doc.page_content
                    else:
                        text = str(doc)
                    
                    # Basic text cleaning
                    text = text.encode('ascii', 'ignore').decode()  # Remove non-ASCII chars
                    text = ' '.join(text.split())  # Normalize whitespace
                    
                    # Get metadata
                    metadata = doc.metadata if hasattr(doc, 'metadata') else {}
                    
                    # Split text into chunks
                    chunks = text_splitter.create_documents(
                        texts=[text],
                        metadatas=[metadata]
                    )
                    processed_docs.extend(chunks)
                
                return processed_docs
                
            except Exception as e:
                print(f"Error processing document: {str(e)}")  # Debug info
                raise ValueError(f"Error processing document: {str(e)}")
                
        raise ValueError(f"Unsupported file type: {ext}")

    @staticmethod
    def store_documents(files: List[Any]) -> str:
        """Store documents in vector DB"""
        if not files:
            return "❌ No files selected"
        try:
            os.makedirs(CHROMA_DIR, exist_ok=True)
            db = DatabaseManager.get_db()
            all_docs = []
            total_chunks = 0
            
            for file in files:
                try:
                    chunks = DocumentHandler.load_document(file.name)
                    batch_size = 20  # Adjust batch size as needed
                    for i in range(0, len(chunks), batch_size):
                        batch = chunks[i:i + batch_size]
                        db.add_documents(batch)
                        total_chunks += len(batch)
                except Exception as e:
                    print(f"Error processing {file.name}: {str(e)}")  # Debug info
                    continue
            
            if total_chunks == 0:
                return "❌ No documents were processed successfully"
            
            return f"✅ Processed {len(files)} files into {total_chunks} chunks successfully!"
            
        except Exception as e:
            print(f"Store error: {str(e)}")  # Debug info
            return f"❌ Error: {str(e)}"

class ChatManager:
    """Manage chat operations"""
    
    @staticmethod
    def save_history(history: List[Dict[str, Any]]) -> None:
        """Save chat history to DB"""
        if not history:
            return
        db = DatabaseManager.get_db(CHAT_COLLECTION)
        docs = [
            Document(
                page_content=msg["content"],
                metadata={
                    "role": msg["role"],
                    "index": i,
                    "timestamp": msg.get("timestamp", time.time())
                }
            )
            for i, msg in enumerate(history)
        ]
        db.add_documents(docs)
    
    @staticmethod
    def load_history() -> List[Dict[str, Any]]:
        """Load chat history from DB"""
        try:
            db = DatabaseManager.get_db(CHAT_COLLECTION)
            docs = db.get()
            if not docs or not docs["documents"]:
                return []
            return sorted(
                [
                    {
                        "role": meta.get("role", "user"),
                        "content": doc,
                        "timestamp": meta.get("timestamp", 0)
                    }
                    for doc, meta in zip(docs["documents"], docs["metadatas"])
                ],
                key=lambda x: (x.get("timestamp", 0), x.get("role", ""))
            )
        except Exception:
            return []
    
    @staticmethod
    def delete_history() -> bool:
        """Delete chat history"""
        return DatabaseManager.delete_collection(CHAT_COLLECTION)

class QASystem:
    """Handle Q&A operations"""
    
    @staticmethod
    def create_chain():
        """Create QA chain with improved context handling"""
        llm = OpenAI(**LLM_CONFIG)
        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are an expert assistant. Analyze the following context thoroughly and answer the question.\n"
                "If you don't know the answer, say \"I don't know\".\n"
                "Provide a detailed answer using the available information.\n\n"
                "Context:\n{context}\n\n"
                "Question:\n{question}\n\n"
                "Detailed Answer:"
            )
        )
        
    
        retriever = DatabaseManager.get_db().as_retriever(
            search_type="mmr",  
            search_kwargs={
                "k": 5,       
                "fetch_k": 8,  
                "lambda_mult": 0.7  
            }
        )
        
        return RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": prompt,
                "document_separator": "\n\n",  # Clear separator between documents
                "verbose": True
            }
        )
    
    @staticmethod
    def process_query(message: str, history: List[Dict[str, Any]]) -> tuple:
        """Process query and update history with improved source handling"""
        if not isinstance(message, str):
            message = str(message) if message else ""
        if not isinstance(history, list):
            history = []
        
        # Get response from QA chain
        result = QASystem.create_chain()({"query": message})
        answer = result["result"]
        
        # source handling
        sources = []
        for i, doc in enumerate(result["source_documents"], 1):
            source = doc.metadata.get("source", "N/A")
            page = doc.metadata.get("page", "")
            chunk = doc.page_content[:150] + "..."  # Preview of content
            
            sources.append(
                f"**Source {i}:**\n"
                f"- File: {os.path.basename(source)}\n"
                f"- Page: {page}\n"
                f"- Preview: {chunk}\n"
            )
        
        source_text = "\n".join(sources)
        
        timestamp = time.time()
        new_messages = [
            {"role": "user", "content": message, "timestamp": timestamp},
            {
                "role": "assistant",
                "content": (
                    f"{answer}\n\n"
                    f"**Referenced Sources:**\n"
                    f"{source_text}"
                ),
                "timestamp": timestamp + 0.0001
            }
        ]
        
        new_history = history + new_messages
        ChatManager.save_history(new_history)
        messages = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in sorted(new_history, key=lambda x: x.get("timestamp", 0))
        ]
        
        return new_history, messages, ""

def create_interface() -> gr.Blocks:
    """Create Gradio interface"""
    with gr.Blocks(
        title="RAG Chat System 🤖",  
        theme=gr.themes.Soft(primary_hue="violet")
    ) as demo:
        gr.Markdown(
            """<h1 style='text-align:center; color:#6C63FF;'>
            📚 RAG System with Local LLM</h1>"""
        )
        
        # Chat tab
        with gr.Tab("Chat"):
            chatbot = gr.Chatbot(
                label="Chat History",
                type="messages",
                height=400
            )
            msg = gr.Textbox(
                label="Your message",
                placeholder="Ask about your documents...",
                lines=2,
                max_lines=10,
                show_label=True,
                interactive=True
            )
            state = gr.State(ChatManager.load_history())
            
            with gr.Row():
                send_btn = gr.Button("Send 📤")
                clear_btn = gr.Button("Clear 🗑️")
            
            def on_submit(history: List, message: str) -> tuple:
                """Handle message submission"""
                if not message.strip():
                    return history, [], message
                return QASystem.process_query(message, history)
            
            msg.submit(on_submit, [state, msg], [state, chatbot, msg])
            send_btn.click(on_submit, [state, msg], [state, chatbot, msg])
            clear_btn.click(
                lambda: ([], [], ""),
                outputs=[state, chatbot, msg]
            )
        
        # Upload & Store tab
        with gr.Tab("Documents"):
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Documents",
                        file_types=list(DOC_LOADERS),
                        file_count="multiple"
                    )
                    upload_btn = gr.Button("Store Documents 📥")
                    status = gr.Textbox(label="Status")
                    upload_btn.click(
                        DocumentHandler.store_documents,
                        inputs=file_input,
                        outputs=status
                    )
        
        # Manage tab
        with gr.Tab("Manage"):
            with gr.Column(scale=1):
                gr.Markdown("### Database Management")
                with gr.Row():
                    delete_btn = gr.Button("🗑️ Delete All Documents", variant="stop")
                    list_btn = gr.Button("📄 List Documents", variant="secondary")
                
                docs_output = gr.Markdown()
                status_output = gr.Textbox(label="Status", interactive=False)
                
                delete_btn.click(
                    lambda: "🗑️ Documents deleted!" if DatabaseManager.delete_collection() else "❌ Error deleting documents",
                    outputs=status_output
                )
                
                def list_documents():
                    docs = DatabaseManager.get_db().get()
                    if not docs or not docs["documents"]:
                        return "No documents found in database"
                    return "\n\n---\n\n".join(
                        f"**Document {i+1}:**\n{doc}\n*Metadata:* {meta}"
                        for i, (doc, meta) in enumerate(zip(
                            docs["documents"],
                            docs["metadatas"]
                        ))
                    )
                
                list_btn.click(list_documents, outputs=docs_output)
        
        gr.Markdown(
            "<div style='text-align:center; margin-top:2em; color:#888;'>"
            "Created by Maurizio Orani"
            "</div>"
        )
    
    return demo

if __name__ == "__main__":
    create_interface().launch(
        server_name="localhost",
        server_port=7860, 
        share=False,
        show_error=True
    )