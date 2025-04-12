import os
from typing import List, Tuple
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
import gradio as gr

class Config:
    PERSIST_DIRECTORY = "./chroma_db"
    EMBEDDING_MODEL = "nomic-embed-text"
    LLM_MODEL = "llama3.2:latest"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    SUPPORTED_TYPES = [".md", ".pdf"]

class RAGApp:
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        self.vector_store = Chroma(
            persist_directory=config.PERSIST_DIRECTORY,
            embedding_function=self.embeddings
        )
        self.llm = Ollama(model=config.LLM_MODEL)
        self.prompt_template = PromptTemplate(
            template="Context: {context}\n\nQuestion: {question}\nAnswer:",
            input_variables=["context", "question"]
        )

    def process_documents(self, files: List[gr.File]) -> str:
        """Process and ingest multiple documents"""
        errors = []
        for file in files:
            try:
                if file.name.endswith('.md'):
                    with open(file.name, 'r', encoding='utf-8') as f:
                        text = f.read()
                    chunks = self.text_splitter.split_text(text)
                elif file.name.endswith('.pdf'):
                    loader = PyPDFLoader(file.name)
                    documents = loader.load()
                    chunks = self.text_splitter.split_documents(documents)
                    chunks = [doc.page_content for doc in chunks]
                else:
                    raise ValueError(f"Unsupported file type: {file.name}")
                    
                self.vector_store.add_texts(chunks)
            except Exception as e:
                errors.append(f"Error processing {file.name}: {str(e)}")
        
        self.vector_store.persist()
        status = f"Processed {len(files)} files. Total chunks: {self.vector_store._collection.count()}"
        return status + self._format_errors(errors)
    
    def rag_query(self, query: str, chat_history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        """Execute RAG query and maintain chat history"""
        try:
            # Retrieve relevant documents
            docs = self.vector_store.similarity_search(query, k=4)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            # Generate response
            prompt = self.prompt_template.format(context=context, question=query)
            response = self.llm.invoke(prompt)
            
            # Update chat history
            new_history = chat_history + [(query, response)]
            return new_history, new_history
        except Exception as e:
            error_msg = f"Error during query execution: {str(e)}"
            return chat_history, [(query, error_msg)]
    
    @staticmethod
    def _format_errors(errors: List[str]) -> str:
        """Format error messages for display"""
        return "\n\nErrors Encountered:\n" + "\n".join(errors) if errors else ""

def create_gradio_interface(app: RAGApp):
    """Construct Gradio UI components"""
    with gr.Blocks(title="RAG System") as interface:
        gr.Markdown("# RAG Application with Llama3.2 and Chroma")
        
        with gr.Tab("Document Ingestion"):
            file_upload = gr.File(
                file_types=Config.SUPPORTED_TYPES,
                label="Upload Files",
                file_count="multiple"
            )
            load_btn = gr.Button("Load Documents")
            status_text = gr.Textbox(label="Status", interactive=False)
            
            load_btn.click(
                app.process_documents,
                inputs=file_upload,
                outputs=status_text
            )
        
        with gr.Tab("Query Interface"):
            chatbot = gr.Chatbot(label="Chat History")
            query_input = gr.Textbox(
                label="Enter your question",
                placeholder="Ask something...",
                scale=4
            )
            submit_btn = gr.Button("Submit", scale=1)
            
            submit_btn.click(
                app.rag_query,
                inputs=[query_input, chatbot],
                outputs=[chatbot, chatbot]
            )
    return interface

if __name__ == "__main__":
    config = Config()
    os.makedirs(config.PERSIST_DIRECTORY, exist_ok=True)
    
    app = RAGApp(config)
    interface = create_gradio_interface(app)
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )