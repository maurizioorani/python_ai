# 🤖🧠 AI PROJECTS - Python, LangChain and Local LLMs

This project demonstrates applications showcasing interaction with a local **Ollama** instance 🦙 for various AI tasks using **Python** and **LangChain**.

## ✨ Applications

### 1. RAG System with Local LLM (Ollama) and ChromaDB (`RAG_example.py`) 📚🔍

This project implements a **Retrieval-Augmented Generation (RAG)** system using:
* A local Large Language Model (LLM) served by **Ollama** 🦙
* A **Chroma** vector database 💾 for document storage and retrieval
* A **Gradio** web interface 🌐 for interaction.

It allows you to upload your documents (PDF, TXT, MD), ask questions about their content, and receive answers generated by a local LLM, ensuring your data stays **private** 🔒. The system also keeps track of your chat history 🕒.

#### Features

* 🔒 **Local Processing:** Uses Ollama to run LLMs locally, keeping your data private.
* 📤 **Document Upload:** Supports uploading PDF, TXT, and Markdown (.md) files.
* 💾 **Vector Storage:** Uses ChromaDB to store document embeddings for efficient retrieval.
* 💬 **Chat Interface:** Provides a user-friendly Gradio interface to chat with your documents.
* 📚 **Source Citation:** Displays the source documents used to generate the answer.
* 🔢 **Token Count:** Shows the number of tokens used in the user query and the assistant's response.
* 🕒 **Chat History:** Saves and loads conversation history (also stored in ChromaDB).
* ⚙️ **Management:** Allows listing stored documents, deleting all documents, and deleting chat history.

#### Prerequisites ✅

* Python 3.8+ 🐍
* `pip` (Python package installer) 📦
* Git (for cloning the repository) 🐙
* [Ollama](https://ollama.com/) installed and running 🦙💨.

#### Setup Instructions ⚙️

1.  **Clone the Repository:** 💻
    ```bash
    git clone https://github.com/maurizioorani/python_ai/tree/main
    ```

2.  **Create a Virtual Environment (Recommended):** 🌱
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Create the Requirements File (`RAG_requirements.txt`):** 📄
    Create a file named `RAG_requirements.txt` in the project directory with the following content:
    ```txt
    gradio
    langchain
    langchain-chroma
    langchain-huggingface
    langchain-openai # Used for Ollama's OpenAI-compatible API
    tiktoken
    pypdf
    unstructured[md] # Installs markdown parsing extras for unstructured
    sentence-transformers # For HuggingFaceEmbeddings
    chromadb >= 0.4.20 # Ensure a recent version for compatibility
    ollama # Optional, but useful for managing Ollama via command line
    ```
    *Note: You might need to adjust versions based on compatibility.*

4.  **Install Dependencies:** ⬇️
    Ensure your virtual environment is active, then run:
    ```bash
    pip install -r RAG_requirements.txt
    ```
    Alternatively, if an `environment.yml` is provided for Conda:
    ```bash
    conda env create -f environment.yml # Or conda install --file environment.yml
    ```

5.  **Setup Ollama:** 🦙
    * **Install Ollama:** Follow instructions on the [Ollama website](https://ollama.com/).
    * **Run Ollama:** Ensure the Ollama service is running (e.g., run the installed application or use `ollama serve` in the terminal).
    * **Verify Ollama:** Check if Ollama is running by opening a terminal and typing:
        ```bash
        ollama list
        ```
        This command should list your installed models if Ollama is running and accessible.

6.  **Pull the LLM Model:** 🧠
    The RAG script is configured by default to use `llama3.2`. Pull this model using the Ollama CLI:
    ```bash
    ollama pull llama3.2
    ```
    Wait for the download to complete. You can verify the model is installed using `ollama list` again.
    *Note: If you wish to use a different model, update the `model` value within the Python script (likely in an `LLM_CONFIG` dictionary) and pull the corresponding model using `ollama pull <new_model_name>`.*

#### Running the RAG Application ▶️

Once the setup is complete and Ollama is running with the required model:
```bash
python RAG_example.py
```
Then, open your web browser and navigate to http://localhost:7860 🌐. You should see the Gradio interface where you can upload documents and interact with the RAG system.

Please note: For optimal performance in languages other than English, consider using larger models or those specifically trained for multilingual tasks, as smaller CPU-only models may produce unexpected results due to limited non-English training.

# AI additional applications 🚀

## 2. Simple Chat with Ollama (`simpleChat.py`) 💬🤖  
A minimalist command-line interface for interacting with locally hosted Ollama models. Perfect for:  
✅ Learning LLM interaction fundamentals  
✅ Testing model responses  
✅ Prototyping without UI complexity  

---

## 3. Intelligent Link Extraction (`inference_with_knowledge.py`) 🔗🌐  
Advanced web analysis tool featuring:  

1. **Website Scraping** - Automatically harvest all hyperlinks  
2. **AI Analysis** - Local Ollama-powered relevance scoring 🧠  
3. **Business Intelligence** - Identifies core business sections:  
   - About Us  
   - Services/Products  
   - Contact Information  
   - Company Values  

*Perfect for competitive analysis and website mapping!*

---


## 🚀 Usage Guide  

### Prerequisites  
🦙 **Ollama Running**:  
```bash
ollama serve  # Keep running in background
