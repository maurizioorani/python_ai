# AI PROJECTS - Python, LangChain and local LLMs

This project demonstrates some simple applications showcasing interaction with a local Ollama instance for different tasks.

## Applications

### 1. Simple Chat with Ollama (simpleChat.py)

This application provides a basic interface to chat with a locally running Ollama language model. It demonstrates how to send prompts and receive responses using a few lines of code, abstracting away the complexities of the underlying API calls.

### 2. Intelligent Link Extraction from Websites (inference_with_knowledge.py)

This more advanced application scrapes the links from a specified website and leverages the local Ollama AI to identify the most relevant URLs to understand companies' core business. This could be useful for tasks like automatically generating a list of key pages (e.g., "About Us", "Careers") for a company brochure.

### 3. RAG System with Ollama (RAG_example.py)

AI-powered question-answering system for markdown documentation using Ollama and LangChain.

## Features

- 📄 Processes markdown files into vector database
- 🤖 Powered by Ollama models (default: Mistral)
- 🔍 Interactive Gradio web interface
- 🗂️ Persistent vector storage with ChromaDB
- 📚 Maintains conversation history
- 📋 Shows answer sources from original documents

## Installation

```bash
# Install dependencies
pip install -r RAG_requirements.txt

# Pull required Ollama models
ollama pull mistral
ollama pull nomic-embed-text
```

## Commit History
Here's a brief overview of the project's development history:

* **2025-04-10:**
    * **feat: Implement simple chat with local Ollama instance (simpleChat.py)**
        * Introduced the basic chat application, enabling interaction with a local Ollama model.

* **2025-04-11:**
    * **feat: Add intelligent link extraction for business description (inference_with_knowledge.py)**
        * Implemented the functionality to scrape website links and use AI to identify relevant URLs for describing a company's core business.

* **2025-04-12:**
    * **feat: RAG system (RAG_example.py)**
        * Implemented the functionality to read md files in a given directory, store chunks of text in a vector database and integrate the knowledge into the local LLM

## Usage

* Ensure you have Ollama installed and running locally (http://localhost:11434). 
* You also need to pip install other packages.
* Please use requirements.txt if you use pip

```python
# install all the dependencies
pip install -r requirements.txt
# Use environment.yml if you use conda

# For simpleChat.py:
python simpleChat.py

# For inference_with_knowledge.py:
python inference_with_knowledge.py [https://www.example-website.com]

# For RAG_example.py:
python RAG_example.py. Browse to http://localhost:7860, upload a PDF document and query the LLM with questions about the content.
