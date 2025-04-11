# Simple Chat with Local Ollama and Intelligent Link Extraction

This project demonstrates two simple applications showcasing interaction with a local Ollama instance for different tasks.

## Applications

### 1. Simple Chat with Ollama (simpleChat.py)

This application provides a basic interface to chat with a locally running Ollama language model. It demonstrates how to send prompts and receive responses using a few lines of code, abstracting away the complexities of the underlying API calls.

### 2. Intelligent Link Extraction from Websites (inference_with_knowledge.py)

This more advanced application scrapes the links from a specified website and leverages the local Ollama AI to identify the URLs most relevant to understanding the company's core business. This could be useful for tasks like automatically generating a list of key pages (e.g., "About Us", "Careers") for a company brochure.

## Commit History

Here's a brief overview of the project's development history:

* **2025-04-10:**
    * **feat: Implement simple chat with local Ollama instance (simpleChat.py)**
        * Introduced the basic chat application, enabling interaction with a local Ollama model.

* **2025-04-11:**
    * **feat: Add intelligent link extraction for business description (inference_with_knowledge.py)**
        * Implemented the functionality to scrape website links and use AI to identify relevant URLs for describing a company's core business.

## Getting Started 

## Ensure you have Ollama installed and running locally (http://localhost:11434). 
## You also need to pip install other packages.
## Please use requirements.txt if you use pip

```markdown
```python
pip install -r requirements.txt

# Use environment.yml if you use conda
# For simpleChat.py:
```markdown
```python
python simpleChat.py

# For inference_with_knowledge.py:
```markdown
```python
python inference_with_knowledge.py [https://www.example-website.com](https://www.example-website.com)
