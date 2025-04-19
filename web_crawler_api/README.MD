# Document Crawler and Query API

This project provides an API to crawl web documentation, process it into chunks, store it in a vector database (ChromaDB), and query for similar content. It utilizes asynchronous operations for efficient crawling and embedding generation.

## Table of Contents

- [Purpose](#purpose)
- [Dependencies](#dependencies)
- [Setup](#setup)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
  - [`POST /crawl`](#post-crawl)
  - [`GET /query`](#get-query)
- [Usage](#usage)
  - [Crawling URLs](#crawling-urls)
  - [Querying Documents](#querying-documents)
- [How it Works](#how-it-works)
- [Environment Variables](#environment-variables)

## Purpose

This API is useful for:

- **Indexing documentation:** Automatically crawl and store content from multiple web sources.
- **Knowledge retrieval:** Quickly find relevant sections within the indexed documentation based on a query.
- **Building RAG applications:** Serve as a backend for Retrieval-Augmented Generation systems that need to access and reason over a body of documentation.

## Dependencies

The following Python libraries are required to run this project. You can install them using pip:

```bash
pip install -r requirements.txt
```

The requirements.txt file should contain:

```bash
python-dotenv
requests
chromadb
xml.etree.ElementTree
fastapi
uvicorn
pydantic
crawl4ai
openai
```

## Setup
Clone the repository (if you haven't already).
Install dependencies: Navigate to the project directory in your terminal and run the command above.
Set up environment variables:
Create a .env file in the project root.
Add your OpenAI API key:
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-4o-mini # Or another suitable OpenAI model
You can optionally change the LLM_MODEL to your preferred OpenAI language model.

## Running the API
To start the FastAPI application, run the following command in your terminal:

```bash
python main.py
```

This will start the API server, typically accessible at http://127.0.0.1:8000.

```bash
API Endpoints
POST /crawl
```

Crawls a list of provided URLs and stores their content in the ChromaDB.

Request Body:


```json

{
  "urls": ["[https://example.com/doc1](https://example.com/doc1)", "[https://example.org/guide](https://example.org/guide)"],
  "max_concurrent": 5
}

```

urls: A list of HTTP or HTTPS URLs to crawl (required).
max_concurrent: An optional integer specifying the maximum number of concurrent crawling tasks (default: 5).

Response:


```json
{
  "status": "success",
  "message": "Processed 2 URLs"
}
In case of an error, it will return an HTTP 500 status code with a JSON response containing the error details.
```

GET /query
Queries the ChromaDB for document chunks that are semantically similar to the provided query.

Query Parameters:

q: The query string (required).
n_results: An optional integer specifying the number of most similar results to return (default: 5).
Response:

```json

{
  "ids": [["[https://example.com/doc1_0](https://example.com/doc1_0)", "[https://example.org/guide_2](https://example.org/guide_2)", ...]],
  "distances": [[0.123, 0.156, ...]],
  "documents": [["Content of chunk 1...", "Content of chunk 2...", ...]],
  "metadatas": [[{"url": "...", "chunk_number": 0, ...}, {"url": "...", "chunk_number": 2, ...}, ...]]
}

```

In case of an error, it will return an HTTP 500 status code with a JSON response containing the error details.

## Usage
Crawling URLs
You can send a POST request to the /crawl endpoint with a JSON payload containing the URLs you want to crawl. For example, using curl:

```bash

curl -X POST -H "Content-Type: application/json" -d '{"urls": ["[https://ai.pydantic.dev/latest/](https://ai.pydantic.dev/latest/)", "[https://ai.pydantic.dev/blog/](https://ai.pydantic.dev/blog/)"]}' [http://127.0.0.1:8000/crawl](http://127.0.0.1:8000/crawl)
You can also specify the concurrency:

```

curl -X POST -H "Content-Type: application/json" -d '{"urls": ["[https://example.com/doc](https://example.com/doc)"], "max_concurrent": 10}' [http://127.0.0.1:8000/crawl](http://127.0.0.1:8000/crawl)
Querying Documents
You can send a GET request to the /query endpoint with the q parameter containing your search query. For example, using curl:

```bash

curl "[http://127.0.0.1:8000/query?q=What](http://127.0.0.1:8000/query?q=What) is Pydantic AI?"
You can also specify the number of results:

```

curl "[http://127.0.0.1:8000/query?q=How](http://127.0.0.1:8000/query?q=How) to use the crawler&n_results=3"

## How it Works
Crawling: The /crawl endpoint receives a list of URLs. It uses the crawl4ai library with Playwright to asynchronously fetch the content of these web pages and extract their Markdown representation.
Chunking: The extracted Markdown content is split into smaller, manageable chunks using a function that tries to respect code block and paragraph boundaries.
Title and Summary Generation: For each chunk, the OpenAI API (specifically the model specified in .env) is used to generate a concise title and summary, providing better context for retrieval.
Embedding Generation: The OpenAI API is also used to generate an embedding vector for each text chunk. These embeddings capture the semantic meaning of the text.
Storage: Each processed chunk, along with its metadata (source URL, chunk number, title, summary, crawl timestamp, and URL path) and its embedding vector, is stored in a ChromaDB instance. ChromaDB is a vector database that allows for efficient similarity search.
Querying: The /query endpoint receives a query string. It generates an embedding for this query using the same OpenAI embedding model. Then, it performs a similarity search in ChromaDB to find the most relevant document chunks based on the cosine distance between their embeddings and the query embedding. The results, including the content, metadata, and distance scores, are returned.
Environment Variables
OPENAI_API_KEY: Your API key for accessing OpenAI services (required).
LLM_MODEL: The OpenAI language model to use for generating titles and summaries (default: gpt-4o-mini). You can choose other models like gpt-3.5-turbo or gpt-4.
