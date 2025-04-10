# imports

import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display

# Costanti

OLLAMA_API = "http://localhost:11434/api/chat"
HEADERS = {"Content-Type": "application/json"}
MODEL = "deepseek-r1:14b"

# List per messaggi
# Inizializza la lista dei messaggi

messages = [
    {"role": "user", "content": "Fammi un elenco di applicazioni in cui puó essere utilizzata l'IA nel business italiano."},
]

# Utilizzo sempre la libreria OpenAI per invocare il modello locale di Ollama

from openai import OpenAI
ollama_via_openai = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

response = ollama_via_openai.chat.completions.create(
    model=MODEL,
    messages=messages
)

print(response.choices[0].message.content)