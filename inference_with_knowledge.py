# imports
import requests
from bs4 import BeautifulSoup
from IPython.display import Markdown, display
from openai import OpenAI

# Constants
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_API_CHAT_URL = f"{OLLAMA_BASE_URL}/api/chat"
OLLAMA_API_V1_URL = f"{OLLAMA_BASE_URL}/v1"
HEADERS = {"Content-Type": "application/json"}
MODEL_NAME = "llama3.2"
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/117.0.0.0 Safari/537.36"

# Initialize OpenAI client for Ollama
ollama_client = OpenAI(base_url=OLLAMA_API_V1_URL, api_key='ollama')

# Initial Ollama interaction
initial_messages = [{"role": "user", "content": "Describe some of the business applications of Generative AI"}]

try:
    initial_response = ollama_client.chat.completions.create(
        model=MODEL_NAME,
        messages=initial_messages
    )
    print("Initial Ollama Response:")
    print(initial_response.choices[0].message.content)
except Exception as e:
    print(f"Error during initial Ollama call: {e}")

# Class to represent a Webpage
class Website:
    """
    A utility class to represent a Website that we have scraped, now with links
    """

    def __init__(self, url):
        self.url = url
        self.body = None
        self.soup = None
        self.title = "No title found"
        self.text = ""
        self.links = []
        self._fetch_and_parse()

    def _fetch_and_parse(self):
        try:
            response = requests.get(self.url, headers={"User-Agent": USER_AGENT})
            response.raise_for_status()  # Raise an exception for bad status codes
            self.body = response.content
            self.soup = BeautifulSoup(self.body, 'html.parser')
            self._extract_title()
            self._extract_text()
            self._extract_links()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching URL '{self.url}': {e}")
        except Exception as e:
            print(f"Error parsing URL '{self.url}': {e}")

    def _extract_title(self):
        if self.soup and self.soup.title and self.soup.title.string:
            self.title = self.soup.title.string

    def _extract_text(self):
        if self.soup and self.soup.body:
            for irrelevant in self.soup.body(["script", "style", "img", "input"]):
                irrelevant.decompose()
            self.text = self.soup.body.get_text(separator="\n", strip=True)

    def _extract_links(self):
        if self.soup:
            links = [link.get('href') for link in self.soup.find_all('a')]
            self.links = [link for link in links if link]

    def get_contents(self):
        return f"Webpage Title:\n{self.title}\nWebpage Contents:\n{self.text}\n\n"

# Prompts for link extraction
LINK_SYSTEM_PROMPT = """You are provided with a list of links found on a webpage.
You are able to decide which of the links would be most relevant to include for a summary of the core business about the company,
such as links to an About page, or a Company page, or Careers/Jobs pages.
You should respond in JSON as in this example:
{
    "links": [
        {"type": "about page", "url": "https://full.url/goes/here/about"},
        {"type": "careers page", "url": "https://another.full.url/careers"}
    ]
}
"""
print("Link System Prompt:")
print(LINK_SYSTEM_PROMPT)

def get_links_user_prompt(website):
    user_prompt = f"Here is the list of links on the website of {website.url} - "
    user_prompt += "please decide which of these are relevant web links for a summary of the core business about the company, respond with the full https URL in JSON format. "
    user_prompt += "Do not include Terms of Service, Privacy, email links.\n"
    user_prompt += "Links (some might be relative links):\n"
    user_prompt += "\n".join(website.links)
    return user_prompt

# Example usage:
civitai_website = Website("https://civitai.com/")

print("\nUser Prompt for Link Extraction:")
user_prompt_links = get_links_user_prompt(civitai_website)
print(user_prompt_links)

try:
    links_response = ollama_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": LINK_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt_links}
        ]
    )
    print("\nOllama Response for Relevant Links:")
    print(links_response.choices[0].message.content)
except Exception as e:
    print(f"Error during link extraction Ollama call: {e}")
