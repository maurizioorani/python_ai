import json
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI client
def initialize_client(use_ollama: bool = False) -> OpenAI:
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama") if use_ollama else OpenAI()

def create_initial_messages() -> List[Dict[str, str]]:
    return [{"role": "system", "content": "You are a helpful assistant."}]

# Function to handle chat interactions
def chat(user_input: str, messages: List[Dict[str, str]], client: OpenAI, model_name: str) -> str:
    messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
        assistant_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_response})
        return assistant_response
    except Exception as e:
        return f"Error with API: {str(e)}"

# Function to summarize the last 5 messages
def summarize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    summary = "Summary: " + " ".join([m["content"][:50] + "..." for m in messages[-5:]])
    return [{"role": "system", "content": summary}] + messages[-5:]

# Function to save and load conversation history
def save_conversation(messages: List[Dict[str, str]], filename: str = "conversation.json"):
    with open(filename, "w") as f:
        json.dump(messages, f)

# Function to load conversation history
def load_conversation(filename: str = "conversation.json") -> List[Dict[str, str]]:
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No conversation file found at {filename}")
        return create_initial_messages()

# Function to handle user input and chat loop
def main():
    print("Select model: 1. OpenAI GPT-4  2. Ollama (Local)")
    use_ollama = input("Enter choice (1 or 2): ") == "2"
    client = initialize_client(use_ollama)
    model_name = "llama3.2" if use_ollama else "gpt-4o-mini"
    messages = create_initial_messages()

    print(f"\nUsing {'Ollama' if use_ollama else 'OpenAI'} model. Type 'quit' to exit.")
    print("Commands: save | load | summary")

    while True:
        user_input = input("\nYou: ").strip().lower()
        if user_input == "quit":
            break
        elif user_input == "save":
            save_conversation(messages)
            print("Conversation saved!")
        elif user_input == "load":
            messages = load_conversation()
            print("Conversation loaded!")
        elif user_input == "summary":
            messages = summarize_messages(messages)
            print("Conversation summarized!")
        else:
            response = chat(user_input, messages, client, model_name)
            print(f"\nAssistant: {response}")
            if len(messages) > 10:
                messages = summarize_messages(messages)
                print("\n(Conversation automatically summarized)")

if __name__ == "__main__":
    main()