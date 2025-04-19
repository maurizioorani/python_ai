import json
from typing import Dict, List
from openai import OpenAI
from dotenv import load_dotenv
import gradio as gr

# Load environment variables from .env file
load_dotenv()

def initialize_client(use_ollama: bool = False) -> OpenAI:
    return OpenAI(base_url="http://localhost:11434/v1", api_key="ollama") if use_ollama else OpenAI()

def create_initial_messages() -> List[Dict[str, str]]:
    return [{"role": "system", "content": "You are a helpful assistant."}]

def chat(user_input: str, messages: List[Dict[str, str]], client: OpenAI, model_name: str) -> str:
    messages.append({"role": "user", "content": user_input})
    try:
        response = client.chat.completions.create(model=model_name, messages=messages)
        assistant_response = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_response})
        return assistant_response
    except Exception as e:
        return f"Error with API: {str(e)}"

def summarize_messages(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    summary = "Summary: " + " ".join([m["content"][:50] + "..." for m in messages[-5:]])
    return [{"role": "system", "content": summary}] + messages[-5:]

def save_conversation(messages: List[Dict[str, str]], filename: str = "conversation.json"):
    with open(filename, "w") as f:
        json.dump(messages, f)

def load_conversation(filename: str = "conversation.json") -> List[Dict[str, str]]:
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return create_initial_messages()

# Gradio Chatbot logic
class GradioChatBot:
    def __init__(self, use_ollama: bool = False):
        self.use_ollama = use_ollama
        self.client = initialize_client(use_ollama)
        self.model_name = "llama3.2" if use_ollama else "gpt-4o-mini"
        self.messages = create_initial_messages()

    def respond(self, user_input, history):
        if user_input.strip().lower() == "save":
            save_conversation(self.messages)
            return history + [[user_input, "Conversation saved!"]]
        elif user_input.strip().lower() == "load":
            self.messages = load_conversation()
            return history + [[user_input, "Conversation loaded!"]]
        elif user_input.strip().lower() == "summary":
            self.messages = summarize_messages(self.messages)
            return history + [[user_input, "Conversation summarized!"]]
        else:
            response = chat(user_input, self.messages, self.client, self.model_name)
            history = history + [[user_input, response]]
            if len(self.messages) > 10:
                self.messages = summarize_messages(self.messages)
                history = history + [["", "(Conversation automatically summarized)"]]
            return history

def launch_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("# Chatbot with History (OpenAI/Ollama)")
        model_selector = gr.Radio(
            choices=["OpenAI", "Ollama"], 
            value="OpenAI", 
            label="Select Model"
        )
        chatbot = gr.Chatbot()
        msg = gr.Textbox(label="Your message")
        clear = gr.Button("Clear")

        # Store the bot instance in a state variable
        state = gr.State()

        def start_chatbot(selected_model):
            use_ollama = selected_model == "Ollama"
            return GradioChatBot(use_ollama)

        def user_fn(user_input, history, bot):
            return "", bot.respond(user_input, history), bot

        # When model is selected, create a new bot instance
        model_selector.change(
            start_chatbot, 
            inputs=model_selector, 
            outputs=state
        )

        # On submit, use the current bot instance
        msg.submit(
            user_fn, 
            [msg, chatbot, state], 
            [msg, chatbot, state]
        )
        clear.click(lambda: None, None, chatbot, queue=False)

        # Initialize with default bot
        demo.load(lambda: GradioChatBot(False), None, state)

    demo.launch()

if __name__ == "__main__":
    launch_gradio()

if __name__ == "__main__":
    # Set use_ollama to True to use Ollama, or False for OpenAI
    launch_gradio(use_ollama=False)