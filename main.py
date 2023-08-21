import openai
from llama_index import SimpleDirectoryReader, GPTListIndex, VectorStoreIndex, LLMPredictor, PromptHelper, StorageContext, load_index_from_storage
from langchain.chat_models import ChatOpenAI
import tiktoken
import gradio as gr
import sys
import os
import json


os.environ["OPENAI_API_KEY"] = 'YOUR_API_KEY'
openai.api_key = os.environ["OPENAI_API_KEY"]
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")

class JSONDirectoryReader(SimpleDirectoryReader):
    def __init__(self, directory_path):
        self.directory_path = directory_path

    def load_data(self):
        data = {}
        for filename in os.listdir(self.directory_path):
            if filename.endswith('.json'):
                with open(os.path.join(self.directory_path, filename), 'r') as f:
                    documents = json.load(f)
                    if isinstance(documents, list):
                        for i, document in enumerate(documents):
                            if isinstance(document, dict):
                                # Store the document as a dictionary, not a string
                                data[filename + f'_{i}'] = document
                            else:
                                print(f"Warning: File {filename} contains a non-dictionary item at the top level.")
                    else:
                        print(f"Warning: File {filename} does not contain a list at the top level.")
        return data




def count_tokens(text):
    return len(list(encoding.encode(text)))

def clear_history():
    global conversation_history
    conversation_history = []
    return "History cleared."

def construct_index(directory_path):
    max_input_size = 4096
    num_outputs = 512
    max_chunk_overlap = 0.1
    chunk_size_limit = 600

    prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap, chunk_size_limit=chunk_size_limit)

    llm_predictor = LLMPredictor(llm=ChatOpenAI(temperature=0.1, model_name="gpt-4", max_tokens=num_outputs))

    documents = SimpleDirectoryReader(directory_path).load_data()

    index = VectorStoreIndex(documents, llm_predictor=llm_predictor, prompt_helper=prompt_helper)

    index.storage_context.persist(persist_dir="index")

    return index


# Initialize an empty conversation history
conversation_history = []

def chatbot(input_text):
    if(len(conversation_history) == 0):
        input_text = "You are a represantative of europ assistance Always reply with specific europ assistance packages like annual or temporary when asked about plans. Always give the plan that meets the minimum requirements at the lowest price. Ask questions if you need more information to reach a specific plan." + "Human:" + input_text
    else:
        input_text = "Human:" + input_text

    # Load the index
    storage_context = StorageContext.from_defaults(persist_dir='index')
    index = load_index_from_storage(storage_context)

    # Add the new user input to the conversation history
    conversation_history.append(input_text)

    # Combine the conversation history into a single string
    full_conversation = ' '.join(conversation_history)

    # Count the number of tokens in the full conversation
    num_tokens = count_tokens(full_conversation)

    # If the conversation is too long, remove the oldest messages until it fits
    while num_tokens > 4096:
        conversation_history.pop(0)
        full_conversation = ' '.join(conversation_history)
        num_tokens = count_tokens(full_conversation)

    # Query the model using the full conversation
    query_engine = index.as_query_engine()
    response = query_engine.query(full_conversation)

    # Add the model's response to the conversation history
    conversation_history.append("Your answer: " + response.response)

    return response.response



iface = gr.Interface(fn=chatbot,
                     inputs=gr.components.Textbox(lines=7, label="Enter your text"),
                     outputs="text",
                     title="Custom-trained AI Chatbot")

index = construct_index("docs")
iface.launch(share=True)
