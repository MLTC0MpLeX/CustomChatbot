# Custom-trained AI Chatbot

This is a Python script for a custom-trained AI chatbot powered by OpenAI's GPT-3.5 Turbo model. The chatbot is designed to interact with users, provide responses, and manage a conversation history. Below is a brief overview of the code and how to use it.

## Prerequisites

Before using this code, make sure you have the following prerequisites installed:

- Python (>=3.6)
- Required Python packages (`openai`, `llama_index`, `tiktoken`, `gradio`, `sys`, `os`, `json`)

You also need to set your OpenAI API key as an environment variable by adding the following line to the code:

```python
os.environ["OPENAI_API_KEY"] = 'your_openai_api_key_here'
```

Replace `'your_openai_api_key_here'` with your actual OpenAI API key.

## Code Overview

### Imports
The script starts by importing necessary Python packages and libraries, including OpenAI's Python library, llama_index, tiktoken, gradio, sys, os, and json.

### JSONDirectoryReader
This class is a custom data reader that loads JSON data from a specified directory.

### count_tokens
A utility function that counts the number of tokens in a given text.

### clear_history
A function to clear the conversation history.

### construct_index
This function constructs an index for the chatbot. It loads documents from a directory and uses them to build an index for quick responses.

### chatbot
The core chatbot function. It takes user input, maintains a conversation history, and interacts with the chatbot model to generate responses.

### gr.Interface
This code sets up a Gradio interface for the chatbot, allowing users to interact with it through a web-based interface.

### Running the Chatbot
The script constructs the index from documents in the "docs" directory and launches the Gradio interface, making the chatbot accessible via a web page.

## Usage

1. Ensure you have the necessary prerequisites installed and have set your OpenAI API key.

2. Place your JSON documents in a directory named "docs" in the same directory as this script. Each JSON file should contain a list of conversational documents.

3. Run the script, and the chatbot will be accessible through a web interface.

4. Enter text in the provided textbox, and the chatbot will respond based on the conversation history and the loaded documents.

## Note


- The chatbot's behavior and responses can be customized by adjusting the code in the `chatbot` function and the documents in the "docs" directory.

- This code uses the Gradio library to create a simple web interface for the chatbot. You can customize the interface further as needed.

- The chatbot may have token limits, so ensure that the conversation and input text do not exceed the model's maximum token limit (4096 tokens for GPT-3.5 Turbo).

