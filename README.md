# Multi-Agent System for Text Processing

This repository contains a Python-based multi-agent system designed for text processing tasks, specifically summarizing and tutoring. The system utilizes a coordinator agent to route input to the appropriate specialized agent.

## Files

Here's a breakdown of the files in this repository:

* `coordinator.py`: This file contains the `CoordinatorAgent` class, which is responsible for determining whether a given text should be summarized or explained in detail (tutored). It uses a TensorFlow model to classify the input and route it to the appropriate agent.
* `dataset.py`: This file provides a predefined dataset of text snippets and their corresponding labels ("summarizer" or "tutor"). This dataset is used to train the coordinator agent's classification model.
* `gui.py`: This file implements a graphical user interface (GUI) using `tkinter` and `customtkinter`. The GUI provides a user-friendly chat interface for interacting with the multi-agent system.
* `main_app.py`: This is the main application file. It sets up the system, including the coordinator agent, and defines the summarizer and tutor agents. It also includes functions for generating text responses using a language model (Qwen).

## System Architecture

The system operates as follows:

1.  **Input:** The user provides a text prompt through the GUI.
2.  **Coordination:** The `CoordinatorAgent` analyzes the prompt to determine whether it requires summarization or tutoring.
3.  **Routing:** The coordinator routes the prompt to the appropriate agent (summarizer or tutor).
4.  **Processing:**
    * The summarizer agent generates a concise summary of the input text.
    * The tutor agent provides a detailed explanation of the input text.
5.  **Output:** The response from the selected agent is displayed in the GUI.

## Key Features

* Multi-agent architecture: The system employs multiple specialized agents to handle different tasks.
* Text classification: The coordinator agent uses a trained model to classify input text.
* GUI interface: The `customtkinter` GUI provides an intuitive way to interact with the system.
* Language model integration: The system uses the Qwen language model for text generation.

## Dependencies

The code requires the following dependencies:

* Python 3.x
* tensorflow
* sentence-transformers
* nltk
* customtkinter
* openai
* dotenv
* transformers
* torch
* (Optionally) torch-directml

## Setup and Installation

1.  Clone the repository:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
2.  Install the required dependencies.  It is recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    pip install -r requirements.txt # You'll need to create a requirements.txt file with the dependencies
    ```
3.  Download NLTK data (if you haven't already):
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    ```
4.  Set up your OpenAI API key:
    * Create a `.env` file in the repository's root directory.
    * Add your OpenAI API key to the `.env` file:
        ```
        OPENAI_API_KEY=your_openai_api_key
        ```
        *(Note:  The `api.py` file which contained the api_key is not present in the file list, so I've assumed it should be in a .env file.  If this is incorrect, you'll need to adjust the instructions.)*
5.  Run the main application:
    ```bash
    python main_app.py
    ```

## Usage

1.  Run the `main_app.py` script to start the application.
2.  The GUI will appear.
3.  Enter your text prompt in the input box.
4.  The system will process your prompt and display the response in the chat window.

## Notes

* The `coordinator.py` file uses a TensorFlow model for classifying input text.  The model is trained using the data in `dataset.py`.
* The `main_app.py` file includes a placeholder for a text generation function.  This function uses the Qwen language model.  You may need to configure this part with your own API key or model setup.
* The GUI in `gui.py` provides a simple chat-like interface for interacting with the system.

## Disclaimer

This project is for informational and educational purposes only.  The performance of the summarizer and tutor agents may vary depending on the input text and the configuration of the language model.
