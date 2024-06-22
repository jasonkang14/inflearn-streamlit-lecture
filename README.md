# Inflearn Streamlit Project

## Overview

This repository contains a project that utilizes LangChain and Streamlit to build a Retrieval Augmented Generation (RAG) application. The primary focus of this application is to provide insights and answers based on the South Korean Income Tax Law (소득세법). By leveraging advanced NLP techniques, this application enhances its responses using a combination of chat history and few-shot learning templates.

## Features

- **LangChain Integration**: Utilizes LangChain to manage and interact with language models effectively.
- **Streamlit Interface**: A user-friendly web interface created with Streamlit for seamless interaction.
- **Retrieval Augmented Generation (RAG)**: Combines retrieval-based techniques with generative models to produce accurate and context-aware answers.
- **Knowledge Base**: Focuses on the South Korean Income Tax Law (소득세법) as the primary knowledge base.
- **Chat History**: Maintains a history of user interactions to provide contextually relevant answers.
- **Few-Shot Learning Templates**: Enhances the model's responses by using predefined templates for better accuracy and consistency.

## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/jasonkang14/inflearn-streamlit.git
    cd inflearn-streamlit
    ```

2. Create and activate a virtual environment:
    ```sh
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```sh
    streamlit run chat.py
    ```

2. Open your web browser and navigate to the displayed local URL to interact with the application.

## Project Structure

- `chat.py`: Main application script that runs the Streamlit interface.
- `llm.py`: Contains utility functions for handling the knowledge base and model interactions.
- `config.py`: File with few-shot learning templates used to generate answers.

## How It Works

1. **Data Retrieval**: The application retrieves relevant sections of the South Korean Income Tax Law based on user queries.
2. **Contextual Processing**: Utilizes chat history to maintain context across multiple interactions.
3. **Template-Based Generation**: Applies few-shot learning templates to enhance the accuracy and relevance of the generated answers.
4. **User Interface**: Provides an intuitive web interface through Streamlit for users to interact with the application seamlessly.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue to discuss improvements, bug fixes, or new features.

## Acknowledgments

- [LangChain](https://langchain.com/)
- [Streamlit](https://streamlit.io/)
- All contributors and users of the project.

---

Feel free to modify and enhance this README to better fit your project's specifics and any additional information you may want to provide.
