# Connecting LLMs to External Data Sources

## Description

This repository provides a framework for connecting large language models (LLMs) to external data sources, enabling intelligent interaction and retrieval of information. It features a chatbot interface that allows users to query a dataset and receive relevant responses while also engaging in natural language conversation.

## Key Features

- **Dynamic Question Answering**: Utilizes advanced LLMs to process user queries and provide contextual answers based on the content of the connected dataset.
- **Natural Language Interaction**: Capable of understanding casual conversational prompts, allowing users to ask questions, express gratitude, or simply greet the bot.
- **Seamless Data Integration**: Designed to work with CSV datasets, transforming raw data into meaningful responses through text generation and information retrieval techniques.
- **Customizable and Extensible**: The code structure is modular, allowing for easy integration of additional data sources and language models.

## Requirements

To run this project, you need to install the following Python packages:

```bash
pip install streamlit transformers torch langchain chromadb streamlit-chat langchain-community
