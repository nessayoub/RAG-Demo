# RAG-Demo
## Menu Query System using RAG Model

This repository contains code for a Menu Query System implemented using the RAG (Retrieval-Augmented Generation) model. The system allows users to inquire about menu items and receive relevant responses based on the provided menu data. It utilizes pre-trained models for efficient entity retrieval and response generation.

## Features

- **Efficient Query Handling**: Utilizes FAISS index for fast approximate nearest neighbor search to handle user queries efficiently.
- **RAG Model Integration**: Integrates Facebook's RAG model for response generation based on retrieved entities.
- **Menu Data Management**: Loads menu data from a JSON file and generates embeddings for menu items using a Sentence Transformers model.
- **User Interaction**: Provides a simple command-line interface for users to interact with the system.

## Requirements

- Python 3.x
- [torch](https://pypi.org/project/torch/)
- [faiss](https://pypi.org/project/faiss/)
- [sentence-transformers](https://pypi.org/project/sentence-transformers/)
- [transformers](https://pypi.org/project/transformers/)

## Usage

1. **Installation**:
   - Clone this repository:
     ```
     git clone https://github.com/nessayoub/RAG-Demo.git
     cd menu-query-system
     ```

2. **Running the System**:
   - Run the main script:
     ```
     python menu_query_system.py
     ```
   - Enter your query when prompted. Type 'quit' to exit the system.

## Optimization and Scalability

- **Prompt Generation Time**: The system ensures prompt generation time is less than 50ms by optimizing embedding generation, utilizing efficient indexing, and streamlining retrieval and response generation.
- **Scalability**: The system can handle large menu datasets efficiently by leveraging FAISS index for fast retrieval and optimizing data processing pipelines.
