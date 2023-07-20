# Conversastional_PDF-LLMA-Index-


This file repository the code for a chatbot template. The chatbot interacts with users, handles user queries, and performs various operations based on user inputs and file uploads.
this bot is based on local llm.

# Dependencies
The following dependencies are required for this script to run:

pandas
numpy
llama_index
langchain
transformers
torch
streamlit

# Functions
handle_user_query(query, chat_engine)
This function handles user queries and generates responses using the provided chat engine.

# Parameters:
query: The user query to be processed.
chat_engine: The chat engine object responsible for generating responses.

main()
This function serves as the main entry point for the program.
It handles user interactions and file uploads.


Usage
To run the chatbot, execute the following command:

📋 Copy code
```
streamlit run chat.py
```
Note: Make sure you have all the required dependencies installed before running the script.
