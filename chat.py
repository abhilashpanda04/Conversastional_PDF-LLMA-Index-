import pandas as pd
import numpy as np
from llama_index import VectorStoreIndex, SimpleDirectoryReader

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from llama_index import LangchainEmbedding, ServiceContext

from llama_index.llms import HuggingFaceLLM
from llama_index.prompts.prompts import SimpleInputPrompt

from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers import pipeline
import torch
from llama_index import Prompt

import streamlit as st

chat_history=[]

st.header("Chat-bot-template")

def handle_user_query(query,chat_engine):


    response = chat_engine.chat(query)
    print(response)
    st.session_state.chat_history.append(query)
    st.session_state.chat_history.append(response)

    for i , message in enumerate(st.session_state.chat_history):
        if i%2==0:
            with st.chat_message('user'):
                st.write(message)
        else:
            with st.chat_message('assistant'):
                st.write(message.response)
    


def main():

    if "conversastion" not in st.session_state:
        st.session_state.conversastion=None

    query=st.chat_input('Ask a question based on the PDF')

    if query:
        template = (
        "You are a financial Analyst assistant.\n"
        "We have provided context information below. \n"
        "---------------------\n"
        "{context_str}"
        "\n---------------------\n"
        "Given this information, please answer the question: {query_str}\n"
        )
        qa_template = Prompt(template)

        chat_engine = st.session_state.conversastion.as_chat_engine(
        chat_mode='condense_question', 
        verbose=True,
        text_qa_template=qa_template
        )
        handle_user_query(query,chat_engine)
        if st.button('clear this chat'):
            chat_engine.reset()
            st.session_state.chat_history=None



    with st.sidebar:
        st.file_uploader("Upload a file", type=["csv",'pdf'],accept_multiple_files=True)
        submit=st.button("submit")
        if not submit:
            st.stop()
        if submit:
                with st.spinner("processing...."):
                    query_wrapper_prompt = SimpleInputPrompt(
                        "Below is an instruction that describes a task. "
                        "Write a response that appropriately completes the request.\n\n"
                        "### Instruction:\n{query_str}\n\n### Response:"
                    )


                    tokenizer = T5Tokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")
                    from langchain import HuggingFacePipeline

                    llm_obj = HuggingFacePipeline.from_model_id(
                        model_id="MBZUAI/LaMini-Flan-T5-248M",
                        task="text2text-generation",
                        model_kwargs={"temperature": 0, "max_length": 300},
                    )
                    service_context = ServiceContext.from_defaults(
                    embed_model="local:sentence-transformers/all-mpnet-base-v2",
                    llm=llm_obj
                    )

                    documents = SimpleDirectoryReader(input_files=[r'/Users/abhi/Desktop/abhilash/development/sample-pdf/10k.pdf']).load_data()
                    st.session_state.conversastion = VectorStoreIndex.from_documents(documents,show_progress=True,service_context=service_context)
                    st.session_state.chat_history=[]

if __name__=="__main__":
    main()  


# query_engine = index.as_retriever(service_context=service_context,streaming=True, similarity_top_k=1)
# response = query_engine.retrieve("What is the Gross margin for products in $ ?")

# query_engine = index.as_query_engine()


# response = query_engine.query("What is the the value for  Right-of-use assets for 2022 ?")

