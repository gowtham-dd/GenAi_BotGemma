import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

#groq 
groq_api_key=os.environ['GROQ_API_KEY']
google_api_key=os.environ['GOOGLE_API_KEY']


if "vector" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001",google_api_key=google_api_key)
    st.session_state.loader=WebBaseLoader("https://en.wikipedia.org/wiki/Attention_Is_All_You_Need")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings)

st.title("ChatGroq Demo")
llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it")

prompt=ChatPromptTemplate.from_template(
    """
Anser the question based on the provided context only.
<context>{context}<context>
Question:{input}

"""
)

document_chain=create_stuff_documents_chain(llm,prompt)
retriever=st.session_state.vectors.as_retriever()
retriever_chain=create_retrieval_chain(retriever,document_chain)

prompt=st.text_input("Question:")

if prompt:
    start=time.process_time()
    response=retriever_chain.invoke({"input":prompt})
    print("Response Time:",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("----------------------------")