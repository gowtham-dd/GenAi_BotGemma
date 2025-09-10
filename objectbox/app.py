import streamlit as st
import os 
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv
#local llama3 
from langchain_community.chat_models import ChatOllama
import time

load_dotenv()

# Load API keys
groq_api_key = os.environ['GROQ_API_KEY']
google_api_key = os.environ['GOOGLE_API_KEY']

llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
# llm = ChatOllama(
#     model="llama3",   # make sure llama3 is pulled via `ollama pull llama3`
#     temperature=0.1
# )
prompt = ChatPromptTemplate.from_template("""
Answer the question based on the provided content. 
<context> {context} </context>
Question: {input}
""")

# Vector Embeddings in Object  Box DB

def vector_embedding():
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=google_api_key
        )
        # ✅ Use PyPDFLoader for a single PDF
    st.session_state.loader = PyPDFDirectoryLoader(r"D:\\Data Science\\LangChain\\GenAi_BotGemma\\groq\\pdfs\\sample.pdf")

    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs
        )
    st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings,
            embedding_dimensiona=768
        )


prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()
    if "vectors" in st.session_state:
        st.success("✅ Vector Store DB Ready")

if prompt1 and "vectors" in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    start = time.process_time()
    response = retrieval_chain.invoke({'input': prompt1})
    st.write("⏱ Response time:", time.process_time() - start)
    st.write("### Answer:", response['answer'])

    with st.expander('Document Similarity Search'):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------------")
