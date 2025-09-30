from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
import os

load_dotenv()

## To load model 
def get_model(question: str):
    groq_api_key = os.environ['GROQ_API_KEY']
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="gemma2-9b-it",
        temperature=0.6
    )
    # Use .invoke instead of calling llm()
    response = llm.invoke(question)
    return response.content   # <-- use .content to get the text only


st.set_page_config(page_title="Q&A Demo")
st.header("Langchain Application")

input_text = st.text_input("Input : ", key="input")   # fixed key issue

submit = st.button("Generate")

if submit and input_text:
    response = get_model(input_text)
    st.subheader("The Response is")
    st.write(response)
