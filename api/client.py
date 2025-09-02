import requests
import streamlit as st

def get_gemini_response(input_text):
    response = requests.post(
        "http://localhost:8000/essay",
        json={'topic': input_text}
    )
    return response.json().get('essay')

def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8000/poem",
        json={'topic': input_text}
    )
    return response.json().get('poem')

# Streamlit framework
st.title('Langchain Demo With Gemini & Ollama')

input_text = st.text_input("Write an essay on")
input_text1 = st.text_input("Write a poem on")

if input_text:
    st.write(get_gemini_response(input_text))

if input_text1:
    st.write(get_ollama_response(input_text1))
