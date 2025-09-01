from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM

import streamlit as st


import os
from dotenv import load_dotenv


# Load .env file
load_dotenv()

# LangSmith tracking (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])

# Streamlit UI
st.title('LangChain Demo with Local Gemma (Ollama)')
input_text = st.text_input("Search the topic you want")

# ✅ Connect to local Ollama server
# Make sure you have started Ollama in background:  ollama serve
# And pulled the model once: ollama pull gemma
llm = OllamaLLM(
    model="gemma",         # Model name in Ollama (check with `ollama list`)
    base_url="http://localhost:11434"  # Ollama server URL
)

output_parser = StrOutputParser()
chain = prompt | llm | output_parser

if input_text:
    response = chain.invoke({"question": input_text})
    st.write("### Answer:")
    st.write(response)
