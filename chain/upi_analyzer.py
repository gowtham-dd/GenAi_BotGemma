import os
import streamlit as st
from dotenv import load_dotenv

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# ===============================
# Stage 1: Setup
# ===============================
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=api_key,
    temperature=0.2
)

# Gemini Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.
Think step by step before providing a detailed answer.

<context>
{context}
</context>

Question: {input}
""")


# ===============================
# Stage 2: Streamlit UI
# ===============================
st.set_page_config(page_title="📄 Gemini PDF Q&A", layout="wide")
st.title("📄 Ask Questions from Your PDF (Gemini + FAISS)")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None:
    # Save uploaded PDF temporarily
    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ PDF uploaded successfully!")

    # ===============================
    # Stage 3: Load & Split PDF
    # ===============================
    loader = PyPDFLoader("uploaded.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(docs)

    st.write(f"📑 PDF split into {len(documents)} chunks.")

    # ===============================
    # Stage 4: Embedding & Vector Store
    # ===============================
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()

    # ===============================
    # Stage 5: Retrieval Chain
    # ===============================
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # ===============================
    # Stage 6: User Query
    # ===============================
    query = st.text_input("Ask a question about the PDF:")

    if query:
        with st.spinner("🤔 Thinking..."):
            response = retrieval_chain.invoke({"input": query})
            st.subheader("📌 Answer")
            st.write(response["answer"])

            # Optional: show retrieved context
            with st.expander("🔍 Retrieved Chunks (for transparency)"):
                for i, doc in enumerate(response["context"]):
                    st.markdown(f"**Chunk {i+1}:** {doc.page_content[:300]}...")

