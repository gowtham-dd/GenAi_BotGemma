from fastapi import FastAPI, HTTPException
from langchain.prompts import ChatPromptTemplate
import uvicorn
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Import Gemini + Ollama models
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import OllamaLLM

# Load .env file
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Define Pydantic models
class EssayRequest(BaseModel):
    topic: str

class PoemRequest(BaseModel):
    topic: str

class GeminiRequest(BaseModel):
    message: str

# Define FastAPI app
app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="LangChain Server with Gemini & Ollama"
)

# Define Models
gemini_model = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",   # ✅ use supported Gemini model
    google_api_key=api_key,
    temperature=0.7
)

ollama_model = OllamaLLM(
    model="gemma",  # ✅ local Ollama model
    base_url="http://localhost:11434",
    temperature=0.7
)

# Define prompts ✅
essay_prompt = ChatPromptTemplate.from_template(
    "Write me an essay about {topic} with around 100 words."
)

poem_prompt = ChatPromptTemplate.from_template(
    "Write me a fun poem about {topic} for a 5-year-old child with ~100 words."
)

# Manual endpoints
@app.post("/gemini")
async def invoke_gemini(request: GeminiRequest):
    try:
        response = gemini_model.invoke(request.message)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini error: {str(e)}")

@app.post("/essay")
async def generate_essay(request: EssayRequest):
    try:
        chain = essay_prompt | gemini_model
        response = chain.invoke({"topic": request.topic})
        return {"essay": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Essay generation error: {str(e)}")

@app.post("/poem")
async def generate_poem(request: PoemRequest):
    try:
        chain = poem_prompt | ollama_model
        response = chain.invoke({"topic": request.topic})
        return {"poem": response}   # ✅ return directly, it's already a string
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Poem generation error: {str(e)}")

# Health check
@app.get("/")
async def root():
    return {"message": "LangChain Server is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
