
# 🤖 GenAI Bot (Gemini 2.5 Pro + Gemma LLM)

A simple GenAI-powered chatbot built with **LangChain** that integrates both:
- **Gemini 2.5 Pro** (cloud-based LLM)
- **Gemma** (local LLM)

With **LangSmith** for monitoring and debugging.

---

## 🚀 Features
- Switch between **local** (Gemma) and **cloud** (Gemini) models
- Uses **LangChain** for orchestration
- **LangSmith** for monitoring and tracing
- Modular and easy to extend

---

## ⚡ Setup
```bash
# Clone the repo
git clone https://github.com/gowtham-dd/GenAi_BotGemma.git
cd GenAi_BotGemma

# Create virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

# Install dependencies
pip install -r requirements.txt
````

Create a `.env` file:

```
GOOGLE_API_KEY=your_gemini_api_key
LANGCHAIN_API_KEY=your_langsmith_key
```

---

## ▶️ Run

```bash
python app.py
```

---

## 📊 Monitoring

* All runs are logged and traced in **LangSmith Dashboard**.

---

## 📌 Tech Stack

* **LangChain**
* **LangSmith**
* **Gemini 2.5 Pro**
* **Gemma (local LLM)**

---

