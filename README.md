# 🧠 Conversational RAG Chatbot (PDF Q&A)

An intelligent chatbot that allows you to upload any PDF file and ask natural language questions about its content — with context-aware, multi-turn conversations powered by RAG (Retrieval-Augmented Generation).

---

## 🚀 Live Demo

👉 [Try it on Streamlit](https://conversational-rag-chatbot-with-pdf-uploads.streamlit.app/)

---

## ✨ Features

- 📄 Upload any PDF file
- 🔍 Semantic search via `HuggingFace` embeddings + `FAISS` vectorstore 
- 🧠 Conversational memory using LangChain `RunnableWithMessageHistory`
- 🤖 Answer generation using Groq-hosted LLM (`gemma2-9b-it`)
- 🔐 Secrets managed via `st.secrets` for safe deployment
- ⚡ Fast & lightweight Streamlit UI

---

## 🛠 Tech Stack

| Component        | Tech Used                    |
|------------------|------------------------------|
| Language Model   | [Groq LLMs](https://console.groq.com) |
| Framework        | [Langchain](https://www.langchain.com) |
| Embeddings       | Hugging Face (`all-MiniLM-L6-v2`) |
| Vector Store     | FAISS                        |
| UI Framework     | Streamlit                    |
| Hosting          | Streamlit Cloud              |

---

## 📦 Installation

> git clone https://github.com/Adityajain8595/Conversational-RAG-Chatbot.git

> cd Conversational-RAG-Chatbot

> pip install -r requirements.txt

> streamlit run chatbotApp.py

---

## 🔐 Secrets Management

- GROQ_API_KEY = "your-groq-api-key"
- HF_TOKEN = "your-huggingface-token"
- LANGCHAIN_API_KEY = "your-langchain-key"
- LANGCHAIN_PROJECT = "Conversational RAG Chatbot"

For Streamlit Cloud, add these in Advanced Settings of the app on Streamlit Cloud while deploying.

---

## 🤝 Author

Made by Aditya Jain

Have suggestions or improvements? Open an issue or a pull request!
Let’s build better AI apps, together. 🔥

Connect with me:

LinkedIn: [LinkedIn URL](https://www.linkedin.com/in/adityajain8595/) 
