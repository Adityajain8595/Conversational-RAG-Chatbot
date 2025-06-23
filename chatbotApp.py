import streamlit as st
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN")

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

embeddings = load_embeddings()

# Streamlit app setup
st.title(os.environ["LANGCHAIN_PROJECT"])
st.write("Upload a PDF document to start chatting with it.")

# Building model and managing chat history
llm = ChatGroq(model="gemma2-9b-it", groq_api_key=os.environ["GROQ_API_KEY"])
session_id = st.text_input("Session ID", value="default_session")

if "store" not in st.session_state:
    st.session_state.store = {}

uploaded_files = st.file_uploader("Upload a PDF file", type="pdf", accept_multiple_files=True)
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temppdf = f"./temp.pdf"
        with open(temppdf, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        loader = PyPDFLoader(temppdf)
        docs = loader.load()
        documents.extend(docs)
        os.remove(temppdf)  
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    contextualize_system_prompt = (
        "Given a chat history and the latest user question"
        " which might reference context in the chat history,"
        " formulate a standalone question which can be understood"
        " without the chat history. Do not answer the question,"
        " just reformulate it if needed and otherwise return it as it is."
    )

    contextualize_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_prompt)

    # Answer question prompt
    system_prompt = (
        "You are an assistant for question-answering tasks."
        "Use the following pieces of retrieval context to answer the question."
        "If you don't know the answer, just say that you don't know."
        "Use three sentences maximum and keep the answer concisely."
        "\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    def get_session_history(session_id: str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key = "input",
        history_messages_key = "chat_history",
        output_messages_key = "answer"
    )

    user_input = st.text_input("Ask a question about the document:")
    if user_input:
        with st.spinner("Generating answer..."):
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input":user_input},
                config = {"configurable" : {"session_id": session_id}}
            )
            
            st.write("Answer: ", response["answer"])
            st.sidebar.write(st.session_state.store)
            st.sidebar.write("Chat History: ", session_history.messages)
