import os
import tempfile
import shutil
import streamlit as st
import sys
import atexit
import gc

# ---- PATCH for ChromaDB: Fix sqlite3 version ----
os.environ["PYSQLITE3_INCLUDE_PATH"] = "/home/adminuser/venv/lib/python3.10/site-packages/pysqlite3"
import pysqlite3
sys.modules["sqlite3"] = pysqlite3

from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, SerpAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain_core.pydantic_v1 import BaseModel
from langchain.schema import Document
from typing import List
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict

# Load environment variables
groq_api_key = st.secrets.get("GROQ_API_KEY", None)
serp_api_key = st.secrets.get("SERP_API_KEY", None)

if not groq_api_key or not serp_api_key:
    st.error("Missing required API keys. Please check your environment variables.")
    st.stop()

# Temporary Chroma DB Directory
chroma_dir = tempfile.mkdtemp()
atexit.register(lambda: shutil.rmtree(chroma_dir, ignore_errors=True))

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.sidebar.title("Options")

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
input_url = st.sidebar.text_input("Paste a URL")

st.title("Multi Agent Chatbot (Wiki, Arxiv & Google)")
question = st.chat_input("Ask your question")

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu", "batch_size": 16}
)

# Load documents
docs_list = []
retriever = None

if uploaded_file:
    with st.spinner("Indexing PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        docs_list = loader.load()
        os.remove(tmp_file_path)

elif input_url:
    with st.spinner("Fetching text from URL..."):
        loader = WebBaseLoader(input_url)
        docs_list = loader.load()

if (uploaded_file or input_url) and (not docs_list or all(len(doc.page_content.strip()) == 0 for doc in docs_list)):
    st.error("No valid text found. Please try another file or URL.")
    st.stop()

if docs_list:
    with st.spinner("Splitting and indexing document..."):
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=300,
            chunk_overlap=20
        )
        doc_splits = text_splitter.split_documents(docs_list)[:50]  # LIMITING TO 50 CHUNKS
        db = Chroma.from_documents(doc_splits, embeddings, persist_directory=chroma_dir)
        retriever = db.as_retriever(search_kwargs={"k": 3})  # Retrieve only top 3
        st.sidebar.success("Document indexed. Ask questions now!")

# Setup LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

# Tool Wrappers
arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1))
serp = SerpAPIWrapper(serpapi_api_key=serp_api_key)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))

# State
class GraphState(TypedDict):
    question: str
    documents: List[str]

# Retrieval Functions
def retrieve(state):
    question = state["question"]
    documents = retriever.invoke(question)
    return {
        "documents": [
            Document(page_content=doc.page_content) for doc in documents
        ],
        "question": question
    }

def wiki_search_fn(state):
    docs = wikipedia.invoke({"query": state["question"]})
    return {"documents": [Document(page_content=str(docs))], "question": state["question"]}

def arxiv_search_fn(state):
    docs = arxiv.invoke({"query": state["question"]})
    return {"documents": [Document(page_content=str(docs))], "question": state["question"]}

def serp_search_fn(state):
    docs = serp.run(state["question"])
    return {"documents": [Document(page_content=str(docs))], "question": state["question"]}

# Graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve)
workflow.add_node("wiki_search", wiki_search_fn)
workflow.add_node("arxiv_search", arxiv_search_fn)
workflow.add_node("serp_search", serp_search_fn)

workflow.add_edge("retrieve", "wiki_search")
workflow.add_edge("wiki_search", "arxiv_search")
workflow.add_edge("arxiv_search", "serp_search")
workflow.add_edge("serp_search", END)
workflow.add_edge(START, "retrieve" if retriever else "wiki_search")

app = workflow.compile()

# Main Q&A
if question:
    st.markdown(f"### Your Question: {question}")
    inputs = {"question": question}
    sources_collected = []

    with st.spinner("Searching..."):
        if retriever:
            st.info("Using uploaded document...")
            docs = retriever.invoke(question)
            sources_collected = [doc.page_content for doc in docs]
        else:
            st.info("Using multi-agent search (Wiki, Arxiv, Google)...")
            for output in app.stream(inputs):
                for key, value in output.items():
                    for doc in value.get("documents", [])[:1]:  # Take only top document per source
                        sources_collected.append(f"Source: {key}\n{doc.page_content}")

    combined_text = "\n\n".join(sources_collected[:3])  # Limit to 3 most relevant sources

    final_prompt = f"""
    You are a helpful assistant. Below are excerpts from relevant sources:

    {combined_text}

    Based on this, answer the following question clearly and concisely:
    {question}
    """

    try:
        final_answer = llm.invoke(final_prompt)
    except Exception as e:
        st.error("LLM failed to generate a response. Try again with a simpler question.")
        st.exception(e)
        final_answer = None

    if final_answer:
        st.markdown("## Your Answer")
        st.write(final_answer.content)

    with st.expander("See Retrieved Sources"):
        for i, content in enumerate(sources_collected[:5]):
            st.markdown(f"### Source {i+1}")
            st.write(content)

# Cleanup memory after each request
gc.collect()
