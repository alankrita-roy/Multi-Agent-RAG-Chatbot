import os
import tempfile
import shutil
#from dotenv import load_dotenv
import streamlit as st

# ---- PATCH for ChromaDB: Fix sqlite3 version ----
import sys
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

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema import Document
from typing import List
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict

# Load environment variables
#load_dotenv()

# Ensure required environment variables are present
#groq_api_key = os.getenv("GROQ_API_KEY")
#serp_api_key = os.getenv("SERP_API_KEY")
groq_api_key = st.secrets["GROQ_API_KEY"]
serp_api_key = st.secrets["SERP_API_KEY"]

if not groq_api_key or not serp_api_key:
    st.error("Missing required API keys. Please check your environment variables.")
    st.stop()

# Temporary Chroma DB Directory
chroma_dir = tempfile.mkdtemp()

# Streamlit UI
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.sidebar.title("Options")

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
input_url = st.sidebar.text_input("Paste a URL")

st.title("Multi Agent Chatbot with Wiki, Arxiv and Google Search")
question = st.chat_input("Ask your question")

# Initialize Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# Load documents
docs_list = []
retriever = None

if uploaded_file:
    with st.spinner("Indexing PDF, please wait..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        docs_list = loader.load()
        os.remove(tmp_file_path)
elif input_url:
    with st.spinner("Indexing text from URL, please wait..."):
        loader = WebBaseLoader(input_url)
        docs_list = loader.load()

if (uploaded_file or input_url) and (not docs_list or all(len(doc.page_content.strip()) == 0 for doc in docs_list)):
    st.error("No text found in the provided PDF or URL. Please try another.")
    st.stop()

if docs_list:
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=50)
    doc_splits = text_splitter.split_documents(docs_list)

    db = Chroma.from_documents(doc_splits, embeddings, persist_directory=chroma_dir)
    retriever = db.as_retriever()

    st.sidebar.success("You can now ask questions based on this document.")

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
    cleaned_docs = [Document(page_content=doc.page_content if isinstance(doc.page_content, str) else str(doc.page_content)) for doc in documents]
    return {"documents": cleaned_docs, "question": question}

def wiki_search_fn(state):
    docs = wikipedia.invoke({"query": state["question"]})
    content = docs if isinstance(docs, str) else str(docs)
    return {"documents": [Document(page_content=content)], "question": state["question"]}

def arxiv_search_fn(state):
    docs = arxiv.invoke({"query": state["question"]})
    content = docs if isinstance(docs, str) else str(docs)
    return {"documents": [Document(page_content=content)], "question": state["question"]}

def serp_search_fn(state):
    docs = serp.run(state["question"])
    content = docs if isinstance(docs, str) else str(docs)
    return {"documents": [Document(page_content=content)], "question": state["question"]}

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

# Run on question directly
if question:
    inputs = {"question": question}
    all_responses = []

    with st.spinner("Searching, please wait..."):
        for output in app.stream(inputs):
            for key, value in output.items():
                docs = value.get("documents", [])
                for doc in docs:
                    all_responses.append((key, doc.page_content))

    combined_text = "\n\n".join([f"Source: {key}\n{content}" for key, content in all_responses])

    final_prompt = f"""
    You are a helpful assistant. Here are the answers from different sources:

    {combined_text}

    Please provide the most accurate and relevant response to the original question: "{question}"
    """

    final_answer = llm.invoke(final_prompt)

    with st.expander("See Agent Work"):
        for key, content in all_responses:
            st.markdown(f"### Source: `{key}`")
            st.write(content)

    st.markdown("## Your Answer")
    st.write(final_answer.content)

# Cleanup on close
@st.cache_resource
def cleanup():
    shutil.rmtree(chroma_dir)

cleanup()
