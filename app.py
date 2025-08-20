import os
import sys
import gc
import atexit
import shutil
import tempfile
from typing import List, Tuple

import streamlit as st

# ===============================
# Streamlit Multi‑Agent RAG (Low‑Memory)
# – Keeps original functionality:
#   * Upload PDF or fetch URL ➜ Chroma retriever
#   * Multi‑agent external search: Wikipedia ➜ Arxiv ➜ Google (SerpAPI)
#   * Gemma2‑9B‑It (Groq) for final synthesis
# – Memory‑safe tactics:
#   * Smaller chunks, capped index size
#   * Top‑K retrieval only
#   * Aggressive context truncation
#   * No large global buffers; stream to UI
#   * Temp Chroma dir cleaned with atexit
# ===============================

# ---- PATCH for ChromaDB: Fix sqlite3 version (kept from your original) ----
os.environ["PYSQLITE3_INCLUDE_PATH"] = "/home/adminuser/venv/lib/python3.10/site-packages/pysqlite3"
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = pysqlite3
except Exception:
    pass  # In case local environment already provides correct sqlite3

# ---- LangChain / LangGraph imports (match your original stack) ----
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, SerpAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun
from langchain.schema import Document
from langgraph.graph import END, StateGraph, START
from typing_extensions import TypedDict

# ===============================
# Config knobs (tuned for Streamlit Community Cloud limits)
# ===============================
CHUNK_SIZE = 300
CHUNK_OVERLAP = 20
MAX_INDEX_CHUNKS = 60        # Cap indexed chunks to avoid memory bloat
RETRIEVER_K = 3              # Top‑K for doc retriever
MAX_CTX_CHARS = 3600         # Total characters to send to LLM
PER_SOURCE_TAKE = 1          # Take at most 1 doc/summary per external source in the graph stream

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.sidebar.title("Options")

uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
input_url = st.sidebar.text_input("Paste a URL")

st.title("Multi‑Agent Chatbot with Wiki, Arxiv & Google Search")
question = st.chat_input("Ask your question")

# ===============================
# Secrets / API Keys
# ===============================
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
SERP_API_KEY = st.secrets.get("SERP_API_KEY")

if not GROQ_API_KEY:
    st.error("Missing GROQ_API_KEY in Streamlit secrets.")
    st.stop()

# ===============================
# Temp Chroma directory + cleanup
# ===============================
chroma_dir = tempfile.mkdtemp(prefix="chroma_")
atexit.register(lambda: shutil.rmtree(chroma_dir, ignore_errors=True))

# ===============================
# Embeddings (kept same model, CPU)
# ===============================
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)

# ===============================
# Load document(s) if provided
# ===============================
docs_list: List[Document] = []
retriever = None

if uploaded_file:
    with st.spinner("Indexing PDF…"):
        # Store to a temp file to let PyPDFLoader read it
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        try:
            loader = PyPDFLoader(tmp_file_path)
            docs_list = loader.load()
        finally:
            try:
                os.remove(tmp_file_path)
            except Exception:
                pass
elif input_url:
    with st.spinner("Fetching and indexing URL…"):
        loader = WebBaseLoader(input_url)
        docs_list = loader.load()

if (uploaded_file or input_url) and (not docs_list or all(len((doc.page_content or "").strip()) == 0 for doc in docs_list)):
    st.error("No text found in the provided PDF or URL. Please try another.")
    st.stop()

# ===============================
# Build lightweight Chroma retriever (keep functionality; lower memory)
# ===============================
if docs_list:
    with st.spinner("Splitting & building vector index…"):
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        doc_splits = splitter.split_documents(docs_list)
        # Cap number of chunks to index to keep RAM usage in check
        doc_splits = doc_splits[:MAX_INDEX_CHUNKS]

        db = Chroma.from_documents(doc_splits, embeddings, persist_directory=chroma_dir)
        retriever = db.as_retriever(search_kwargs={"k": RETRIEVER_K})
        st.sidebar.success("Document indexed. You can now ask questions.")

# ===============================
# LLM (same as your original)
# ===============================
llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="Gemma2-9b-It")

# ===============================
# External Agents (same as your original, wrapped for LangGraph)
# ===============================
wikipedia_tool = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1))
arxiv_tool = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1))
serp_wrapper = SerpAPIWrapper(serpapi_api_key=SERP_API_KEY) if SERP_API_KEY else None

class GraphState(TypedDict):
    question: str
    documents: List[Document]


def _wrap_text_to_doc(text: str) -> Document:
    return Document(page_content=text if isinstance(text, str) else str(text))

# ---- Retrieval node (uploaded doc retriever) ----
def retrieve_node(state: GraphState) -> GraphState:
    q = state["question"]
    docs = retriever.invoke(q) if retriever is not None else []
    # Ensure page_content is str
    cleaned = [_wrap_text_to_doc(d.page_content) for d in docs]
    return {"question": q, "documents": cleaned}

# ---- Wikipedia node ----
def wiki_node(state: GraphState) -> GraphState:
    q = state["question"]
    try:
        res = wikipedia_tool.invoke({"query": q})
        text = res if isinstance(res, str) else str(res)
    except Exception as e:
        text = f"[Wikipedia error] {e}"
    return {"question": q, "documents": [_wrap_text_to_doc(text)]}

# ---- Arxiv node ----
def arxiv_node(state: GraphState) -> GraphState:
    q = state["question"]
    try:
        res = arxiv_tool.invoke({"query": q})
        text = res if isinstance(res, str) else str(res)
    except Exception as e:
        text = f"[Arxiv error] {e}"
    return {"question": q, "documents": [_wrap_text_to_doc(text)]}

# ---- Serp (Google) node ----
def serp_node(state: GraphState) -> GraphState:
    q = state["question"]
    if not serp_wrapper:
        return {"question": q, "documents": [_wrap_text_to_doc("[SerpAPI key missing] Skipping Google search.")]}
    try:
        res = serp_wrapper.run(q)
        text = res if isinstance(res, str) else str(res)
    except Exception as e:
        text = f"[SerpAPI error] {e}"
    return {"question": q, "documents": [_wrap_text_to_doc(text)]}

# ---- Build graph (preserve original order & behavior) ----
workflow = StateGraph(GraphState)

# Nodes
if retriever is not None:
    workflow.add_node("retrieve", retrieve_node)
workflow.add_node("wiki_search", wiki_node)
workflow.add_node("arxiv_search", arxiv_node)
workflow.add_node("serp_search", serp_node)

# Edges
if retriever is not None:
    workflow.add_edge("retrieve", "wiki_search")
else:
    # If no document, start directly from wiki
    pass
workflow.add_edge("wiki_search", "arxiv_search")
workflow.add_edge("arxiv_search", "serp_search")
workflow.add_edge("serp_search", END)
workflow.add_edge(START, "retrieve" if retriever is not None else "wiki_search")

app = workflow.compile()

# ===============================
# Ask/Answer loop
# ===============================
if question:
    st.markdown(f"### Your Question\n> {question}")

    contexts: List[str] = []

    # Strategy:
    #   • If we have a doc retriever, use it and show top chunks
    #   • Always run the graph stream; keep only PER_SOURCE_TAKE item per node
    #   • Merge doc chunks + agent snippets; hard‑cap chars

    # 1) Document retrieval (if available)
    if retriever is not None:
        st.info("🔎 Using uploaded document for answering…")
        doc_hits = retriever.invoke(question)
        doc_hits = doc_hits[:RETRIEVER_K]
        for i, d in enumerate(doc_hits, 1):
            with st.expander(f"Document chunk {i}"):
                st.write(d.page_content)
            contexts.append(d.page_content)
    else:
        st.info("🌐 No document provided — using multi‑agent search…")

    # 2) Multi‑agent stream (Wiki ➜ Arxiv ➜ Serp)
    seen_per_key = {"wiki_search": 0, "arxiv_search": 0, "serp_search": 0}
    for output in app.stream({"question": question, "documents": []}):
        for key, value in output.items():
            docs = value.get("documents", []) if isinstance(value, dict) else []
            for doc in docs:
                if key in seen_per_key and seen_per_key[key] >= PER_SOURCE_TAKE:
                    continue
                snippet = str(doc.page_content)[:1200]
                with st.expander(f"{key} result #{seen_per_key.get(key, 0) + 1}"):
                    st.write(snippet)
                contexts.append(f"Source: {key}\n{snippet}")
                if key in seen_per_key:
                    seen_per_key[key] += 1

    # 3) Build final context (cap aggressively)
    merged = "\n\n".join(contexts)
    if len(merged) > MAX_CTX_CHARS:
        merged = merged[:MAX_CTX_CHARS] + "\n[...truncated...]"

    final_prompt = f"""
    You are a helpful assistant. Below are excerpts from relevant sources (document chunks and external agents).

    {merged}

    Based on this, provide the most accurate and relevant answer to: {question}
    If sources disagree or are insufficient, say so briefly and suggest next steps.
    """

    try:
        with st.spinner("Synthesizing answer…"):
            final_answer = llm.invoke(final_prompt)
        st.markdown("## Your Answer")
        st.write(final_answer.content)
    except Exception as e:
        st.error("❌ LLM failed to generate a response. Try a simpler question or smaller document.")
        st.exception(e)

    # Attempt to free memory proactively
    del contexts
    gc.collect()
