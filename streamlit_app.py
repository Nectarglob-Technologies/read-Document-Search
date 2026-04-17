"""🚀 FINAL STREAMLIT (CLEAN + NO DUPLICATION + BACKGROUND INGESTION)"""

import streamlit as st
from pathlib import Path
import sys
import time
import threading

import builtins
from backend.src.utils.logger import get_logger

logger = get_logger("streamlit")




# =========================================================
# PATH FIX
# =========================================================
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

# =========================================================
# IMPORTS
# =========================================================
from backend.src.config.config import Config
from backend.src.document_ingestion.document_processor import DocumentProcessor
from backend.src.vectorstore.faiss_store import FAISSStore
from backend.src.pipeline.ingestion_pipeline import IngestionPipeline

from backend.src.agents.engineering_agent import EngineeringAgent
from backend.src.agents.weather_agent import WeatherAgent
from backend.src.agents.tender_agent import TenderAgent
from backend.src.agents.project_agent import ProjectAgent
from backend.src.agents.graph_agent import GraphAgent
from backend.src.agents.orchestrator_agent import OrchestratorAgent

from backend.src.graph_rag.graph_query import GraphQueryEngine
from backend.src.memory.conversation_memory import ConversationMemory


# =========================================================
# CONFIG
# =========================================================
DATA_DIR = BASE_DIR / "data"
GRAPH_FILE = BASE_DIR / "graph.pkl"


# =========================================================
# BACKGROUND INGESTION
# =========================================================
def run_ingestion_in_background(pipeline, chunks):
    """
    🔥 Run ingestion in background (non-blocking UI)
    """
    def task():
        logger.info("Background ingestion started...")
        pipeline.ingest(chunks)
        logger.info("Background ingestion completed")

    thread = threading.Thread(target=task)
    thread.start()


# =========================================================
# INITIALIZE SYSTEM (CACHED)
# =========================================================
@st.cache_resource
def initialize_system():

    logger.info(" Initializing system...")

    llm = Config.get_llm()

    store = FAISSStore()

    # 🔥 Load FAISS (NO CREATION HERE)
    faiss_exists = store.load()
    logger.info(f"FAISS loaded: {faiss_exists}")

    pipeline = IngestionPipeline(store, GRAPH_FILE, llm)

    # =====================================================
    # 🔥 FIRST TIME ONLY (NO DUPLICATION)
    # =====================================================
    if not faiss_exists or not pipeline.graph_store:

        logger.info("First-time ingestion...")

        processor = DocumentProcessor()

        raw_docs = processor.load_documents(DATA_DIR)
        chunks = processor.split_documents(raw_docs)

        logger.info(f"Initial chunks: {len(chunks)}")

        pipeline.ingest(chunks)

    graph_store = pipeline.graph_store

    # =====================================================
    # RETRIEVER
    # =====================================================
    retriever = store.get_retriever()

    # =====================================================
    # AGENTS
    # =====================================================
    engineering = EngineeringAgent(llm, retriever)
    weather = WeatherAgent(llm)
    tender = TenderAgent(llm, retriever)
    project = ProjectAgent(graph_store)

    query_engine = GraphQueryEngine(graph_store)
    graph_agent = GraphAgent(llm, query_engine)

    memory = ConversationMemory()

    orchestrator = OrchestratorAgent(
        llm,
        engineering,
        weather,
        tender,
        project,
        graph_agent,
        memory
    )

    logger.info("System Ready")

    return orchestrator, store, pipeline


# =========================================================
# FILE UPLOAD (SAFE + NON-BLOCKING)
# =========================================================
def process_uploaded_file(uploaded_file, store, pipeline):

    file_path = DATA_DIR / uploaded_file.name

    # 🔥 Prevent re-processing same file
    if file_path.exists():
        st.warning("⚠️ File already exists. Skipping ingestion.")
        return

    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"📄 Uploaded: {uploaded_file.name}")

    processor = DocumentProcessor()

    # Load only uploaded file
    raw_docs = processor.load_single_document(file_path)

    chunks = processor.split_documents(raw_docs)

    logger.info(f"New chunks: {len(chunks)}")

    # 🔥 BACKGROUND INGESTION
    run_ingestion_in_background(pipeline, chunks)

    st.info("⚡ File ingestion running in background...")


# =========================================================
# MAIN UI
# =========================================================
def main():

    st.set_page_config(page_title="🤖 Agentic RAG", layout="centered")

    st.title("🤖 Agentic RAG System")
    st.markdown("Upload documents or ask questions")

    DATA_DIR.mkdir(exist_ok=True)

    orchestrator, store, pipeline = initialize_system()

    if not orchestrator:
        st.warning("⚠️ No documents found.")
        return

    # =====================================================
    # FILE UPLOAD
    # =====================================================
    st.markdown("### 📂 Upload Document")

    uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

    if uploaded_file:
        process_uploaded_file(uploaded_file, store, pipeline)

    st.markdown("---")

    # =====================================================
    # QUESTION
    # =====================================================
    question = st.text_input("Ask your question")

    if st.button("🔍 Search") and question:

        with st.spinner("Thinking..."):
            start = time.time()

            answer = orchestrator.answer(question)

            elapsed = time.time() - start

        st.markdown("### 💡 Answer")
        st.success(answer)

        st.caption(f"⏱️ Response time: {elapsed:.2f}s")

        st.markdown("### ⚙️ Debug")
        st.info("Answer generated via OrchestratorAgent")


# =========================================================
if __name__ == "__main__":
    main()
