"""🚀 FINAL STREAMLIT (STABLE + DEBUG + MEMORY + CHAT UI)"""

import streamlit as st
from pathlib import Path
import sys
import time
import threading
import uuid

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

    def task():
        logger.info("Background ingestion started...")
        pipeline.ingest(chunks)
        logger.info("Background ingestion completed")

    thread = threading.Thread(target=task)
    thread.start()


# =========================================================
# SYSTEM INIT (DEBUG SAFE)
# =========================================================
@st.cache_resource
def initialize_system():

    logger.info("🚀 Initializing system...")

    # STEP A: LLM
    logger.info("STEP A: Loading LLM")
    llm = Config.get_llm()

    # STEP B: FAISS
    logger.info("STEP B: Loading FAISS")
    store = FAISSStore()
    faiss_exists = store.load()
    logger.info(f"FAISS exists: {faiss_exists}")

    # STEP C: Pipeline
    logger.info("STEP C: Initializing pipeline")
    pipeline = IngestionPipeline(store, GRAPH_FILE, llm)

    # STEP D: First-time ingestion
    if not faiss_exists or not getattr(pipeline, "graph_store", None):

        logger.info("⚠️ Running ingestion (first time or missing graph)...")

        processor = DocumentProcessor()
        raw_docs = processor.load_documents(DATA_DIR)
        chunks = processor.split_documents(raw_docs)

        logger.info(f"Chunks: {len(chunks)}")

        pipeline.ingest(chunks)

    graph_store = pipeline.graph_store

    if graph_store is None:
        raise Exception("❌ Graph store is None (ingestion failed)")

    retriever = store.get_retriever()

    logger.info("STEP D: System core ready")

    return llm, retriever, graph_store, store, pipeline


# =========================================================
# BUILD AGENTS (SAFE)
# =========================================================
def build_agents(llm, retriever, graph_store, memory):

    logger.info("⚙️ Building agents...")

    query_engine = GraphQueryEngine(graph_store)

    graph_agent = GraphAgent(llm, query_engine, memory=memory)

    engineering = EngineeringAgent(
        llm,
        retriever,
        graph_agent=graph_agent,
        memory=memory
    )

    weather = WeatherAgent(llm)
    tender = TenderAgent(llm, retriever)
    project = ProjectAgent(graph_store)

    orchestrator = OrchestratorAgent(
        llm,
        engineering,
        weather,
        tender,
        project,
        graph_agent,
        memory
    )

    logger.info("✅ Agents Ready")

    return orchestrator, graph_agent


# =========================================================
# SESSION INIT
# =========================================================
def initialize_session():

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()

    if "messages" not in st.session_state:
        st.session_state.messages = []


# =========================================================
# FILE UPLOAD
# =========================================================
def process_uploaded_file(uploaded_file, store, pipeline):

    file_path = DATA_DIR / uploaded_file.name

    if file_path.exists():
        st.warning("⚠️ File already exists.")
        return

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"📄 Uploaded: {uploaded_file.name}")

    processor = DocumentProcessor()
    raw_docs = processor.load_single_document(file_path)
    chunks = processor.split_documents(raw_docs)

    run_ingestion_in_background(pipeline, chunks)

    st.info("⚡ Background ingestion started...")


# =========================================================
# MAIN UI
# =========================================================
def main():

    st.set_page_config(page_title="🤖 Agentic RAG", layout="centered")
    st.title("🤖 Agentic RAG System")

    DATA_DIR.mkdir(exist_ok=True)

    # STEP 1: SESSION
    initialize_session()
    st.success("✅ STEP 1: Session Ready")

    session_id = st.session_state.session_id
    memory = st.session_state.memory

    # STEP 2: SYSTEM INIT
    try:
        with st.spinner("🚀 Initializing system..."):
            llm, retriever, graph_store, store, pipeline = initialize_system()

        st.success("✅ STEP 2: System Ready")

    except Exception as e:
        st.error(f"❌ System init failed: {e}")
        st.stop()

    # DEBUG
    st.write("📊 Graph Store:", "Loaded" if graph_store else "❌ None")

    # STEP 3: AGENTS
    try:
        with st.spinner("⚙️ Building agents..."):
            orchestrator, graph_agent = build_agents(
                llm, retriever, graph_store, memory
            )

        st.success("✅ STEP 3: Agents Ready")

    except Exception as e:
        st.error(f"❌ Agent build failed: {e}")
        st.stop()

    st.success("🎯 STEP 4: UI Ready")

    # =====================================================
    # FILE UPLOAD
    # =====================================================
    with st.expander("📂 Upload Document"):
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded_file:
            process_uploaded_file(uploaded_file, store, pipeline)

    # =====================================================
    # CHAT UI
    # =====================================================
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # =====================================================
    # USER INPUT
    # =====================================================
    if prompt := st.chat_input("Ask your question..."):

        # USER
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # AI
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                start = time.time()

                result = orchestrator.answer(prompt, session_id)
                answer = result.get("answer", "No response")

                elapsed = time.time() - start

                st.markdown(answer)
                st.caption(f"⏱️ {elapsed:.2f}s | Confidence: {result.get('confidence')}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    # =====================================================
    # DEBUG PANEL
    # =====================================================
    with st.expander("🧠 Debug Info"):

        st.write("Session ID:", session_id)

        if st.button("Show Summary"):
            st.write(memory.get_summary(session_id))

        if st.button("Show Recent History"):
            st.json(memory.get_recent_history(session_id))

        if st.button("Clear Memory"):
            memory.clear(session_id)
            st.success("Memory cleared")

    # =====================================================
    # CLEAR CHAT
    # =====================================================
    if st.button("🧹 Clear Chat UI"):
        st.session_state.messages = []
        st.rerun()


# =========================================================
if __name__ == "__main__":
    main()
