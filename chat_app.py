import streamlit as st
from backend.src.config.config import Config
from backend.src.document_ingestion.document_processor import DocumentProcessor
from backend.src.vectorstore.faiss_store import FAISSStore
from backend.src.graph_rag.graph_builder import GraphBuilder
from backend.src.graph_rag.graph_query import GraphQueryEngine

from backend.src.agents.engineering_agent_old import EngineeringAgent
from backend.src.agents.weather_agent import WeatherAgent
from backend.src.agents.tender_agent import TenderAgent
from backend.src.agents.project_agent import ProjectAgent
from backend.src.agents.graph_agent import GraphAgent
from backend.src.agents.orchestrator_agent import OrchestratorAgent

from backend.src.memory.conversation_memory import ConversationMemory

import pickle
import os

# ---------------- INIT SYSTEM (CACHE) ----------------
@st.cache_resource
def initialize_system():

    llm = Config.get_llm()

    # ---------------- DOCUMENTS ----------------
    processor = DocumentProcessor()
    raw_docs = processor.load_documents()
    docs = processor.split_documents(raw_docs)

    # ---------------- VECTOR DB ----------------
    store = FAISSStore()

    if os.path.exists("faiss_index"):
        store.load("faiss_index")
    else:
        store.create(docs)
        store.save("faiss_index")

    retriever = store.get_retriever()

    # ---------------- AGENTS ----------------
    engineering = EngineeringAgent(llm, retriever)
    weather = WeatherAgent(llm)
    tender = TenderAgent()
    project = ProjectAgent()

    # ---------------- GRAPH ----------------
    graph_file = "graph.pkl"

    if os.path.exists(graph_file):
        graph_store = pickle.load(open(graph_file, "rb"))
    else:
        graph_builder = GraphBuilder(llm)
        for doc in raw_docs:
            graph_builder.process_document(doc.page_content)

        graph_store = graph_builder.get_graph()
        pickle.dump(graph_store, open(graph_file, "wb"))

    query_engine = GraphQueryEngine(graph_store, llm)
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

    return orchestrator


# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Infra AI Assistant", layout="wide")

st.title("🤖 Infrastructure AI Assistant")

# Initialize system once
orchestrator = initialize_system()

# ---------------- CHAT MEMORY ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- DISPLAY CHAT ----------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------- USER INPUT ----------------
if prompt := st.chat_input("Ask about projects, materials, tenders, weather..."):

    # Show user message
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.spinner("Thinking..."):

        response = orchestrator.answer(prompt)

    # Show assistant response
    st.chat_message("assistant").markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
