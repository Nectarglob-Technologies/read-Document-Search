from pydantic import BaseModel
import threading

from fastapi import UploadFile, File, FastAPI
import shutil
from pathlib import Path


# 🔥 Import your existing setup
from backend.src.config.config import Config
from backend.src.graph_rag.graph_builder import GraphBuilder
from backend.src.graph_rag.graph_query import GraphQueryEngine
from backend.src.vectorstore.faiss_store import FAISSStore
from backend.src.document_ingestion.document_processor import DocumentProcessor

from backend.src.agents.engineering_agent import EngineeringAgent
from backend.src.agents.weather_agent import WeatherAgent
from backend.src.agents.tender_agent import TenderAgent
from backend.src.agents.project_agent import ProjectAgent
from backend.src.agents.orchestrator_agent import OrchestratorAgent
from backend.src.agents.graph_agent import GraphAgent

from backend.src.memory.conversation_memory import ConversationMemory

import os
import pickle
import hashlib

app = FastAPI(title="AI Engineering Assistant API")

UPLOAD_DIR = "uploaded_docs"
Path(UPLOAD_DIR).mkdir(exist_ok=True)

# 🔥 Global instance (loaded once)
orchestrator = None
lock = threading.Lock()


# ---------------- REQUEST MODEL ----------------
class QueryRequest(BaseModel):
    question: str


# ---------------- HEALTH CHECK ----------------
@app.get("/")
def root():
    return {"status": "API is running"}


# ---------------- HASH ----------------
def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


# ---------------- SYSTEM INIT ----------------
def initialize_system():
    global orchestrator

    print("🚀 Initializing system (API mode)...")

    llm = Config.get_llm()

    # ---------------- DOCUMENTS ----------------
    processor = DocumentProcessor()
    raw_docs = processor.load_documents()
    docs = processor.split_documents(raw_docs)

    # ---------------- FAISS ----------------
    store = FAISSStore()

    if not store.load():
        store.create(docs)
        store.save()

    retriever = store.get_retriever()

    # ---------------- AGENTS ----------------
    engineering = EngineeringAgent(llm, retriever)
    weather = WeatherAgent(llm)
    tender = TenderAgent(llm, retriever)
    

    # ---------------- GRAPH ----------------
    graph_file = "graph.pkl"

    if os.path.exists(graph_file):
        graph_store = pickle.load(open(graph_file, "rb"))

    else:
        graph_builder = GraphBuilder(llm)

        processed_hashes = set()
        batch = []

        for doc in raw_docs:
            text = doc.page_content.strip()

            if len(text) < 200:
                continue

            h = get_hash(text)
            if h in processed_hashes:
                continue

            processed_hashes.add(h)
            batch.append(text)

            if len(batch) == 3:
                graph_builder.process_document("\n\n".join(batch))
                batch = []

        if batch:
            graph_builder.process_document("\n\n".join(batch))

        graph_store = graph_builder.get_graph()
        pickle.dump(graph_store, open(graph_file, "wb"))


    query_engine = GraphQueryEngine(graph_store)
    graph_agent = GraphAgent(llm, query_engine)
    project = ProjectAgent(graph_store)
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

    print("✅ System Ready (API)")


# ---------------- STARTUP ----------------
@app.on_event("startup")
def startup_event():
    with lock:
        initialize_system()


# ---------------- ASK API ----------------
@app.post("/ask")
def ask_question(request: QueryRequest):

    if not orchestrator:
        return {"error": "System not initialized"}

    answer = orchestrator.answer(request.question)

    return {
        "question": request.question,
        "answer": answer
    }

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):

    if not file.filename.endswith(".pdf"):
        return {"error": "Only PDF files are supported"}

    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print(f"📄 File uploaded: {file.filename}")

    try:
        process_uploaded_document(file_path)
    except Exception as e:
        return {"error": str(e)}

    return {
        "message": f"{file.filename} processed successfully"
    }

def process_uploaded_document(file_path):

    global orchestrator

    print("⚙️ Processing uploaded document...")

    # 🔹 Document processor
    processor = DocumentProcessor()

    # Load single file
    docs = processor.load_documents(os.path.dirname(file_path))

    # Filter only this file
    docs = [d for d in docs if file_path in d.metadata.get("source", "")]

    if not docs:
        print("No docs found after upload")
        return

    chunks = processor.split_documents(docs)

    # ---------------- UPDATE FAISS ----------------
    print("🔄 Updating FAISS index...")

    store = FAISSStore()

    if store.load():
        store.add_documents(chunks)
        store.save()
    else:
        store.create(chunks)
        store.save()

    # ---------------- UPDATE GRAPH ----------------
    print("🔄 Updating Knowledge Graph...")

    graph_file = "graph.pkl"

    if os.path.exists(graph_file):
        graph_store = pickle.load(open(graph_file, "rb"))
    else:
        graph_store = None

    graph_builder = GraphBuilder(orchestrator.llm)

    if graph_store:
        graph_builder.store = graph_store

    for doc in docs:
        graph_builder.process_document(
            doc.page_content,
            metadata=getattr(doc, "metadata", {})
        )

    graph_store = graph_builder.get_graph()

    pickle.dump(graph_store, open(graph_file, "wb"))

    print("✅ Upload processing completed")

