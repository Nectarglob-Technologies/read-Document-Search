from backend.src.config.config import Config
from backend.src.graph_rag.graph_builder import GraphBuilder
from backend.src.graph_rag.graph_query import GraphQueryEngine
from backend.src.vectorstore.faiss_store import FAISSStore
from backend.src.document_ingestion.document_processor import DocumentProcessor

from backend.src.agents.engineering_agent_old import EngineeringAgent
from backend.src.agents.weather_agent import WeatherAgent
from backend.src.agents.tender_agent import TenderAgent
from backend.src.agents.project_agent import ProjectAgent
from backend.src.agents.orchestrator_agent import OrchestratorAgent
from backend.src.agents.graph_agent import GraphAgent

from backend.src.memory.conversation_memory import ConversationMemory

import os
import pickle
import hashlib


# 🔹 Hash function for caching
def get_hash(text):
    return hashlib.md5(text.encode()).hexdigest()


def main():

    print("Initializing system... get LLM instance")
    llm = Config.get_llm()

    # ---------------- DOCUMENT PROCESSING ----------------
    print("Initializing document processor...")
    processor = DocumentProcessor()

    print("Loading and processing documents...")
    # 🔥 Load RAW documents (for graph)
    raw_docs = processor.load_documents()
    print("Total raw documents loaded:", len(raw_docs))

    
    # 🔥 Split documents (for FAISS)
    docs = processor.split_documents(raw_docs)

    print("Total chunks created:", len(docs))
    print("Sample chunk:", docs[0].page_content[:200])

    # ---------------- FAISS ----------------
    print("Initializing FAISS vector store...")
    store = FAISSStore()
    if store.load():
        print("✅ Loaded existing FAISS index (FAST 🚀)")

    else:
        print("⚡ Creating FAISS index (first time)...")

        store.create(docs)

        print("💾 Saving FAISS index to disk...")
        store.save()

        print("✅ FAISS index saved.")
    retriever = store.get_retriever()

    # ---------------- AGENTS ----------------
    print("Initializing agents...")
    engineering = EngineeringAgent(llm, retriever)
    weather = WeatherAgent(llm)
    tender = TenderAgent()
    project = ProjectAgent()

    # ---------------- GRAPH ----------------
    print("Building/loading knowledge graph...")
    graph_file = "graph.pkl"

    print("Graph file path:", graph_file)
    if os.path.exists(graph_file):

        print("Loading existing knowledge graph...")
        graph_store = pickle.load(open(graph_file, "rb"))

    else:

        print("Building knowledge graph (Optimized 🚀)...")
        graph_builder = GraphBuilder(llm)  # LLM only as fallback

        processed_hashes = set()
        batch = []
        batch_size = 3  # adjust if needed

        print("Processing documents in batches...")
        for doc in raw_docs:
            print("Processing document:", doc.metadata.get("source", "unknown source"))
            text = doc.page_content.strip()

            # ✅ Skip small docs
            if len(text) < 200:
                continue

            print("hash based caching, batch processing, and skipping small docs for efficiency")
            # ✅ Hash-based caching
            h = get_hash(text)
            if h in processed_hashes:
                continue

            print("Processing document with hash:", h)
            processed_hashes.add(h)

            print("Adding document to batch...")
            # ✅ Batch processing
            batch.append(text)

            print("Current batch size:", len(batch))
            if len(batch) == batch_size:

                combined_text = "\n\n".join(batch)
                graph_builder.process_document(combined_text)

                batch = []

        print("Finished processing all documents.")
        # process remaining batch
        if batch:
            combined_text = "\n\n".join(batch)
            graph_builder.process_document(combined_text)

        print("get graph store...")
        graph_store = graph_builder.get_graph()

        print("Saving graph to disk...")
        pickle.dump(graph_store, open(graph_file, "wb"))

        print("Graph saved to disk.")

    # ---------------- GRAPH QUERY ----------------
    query_engine = GraphQueryEngine(graph_store,llm)
    graph_agent = GraphAgent(llm, query_engine)

    print("Graph ready for querying.")

    # ---------------- MEMORY ----------------

    print("Initializing conversation memory...")
    memory = ConversationMemory()

    # ---------------- ORCHESTRATOR ----------------
    print("Initializing orchestrator agent...")
    orchestrator = OrchestratorAgent(
        llm,
        engineering,
        weather,
        tender,
        project,
        graph_agent,
        memory
    )

    print("System Ready ✅")

    # ---------------- INTERACTIVE LOOP ----------------
    while True:

        q = input("\nAsk question: ")

        if q.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        # ✅ IMPORTANT: use answer(), not route()
        answer = orchestrator.answer(q)

        print("\nAnswer:\n", answer)


if __name__ == "__main__":
    main()
