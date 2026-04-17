# backend/src/pipeline/ingestion_pipeline.py

import pickle
from pathlib import Path
from copy import deepcopy

from backend.src.graph_rag.graph_builder import GraphBuilder

from backend.src.utils.logger import get_logger

logger = get_logger(__name__)


# 🔥 OPTIONAL (you can disable easily)
USE_SEMANTIC_CHUNKING = False

try:
    from backend.src.pipeline.chunking import semantic_chunk_documents
except:
    semantic_chunk_documents = None


class IngestionPipeline:

    def __init__(self, store, graph_file, llm):

        # 🔹 FAISS store
        self.store = store

        # 🔹 Graph file path
        self.graph_file = Path(graph_file)

        # 🔹 LLM
        self.llm = llm

        # 🔹 Load existing graph
        self.graph_store = self._load_graph()

    # =========================================================
    # LOAD GRAPH
    # =========================================================
    def _load_graph(self):

        if self.graph_file.exists():
            logger.info("Loading existing graph...")
            return pickle.load(open(self.graph_file, "rb"))

        logger.info("No graph found")
        return None

    # =========================================================
    # SAVE GRAPH
    # =========================================================
    def _save_graph(self):

        pickle.dump(self.graph_store, open(self.graph_file, "wb"))
        logger.info("Graph saved")

    # =========================================================
    # SAFE TEXT (TOKEN CONTROL)
    # =========================================================
    def _safe_text(self, text, max_chars=4000):
        return text[:max_chars] if text else ""

    # =========================================================
    # 🔥 MAIN PIPELINE
    # =========================================================
    def ingest(self, chunks, batch_size=50):

        if not chunks:
            return

        logger.info("Total input chunks: {len(chunks)}")

        # =====================================================
        # 🔥 STEP 1: OPTIONAL SEMANTIC CHUNKING
        # =====================================================
        if USE_SEMANTIC_CHUNKING and semantic_chunk_documents:
            logger.info("Applying semantic chunking...")
            chunks = semantic_chunk_documents(chunks)
            logger.info(f"After semantic chunking: {len(chunks)} chunks")

        # =====================================================
        # 🔥 STEP 2: FAISS (SAFE BATCHING)
        # =====================================================
        for i in range(0, len(chunks), batch_size):

            batch = chunks[i:i + batch_size]

            logger.info(f"FAISS Batch {i} → {i + len(batch)}")

            # 🔹 IMPORTANT: don't modify original chunks
            safe_batch = deepcopy(batch)

            # 🔹 TOKEN CONTROL (critical fix)
            for doc in safe_batch:
                if hasattr(doc, "page_content"):
                    doc.page_content = self._safe_text(doc.page_content)

            # 🔹 CREATE / ADD
            if self.store.vectorstore:
                self.store.add_documents(safe_batch)
            else:
                self.store.create(safe_batch)

        # 🔹 Save FAISS
        self.store.save()

        # =====================================================
        # 🔥 STEP 3: GRAPH (FULL TEXT - NO TRIM)
        # =====================================================
        logger.info("Processing graph...")

        graph_builder = GraphBuilder(self.llm)

        # 🔹 Reuse existing graph
        if self.graph_store:
            logger.info("Updating existing graph...")
            graph_builder.store = self.graph_store
        else:
            logger.info("Creating new graph...")

        # 🔹 IMPORTANT: use ORIGINAL chunks (not trimmed)
        graph_builder.process_batch(chunks)

        self.graph_store = graph_builder.get_graph()

        # 🔹 Save graph
        self._save_graph()
