# backend/src/vectorstore/faiss_store.py

import os
from concurrent.futures import ThreadPoolExecutor
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from backend.src.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSStore:

    def __init__(self, path="faiss_index"):

        # 🔹 Embedding model
        self.embedding = OpenAIEmbeddings()

        # 🔹 FAISS instance
        self.vectorstore = None

        # 🔹 Storage path
        self.path = path

    # =========================================================
    # 🔥 SAFE TEXT (PREVENT TOKEN OVERFLOW)
    # =========================================================
    def _safe_text(self, text, max_chars=4000):
        """
        Trim long text to avoid token overflow error
        """
        return text[:max_chars] if text else ""

    # =========================================================
    # 🔥 FILTER USELESS TEXT (COST REDUCTION)
    # =========================================================
    def _is_useful(self, text):

        if not text or len(text.strip()) < 100:
            return False

        # ❌ Skip numeric-heavy chunks (tables, noise)
        if sum(c.isdigit() for c in text) > len(text) * 0.5:
            return False

        # ❌ Skip garbage/repeated text
        if len(set(text)) < 10:
            return False

        return True

    # =========================================================
    # 🔥 DOMAIN FILTER (SMART FILTERING)
    # =========================================================
    def _is_domain_relevant(self, text):

        keywords = [
            "project", "contract", "construction",
            "material", "concrete", "clause",
            "contractor", "bridge"
        ]

        text = text.lower()

        return any(k in text for k in keywords)

    # =========================================================
    # 🔥 PREPARE DOCUMENTS (CORE OPTIMIZATION)
    # =========================================================
    def _prepare_docs(self, docs):

        clean_docs = []

        for doc in docs:

            text = doc.page_content if hasattr(doc, "page_content") else ""

            # ❌ Remove useless chunks
            if not self._is_useful(text):
                continue

            # ❌ Remove irrelevant chunks
            if not self._is_domain_relevant(text):
                continue

            # 🔹 Trim text
            doc.page_content = self._safe_text(text)

            clean_docs.append(doc)

        return clean_docs

    # =========================================================
    # ✅ CREATE (FIRST TIME BUILD)
    # =========================================================
    def create(self, documents, batch_size=50):

        logger.info("Creating FAISS in batches...")

        self.vectorstore = None

        for i in range(0, len(documents), batch_size):

            batch = documents[i:i + batch_size]

            # 🔹 Clean + filter
            batch = self._prepare_docs(batch)

            if not batch:
                continue

            logger.info(f"Batch {i} → {i + len(batch)}")

            # First batch → create index
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(batch, self.embedding)

            # Next batches → append
            else:
                self.vectorstore.add_documents(batch)

    # =========================================================
    # ✅ ADD DOCUMENTS (INCREMENTAL SAFE)
    # =========================================================
    def add_documents(self, docs, batch_size=50):

        if not self.vectorstore:
            raise ValueError("FAISS not initialized")

        logger.info(f"Adding {len(docs)} docs to FAISS")

        for i in range(0, len(docs), batch_size):

            batch = docs[i:i + batch_size]

            # 🔹 Clean + filter
            batch = self._prepare_docs(batch)

            if not batch:
                continue

            logger.info(f"Batch {i} → {i + len(batch)}")

            self.vectorstore.add_documents(batch)

    # =========================================================
    # ⚡ OPTIONAL PARALLEL (DISABLED BY DEFAULT)
    # =========================================================
    def add_documents_parallel(self, docs, batch_size=50, max_workers=3):

        """
        ⚠️ USE ONLY IF:
        - You want speed
        - Your API limits allow parallel calls

        Default → DO NOT USE
        """

        if not self.vectorstore:
            raise ValueError("❌ FAISS not initialized")

        print("⚡ Running PARALLEL embedding...")

        def process(batch):
            batch = self._prepare_docs(batch)
            if batch:
                self.vectorstore.add_documents(batch)

        batches = [
            docs[i:i + batch_size]
            for i in range(0, len(docs), batch_size)
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            executor.map(process, batches)

    # =========================================================
    # 💾 SAVE
    # =========================================================
    def save(self):

        if self.vectorstore:
            self.vectorstore.save_local(self.path)
            print("💾 FAISS saved")

    # =========================================================
    # 📂 LOAD (SAFE FIX FOR YOUR ERROR)
    # =========================================================
    def load(self):

        index_file = os.path.join(self.path, "index.faiss")
        pkl_file = os.path.join(self.path, "index.pkl")

        # 🔥 CRITICAL FIX: both files must exist
        if os.path.exists(index_file) and os.path.exists(pkl_file):

            print("📂 Loading FAISS...")

            self.vectorstore = FAISS.load_local(
                self.path,
                self.embedding,
                allow_dangerous_deserialization=True
            )

            return True

        print("⚠️ FAISS index not found")
        return False

    # =========================================================
    # 🔎 RETRIEVER
    # =========================================================
    def get_retriever(self):

        if not self.vectorstore:
            raise ValueError("❌ FAISS not initialized")

        return self.vectorstore.as_retriever()
