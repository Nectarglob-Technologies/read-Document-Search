from pathlib import Path
from typing import List
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.src.config.config import Config
from backend.src.document_ingestion.hierarchical_chunker import HierarchicalChunker

from backend.src.parsers.parser_router import ParserRouter
from backend.src.parsers.azure_doc_intelligence import AzureDocIntelligence
from backend.src.parsers.unstructrured_parser import UnstructuredParser
from backend.src.parsers.document_quality import DocumentQualityScorer
from backend.src.parsers.cache_manager import CacheManager
from backend.src.utils.logger import get_logger

import re

logger = get_logger(__name__)


class DocumentProcessor:

    def __init__(self, chunk_size=None, chunk_overlap=None):

        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        self.chunker = HierarchicalChunker()

        # SMART COMPONENTS
        logger.info("Initializing ParserRouter...")
        self.router = ParserRouter()

        logger.info("Initializing DocumentQualityScorer...")
        self.quality = DocumentQualityScorer()

        logger.info("Initializing CacheManager...")
        self.cache = CacheManager()

        # PARSERS
        self.unstructured_parser = UnstructuredParser()

        if getattr(Config, "USE_AZURE_DOC_INTELLIGENCE", False):
            self.azure_parser = AzureDocIntelligence()
        else:
            self.azure_parser = None

    # =========================================================
    # PARSER EXECUTION
    # =========================================================
    def _run_parser(self, parser, file_path):

        try:
            logger.info(f"Running parser: {parser} on {file_path}")

            if parser == "pypdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()

            elif parser == "unstructured":
                docs = self.unstructured_parser.parse(file_path)

            elif parser == "azure" and self.azure_parser:
                docs = self.azure_parser.parse(file_path)

            else:
                return None

            # 🔥 FIX: Ensure metadata consistency
            for i, d in enumerate(docs):
                d.metadata["parser"] = parser
                d.metadata["source"] = file_path

                # 🔥 IMPORTANT FIX FOR PAGE NUMBER
                if "page" not in d.metadata:
                    d.metadata["page"] = i + 1

            return docs

        except Exception as e:
            logger.error(f"Parser {parser} failed: {e}")
            return None

    # =========================================================
    # SINGLE FILE LOAD (DO NOT REMOVE)
    # =========================================================
    def load_single_document(self, file_path):

        logger.info(f"Loading single document: {file_path}")

        file_path = str(file_path)
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        # 🔥 Ensure metadata consistency
        for i, d in enumerate(docs):
            d.metadata["source"] = file_path
            if "page" not in d.metadata:
                d.metadata["page"] = i + 1

        return docs

    # =========================================================
    # LOAD DOCUMENTS
    # =========================================================
    def load_documents(self, directory=None) -> List[Document]:

        if directory is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            directory = base_dir / "data"

        path = Path(directory)
        logger.info(f"Loading documents from: {path}")

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            return []

        docs = []

        for file in path.glob("*.pdf"):

            file_path = str(file)
            logger.info(f"\n📄 Processing: {file_path}")

            # CACHE CHECK
            cached_parser = self.cache.get(file_path)

            if cached_parser:
                logger.info(f"Cache HIT → {cached_parser}")

                parsed_docs = self._run_parser(cached_parser, file_path)

                if parsed_docs:
                    docs.extend(parsed_docs)
                    continue

            # PARSER ORDER
            parser_order = self.router.get_parser_order(file_path)

            best_docs = None
            best_score = 0
            best_parser = None

            for parser in parser_order:

                parsed_docs = self._run_parser(parser, file_path)

                if not parsed_docs:
                    continue

                score = self.quality.score(parsed_docs)
                logger.info(f"{parser} score: {score:.2f}")

                if score > best_score:
                    best_docs = parsed_docs
                    best_score = score
                    best_parser = parser

                if score > 0.85:
                    break

            # SAVE LEARNING
            if best_parser:
                self.router.update_learning(file_path, best_parser)
                self.cache.set(file_path, best_parser)

            docs.extend(best_docs or [])

        logger.info(f"✅ Total documents loaded: {len(docs)}")
        return docs

    # =========================================================
    # STRUCTURE DETECTION
    # =========================================================
    def _has_structure(self, text):

        patterns = [
            r"Clause\s+\d+(\.\d+)*",
            r"Section\s+\d+",
            r"\b\d+\.\d+(\.\d+)*\b",
            r"\([a-zA-Z]\)",
            r"\([ivx]+\)"
        ]

        return any(re.search(p, text, re.IGNORECASE) for p in patterns)

    # =========================================================
    # SPLITTING (🔥 FIXED METADATA)
    # =========================================================
    def split_documents(self, docs: List[Document]) -> List[Document]:

        all_chunks = []

        for doc in docs:

            text = doc.page_content
            metadata = doc.metadata.copy()

            # TABLE
            if metadata.get("type") == "table":

                chunks = self.splitter.split_text(text)

                chunks = [
                    Document(
                        page_content=c,
                        metadata=metadata  # 🔥 PRESERVE METADATA
                    )
                    for c in chunks
                ]

            # STRUCTURED
            elif self._has_structure(text):

                chunks = self.chunker.process(text, metadata.get("source"))

                for c in chunks:
                    c.metadata.update(metadata)

            # DEFAULT
            else:
                chunks = self.splitter.split_documents([doc])

                for c in chunks:
                    c.metadata.update(metadata)

            all_chunks.extend(chunks)

        logger.info(f"✅ Total chunks created: {len(all_chunks)}")
        return all_chunks

    # =========================================================
    # MAIN PIPELINE
    # =========================================================
    def process(self, directory="data", return_raw=False):

        raw_docs = self.load_documents(directory)
        chunks = self.split_documents(raw_docs)

        logger.info(f"Loaded {len(raw_docs)} docs")
        logger.info(f"Created {len(chunks)} chunks")

        if return_raw:
            return raw_docs, chunks

        return chunks
