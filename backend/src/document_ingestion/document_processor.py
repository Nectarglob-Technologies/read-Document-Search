from asyncio.log import logger
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

from backend.src.utils.logger import get_logger

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

        # 🔥 Smart components
        logger.info("Call parserRouter to load memory file parser_memory.json which store pypdf value")
        self.router = ParserRouter()
        logger.info("ParserRouter initialized. Call internally _load to load file parser_router.json ")
        self.quality = DocumentQualityScorer()
        logger.info("CacheManager initialized. Call internally _load to load file doc_cache.json which store pypdf value ")
        self.cache = CacheManager()

        # 🔥 Parsers
        self.unstructured_parser = UnstructuredParser()

        if getattr(Config, "USE_AZURE_DOC_INTELLIGENCE", False):
            self.azure_parser = AzureDocIntelligence()
        else:
            self.azure_parser = None

    # =========================================================
    # 🔥 PARSER EXECUTOR (IMPROVED)
    # =========================================================
    def _run_parser(self, parser, file_path):

        try:
            if parser == "pypdf":
                loader = PyPDFLoader(file_path)
                docs = loader.load()

            elif parser == "unstructured":
                docs = self.unstructured_parser.parse(file_path)

            elif parser == "azure" and self.azure_parser:
                docs = self.azure_parser.parse(file_path)

            else:
                return None

            # 🔥 Normalize metadata
            for d in docs:
                d.metadata["parser"] = parser
                d.metadata["source"] = file_path

            return docs

        except Exception as e:
            logger.info(f"Parser {parser} failed: {e}")
            return None

    def load_single_document(self, file_path):
        """
        Load ONLY one document (used for upload)
        """

        from langchain_community.document_loaders import PyPDFLoader

        file_path = str(file_path)

        # You can still use ParserRouter here if needed
        loader = PyPDFLoader(file_path)

        docs = loader.load()

        return docs

    # =========================================================
    # 🔥 LOAD DOCUMENTS (SMART PIPELINE)
    # =========================================================
    def load_documents(self, directory=None) -> List[Document]:

        if directory is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            directory = base_dir / "data"
            logger.info(f"directory specified, defaulting to: {directory}")

        path = Path(directory)
        logger.info(f"Loading documents from: {path}")

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            return []

        docs = []

        for file in path.glob("*.pdf"):
            
            file_path = str(file)
            logger.info("\n Processing: {file_path}")
        
            # =====================================================
            # 🔥 STEP 1: CACHE CHECK
            # =====================================================
            cached_parser = self.cache.get(file_path)
            logger.info(f"Cache check for {file_path}: {'HIT' if cached_parser else 'MISS'}")

            if cached_parser:
                logger.info(f"Using cached parser: {cached_parser}")

                parsed_docs = self._run_parser(cached_parser, file_path)
                logger.info(f"Cached parser {cached_parser} returned {len(parsed_docs) if parsed_docs else 0} docs for file {file_path} ")

                if parsed_docs:
                    docs.extend(parsed_docs)
                    continue

            # =====================================================
            # 🔥 STEP 2: GET PARSER ORDER
            # =====================================================
            parser_order = self.router.get_parser_order(file_path)

            best_docs = None
            best_score = 0
            best_parser = None

            # =====================================================
            # 🔥 STEP 3: TRY MULTIPLE PARSERS
            # =====================================================
            for parser in parser_order:

                logger.info(f"Trying: {parser}")

                parsed_docs = self._run_parser(parser, file_path)

                if not parsed_docs:
                    continue

                # 🔥 QUALITY SCORING (VERY IMPORTANT)
                score = self.quality.score(parsed_docs)

                logger.info(f"Score ({parser}): {score:.2f}")

                if score > best_score:
                    best_docs = parsed_docs
                    best_score = score
                    best_parser = parser

                # 🔥 EARLY EXIT (performance + cost saving)
                if score > 0.85:
                    logger.info(" High quality achieved, stopping early")
                    break

            # =====================================================
            # 🔥 STEP 4: LEARN + CACHE
            # =====================================================
            if best_parser:
                self.router.update_learning(file_path, best_parser)
                self.cache.set(file_path, best_parser)

            docs.extend(best_docs or [])

        logger.info(f"\n Total documents loaded: {len(docs)}")
        return docs

    # =========================================================
    # 🔥 STRUCTURE DETECTION
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
    # 🔥 SPLIT DOCUMENTS
    # =========================================================
    def split_documents(self, docs: List[Document]) -> List[Document]:

        all_chunks = []

        for doc in docs:

            text = doc.page_content
            source = doc.metadata.get("source", "unknown")

            # 🔥 TABLE
            if doc.metadata.get("type") == "table":

                chunks = self.splitter.split_text(text)

                chunks = [
                    Document(
                        page_content=c,
                        metadata={
                            "source": source,
                            "type": "table"
                        }
                    )
                    for c in chunks
                ]

            # 🔥 LEGAL STRUCTURE
            elif self._has_structure(text):

                chunks = self.chunker.process(text, source)

            # 🔥 DEFAULT
            else:
                chunks = self.splitter.split_documents([doc])

            all_chunks.extend(chunks)

        logger.info(f"Total chunks: {len(all_chunks)}")

        return all_chunks

    # =========================================================
    # 🔥 MAIN PIPELINE
    # =========================================================
    '''
    def process(self, directory="data"):

        docs = self.load_documents(directory)
        chunks = self.split_documents(docs)

        logger.info(f"Loaded {len(docs)} docs")
        logger.info("Created {len(chunks)} chunks")

        # 🔥 Optional graph integration
        if hasattr(self, "graph_builder") and self.graph_builder:
            logger.info(" Building Knowledge Graph...")
            self.graph_builder.process_batch(chunks)

        return chunks
    '''

    def process(self, directory="data", return_raw=False):

        raw_docs = self.load_documents(directory)
        chunks = self.split_documents(raw_docs)

        logger.info("Loaded {len(raw_docs)} docs")
        logger.info("Created {len(chunks)} chunks")

        if return_raw:
            return raw_docs, chunks

        return chunks
