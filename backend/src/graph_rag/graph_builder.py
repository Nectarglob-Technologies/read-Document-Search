from backend.src.graph_rag.graph_store import GraphStore
from backend.src.graph_rag.graph_extractor import GraphExtractor
from backend.src.graph_rag.graph_extractor_fast import FastGraphExtractor
from backend.src.utils.threshold_optimizer import ThresholdOptimizer

# 🔥 Optional table extractor
try:
    from backend.src.graph_rag.table_graph_extractor import TableGraphExtractor
    TABLE_EXTRACTION_AVAILABLE = True
except:
    TABLE_EXTRACTION_AVAILABLE = False

import re
import hashlib

from backend.src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphBuilder:

    def __init__(self, llm=None):

        self.fast_extractor = FastGraphExtractor()
        self.llm_extractor = GraphExtractor(llm) if llm else None
        self.table_extractor = TableGraphExtractor() if TABLE_EXTRACTION_AVAILABLE else None

        self.store = GraphStore()

        self.seen_entities = set()
        self.seen_relations = set()

        self.processed_hashes = set()
        self.llm_cache = {}

        self.optimizer = ThresholdOptimizer()

    # =========================================================
    def _get_hash(self, text):
        return hashlib.md5(text.strip().encode()).hexdigest()

    # =========================================================
    def _clean_text(self, text):
        if not text:
            return ""
        return str(text).strip().replace("\n", " ").replace("  ", " ")

    # =========================================================
    def _add_entity_safe(self, name, entity_type):

        name = self._clean_text(name)

        if not name or len(name) < 2:
            return

        key = (name.lower(), entity_type)

        if key not in self.seen_entities:
            self.store.add_entity(name, entity_type)
            self.seen_entities.add(key)

    # =========================================================
    def _add_relation_safe(self, source, relation, target):

        source = self._clean_text(source)
        target = self._clean_text(target)

        if not source or not target or not relation:
            logger.info(f"Skipping invalid relation: {source}, {relation}, {target}")
            return

        relation = relation.upper()

        key = (source.lower(), relation.lower(), target.lower())

        if key not in self.seen_relations:
            self.store.add_relation(source, relation, target)
            self.seen_relations.add(key)

    # =========================================================
    def _should_use_llm(self, data, text):

        if not self.llm_extractor:
            return False

        text_len = len(text)

        project_threshold = self.optimizer.get_project_threshold()
        relation_threshold = self.optimizer.get_relation_threshold()

        if not data.get("projects") and text_len > project_threshold:
            logger.info(f"LLM Decision: No projects (threshold={project_threshold})")
            return True

        if len(data.get("relations", [])) == 0 and text_len > relation_threshold:
            logger.info(f"LLM Decision: No relations (threshold={relation_threshold})")
            return True

        if (
            not data.get("projects") and
            not data.get("materials") and
            not data.get("relations") and
            text_len > project_threshold
        ):
            logger.info("LLM Decision: Weak extraction")
            return True

        return False

    # =========================================================
    def process_document(self, text, metadata=None):

        if not text or len(text.strip()) < 50:
            return

        doc_hash = self._get_hash(text)

        if doc_hash in self.processed_hashes:
            return

        self.processed_hashes.add(doc_hash)

        metadata = metadata or {}
        doc_type = metadata.get("type")

        logger.info(f"Processing doc (type={doc_type}) | hash={doc_hash[:8]}")

        # =====================================================
        # TABLE HANDLING
        # =====================================================
        if doc_type == "table" and self.table_extractor:

            relations = self.table_extractor.extract(text)
            logger.info(f"Table relations found: {len(relations)}")

            for r in relations:
                self._add_entity_safe(r.get("project"), "project")
                self._add_entity_safe(r.get("entity"), "entity")
                self._add_relation_safe(r.get("project"), r.get("relation"), r.get("entity"))

            return

        # =====================================================
        # FAST EXTRACTION
        # =====================================================
        data = self.fast_extractor.extract(text)

        logger.info(
            f"Fast → projects:{len(data.get('projects', []))}, "
            f"materials:{len(data.get('materials', []))}, "
            f"relations:{len(data.get('relations', []))}"
        )

        # =====================================================
        # 🔥 NEW: DECISION BLOCK (CLEAN)
        # =====================================================
        use_llm = False

        if self.llm_extractor:
            use_llm = self._should_use_llm(data, text)

        text_len = len(text)

        logger.info(
            f"LLM Decision | text_len={text_len} | "
            f"projects={len(data.get('projects', []))} | "
            f"materials={len(data.get('materials', []))} | "
            f"relations={len(data.get('relations', []))} | "
            f"use_llm={use_llm}"
        )

        # =====================================================
        # LLM EXECUTION
        # =====================================================
        if use_llm:

            logger.info("LLM triggered...")

            if doc_hash in self.llm_cache:
                llm_data = self.llm_cache[doc_hash]
                logger.info("Using cached LLM result")

            else:
                try:
                    llm_data = self.llm_extractor.extract(text)

                    success = any([
                        llm_data.get("projects"),
                        llm_data.get("materials"),
                        llm_data.get("relations")
                    ])

                    self.optimizer.record_llm_result(success)

                    if not success:
                        logger.info("LLM returned empty → skipping")
                        llm_data = {}
                    else:
                        logger.info(
                            f"LLM → projects:{len(llm_data.get('projects', []))}, "
                            f"materials:{len(llm_data.get('materials', []))}, "
                            f"relations:{len(llm_data.get('relations', []))}"
                        )

                    self.llm_cache[doc_hash] = llm_data

                except Exception as e:
                    logger.error(f"LLM failed: {e}")
                    llm_data = {}

            # MERGE
            for key in ["projects", "materials", "locations", "contractors"]:
                data[key] = list(set(data.get(key, []) + llm_data.get(key, [])))

            data["relations"] = data.get("relations", []) + llm_data.get("relations", [])

        # =====================================================
        # STORE ENTITIES
        # =====================================================
        for p in data.get("projects", []):
            self._add_entity_safe(p, "project")

        for c in data.get("contractors", []):
            self._add_entity_safe(c, "contractor")

        for m in data.get("materials", []):
            self._add_entity_safe(m, "material")

        for l in data.get("locations", []):
            self._add_entity_safe(l, "location")

        # =====================================================
        # STORE RELATIONS
        # =====================================================
        for r in data.get("relations", []):

            if not r.get("project") or not r.get("entity"):
                logger.info(f"Skipping invalid relation: {r}")
                continue

            self._add_relation_safe(
                r.get("project"),
                r.get("relation"),
                r.get("entity")
            )

        # =====================================================
        # CLAUSE LINKING
        # =====================================================
        for link in self._link_clauses(text):
            self._add_relation_safe(
                link["source"],
                link["relation"],
                link["target"]
            )

    # =========================================================
    def _link_clauses(self, text):

        links = []

        patterns = [
            (r"Clause\s+(\d+).*?Clause\s+(\d+)", "REFERS_TO")
        ]

        for pattern, relation in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)

            for src, tgt in matches:
                links.append({
                    "source": f"Clause {src}",
                    "relation": relation,
                    "target": f"Clause {tgt}"
                })

        return links

    # =========================================================
    def process_batch(self, docs):

        logger.info("Processing batch...")

        for doc in docs:

            if hasattr(doc, "page_content"):
                self.process_document(
                    doc.page_content,
                    metadata=getattr(doc, "metadata", {})
                )
            else:
                self.process_document(doc)

    # =========================================================
    def get_graph(self):
        return self.store
