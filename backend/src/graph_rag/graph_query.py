import re
import json
import os
import numpy as np
from backend.src.utils.logger import get_logger

logger = get_logger(__name__)


class GraphQueryEngine:

    def __init__(self, graph_store, memory=None, redis_client=None):

        self.graph = graph_store
        self.memory = memory
        self.redis = redis_client

        # 🔥 Load embedding model ONCE
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # 🔥 FAISS DATA
        self.materials = graph_store.data.get("materials", [])
        self.locations = graph_store.data.get("locations", [])

        self.material_index = graph_store.data.get("material_index")
        self.location_index = graph_store.data.get("location_index")

        # 🔥 Pattern learning
        self.pattern_file = "patterns.json"
        self.patterns = self._load_patterns()

        logger.info("⚡ GraphQueryEngine ready (FAISS + Redis + Explainability)")

    # =========================================================
    # 🔥 REDIS CACHE
    # =========================================================
    def _cache_get(self, key):
        if not self.redis:
            return None
        try:
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except:
            return None

    def _cache_set(self, key, value):
        if not self.redis:
            return
        try:
            self.redis.setex(key, 3600, json.dumps(value))
        except:
            pass

    # =========================================================
    # 🔥 PATTERN STORE
    # =========================================================
    def _load_patterns(self):
        if os.path.exists(self.pattern_file):
            return json.load(open(self.pattern_file))
        return {"materials": [], "locations": []}

    def _save_patterns(self):
        json.dump(self.patterns, open(self.pattern_file, "w"), indent=2)

    def _learn_pattern(self, material, location):
        updated = False

        if material and material not in self.patterns["materials"]:
            self.patterns["materials"].append(material)
            updated = True

        if location and location not in self.patterns["locations"]:
            self.patterns["locations"].append(location)
            updated = True

        if updated:
            self._save_patterns()

    # =========================================================
    # 🔥 RULE-BASED EXTRACTION
    # =========================================================
    def _extract_rule(self, question):

        material = None
        location = None

        mat = re.search(r"M\d+\s*Concrete", question, re.IGNORECASE)
        if mat:
            material = mat.group(0)

        loc = re.search(
            r"(Mumbai|Delhi|Pune|Bangalore|Maharashtra)",
            question,
            re.IGNORECASE
        )
        if loc:
            location = loc.group(0)

        return material, location

    # =========================================================
    # 🔥 DYNAMIC PATTERN
    # =========================================================
    def _extract_dynamic(self, question):

        material = None
        location = None

        for p in self.patterns["materials"]:
            if p.lower() in question.lower():
                material = p

        for p in self.patterns["locations"]:
            if p.lower() in question.lower():
                location = p

        return material, location

    # =========================================================
    # 🔥 FAISS SEARCH (CORE)
    # =========================================================
    def _search_faiss(self, index, items, query_vec):

        if index is None or len(items) == 0:
            return None

        D, I = index.search(np.array([query_vec]), 1)

        idx = I[0][0]

        if idx < len(items):
            return items[idx]

        return None

    # =========================================================
    # 🔥 MAIN QUERY
    # =========================================================
    def query(self, question, session_id=None, rag_results=None):

        logger.info("📊 GraphQueryEngine running...")

        # =====================================================
        # 🔥 CACHE
        # =====================================================
        cache_key = f"graph:{question}"

        cached = self._cache_get(cache_key)
        if cached:
            logger.info("⚡ Cache hit")
            return cached

        # =====================================================
        # 🔥 STEP 1: RULE
        # =====================================================
        material, location = self._extract_rule(question)

        # =====================================================
        # 🔥 STEP 2: PATTERN
        # =====================================================
        if not material and not location:
            material, location = self._extract_dynamic(question)

        # =====================================================
        # 🔥 STEP 3: FAISS (MAIN ENGINE)
        # =====================================================
        if not material and not location:

            q_vec = self.model.encode([question])[0].astype("float32")

            material = self._search_faiss(
                self.material_index,
                self.materials,
                q_vec
            )

            location = self._search_faiss(
                self.location_index,
                self.locations,
                q_vec
            )

        logger.info(f"Extracted → material={material}, location={location}")

        # 🔥 Learn patterns automatically
        self._learn_pattern(material, location)

        # =====================================================
        # 🔥 GRAPH SCORING + EXPLAINABILITY
        # =====================================================
        scores = {}
        explanations = {}

        if material:
            for p in self.graph.get_projects_using_material(material):
                scores[p] = scores.get(p, 0) + 2
                explanations.setdefault(p, []).append(f"uses {material}")

        if location:
            for p in self.graph.get_projects_by_location(location):
                scores[p] = scores.get(p, 0) + 1
                explanations.setdefault(p, []).append(f"located in {location}")

        # 🔥 HYBRID BOOST
        if rag_results:
            for r in rag_results:
                scores[r] = scores.get(r, 0) + 1
                explanations.setdefault(r, []).append("matched in documents")

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

        results = [
            f"{p} → {', '.join(explanations.get(p, []))}"
            for p, _ in ranked[:10]
        ]

        # =====================================================
        # 🔥 CACHE STORE
        # =====================================================
        self._cache_set(cache_key, results)

        return results if results else ["No matching data found."]
