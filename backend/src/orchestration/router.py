import hashlib
import json
from backend.src.utils.logger import get_logger

logger = get_logger(__name__)


class QueryRouter:

    def __init__(self, llm, redis_client=None):
        self.llm = llm
        self.redis = redis_client

    # =========================================================
    # 🔥 CACHE
    # =========================================================
    def _cache_get(self, key):
        if not self.redis:
            return None
        try:
            val = self.redis.get(key)
            return val.decode() if val else None
        except:
            return None

    def _cache_set(self, key, value):
        if not self.redis:
            return
        try:
            self.redis.setex(key, 3600, value)
        except:
            pass

    # =========================================================
    # 🔥 FAST CLASSIFIER (RULE-BASED WITH CONFIDENCE)
    # =========================================================
    def _fast_classify(self, question):

        q = question.lower()

        score = 0
        intent = "rag"

        if any(k in q for k in ["material", "cement", "steel", "concrete"]):
            intent = "graph"
            score += 1

        if any(k in q for k in ["location", "city", "site"]):
            intent = "graph"
            score += 1

        if any(k in q for k in ["project", "contractor"]):
            intent = "graph"
            score += 1

        if any(k in q for k in ["why", "how", "impact", "reason"]):
            score += 1
            if intent == "graph":
                intent = "hybrid"

        confidence = min(score / 3, 1.0)

        logger.info(f"⚡ Fast classify → intent={intent}, confidence={confidence:.2f}")

        return intent, confidence

    # =========================================================
    # 🔥 LLM ROUTER (STRICT + ZERO HALLUCINATION)
    # =========================================================
    def _llm_route(self, question):

        prompt = f"""
You are a strict classifier.

Classify the query into ONLY ONE category:

- graph → structured data (projects, materials, locations, relationships)
- rag → document-based explanation or general knowledge
- hybrid → requires BOTH structured + explanation

Rules:
- Output ONLY one word: graph OR rag OR hybrid
- No explanation
- No sentence
- No punctuation

Query:
{question}
"""

        try:
            res = self.llm.invoke(prompt).content.strip().lower()

            if res not in ["graph", "rag", "hybrid"]:
                logger.warning(f"⚠️ Invalid LLM output: {res}")
                return "rag"

            logger.info(f"🧠 LLM classify → {res}")

            return res

        except Exception as e:
            logger.error(f"LLM router failed: {e}")
            return "rag"

    # =========================================================
    # 🔥 MAIN ROUTE FUNCTION
    # =========================================================
    def route(self, question: str) -> str:

        cache_key = f"router:{hashlib.md5(question.encode()).hexdigest()}"

        cached = self._cache_get(cache_key)
        if cached:
            logger.info(f" Router cache hit → {cached}")
            return cached

        # STEP 1 → FAST CLASSIFIER
        intent, confidence = self._fast_classify(question)

        # STEP 2 → IF CONFIDENT, RETURN
        if confidence >= 0.7:
            self._cache_set(cache_key, intent)
            return intent

        # STEP 3 → LLM FALLBACK
        intent = self._llm_route(question)

        self._cache_set(cache_key, intent)

        return intent
