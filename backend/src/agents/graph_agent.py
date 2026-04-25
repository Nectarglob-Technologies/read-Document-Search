from backend.src.config.config import Config
from backend.src.utils.logger import get_logger

import difflib
import hashlib

logger = get_logger(__name__)


class GraphAgent:

    def __init__(self, llm, query_engine, memory=None):
        self.llm = llm
        self.query_engine = query_engine
        self.memory = memory
        self.llm_cache = {}

    # =========================================================
    # 🔹 HASH (CACHE KEY)
    # =========================================================
    def _get_hash(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    # =========================================================
    # 🔹 MEMORY CONTEXT (ROBUST)
    # =========================================================
    def _get_memory_context(self, session_id):

        if not self.memory or not session_id:
            return ""

        try:
            summary = self.memory.get_summary(session_id)
            recent = self.memory.get_recent_history(session_id)

            recent_text = ""

            for msg in recent:
                if isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                else:
                    role = "user"
                    content = str(msg)

                if role == "user":
                    recent_text += f"User: {content}\n"
                else:
                    recent_text += f"Assistant: {content}\n"

            return f"""
Summary:
{summary}

Recent:
{recent_text}
"""

        except Exception as e:
            logger.warning(f"Memory context failed: {e}")
            return ""

    # =========================================================
    # 🔹 FILTER RESULTS (RESTORED)
    # =========================================================
    def _filter_results(self, question, results):

        if not results:
            return results

        filtered = []

        for r in results:
            sim = difflib.SequenceMatcher(
                None,
                question.lower(),
                str(r).lower()
            ).ratio()

            if sim > 0.2:
                filtered.append(r)

        return filtered if filtered else results[:3]

    # =========================================================
    # 🔹 RANK RESULTS (RESTORED - IMPORTANT)
    # =========================================================
    def _rank_results(self, question, results, memory_context):

        ranked = []

        for r in results:
            text = str(r).lower()
            score = 0

            # question similarity
            score += difflib.SequenceMatcher(
                None, question.lower(), text
            ).ratio() * 0.6

            # memory boost
            if memory_context:
                score += difflib.SequenceMatcher(
                    None, memory_context.lower(), text
                ).ratio() * 0.3

            # domain boost
            if any(k in text for k in ["project", "bridge", "road", "concrete"]):
                score += 0.1

            ranked.append((r, score))

        ranked.sort(key=lambda x: x[1], reverse=True)

        return [r[0] for r in ranked]

    # =========================================================
    # 🔹 FORMAT OUTPUT (CLEAN)
    # =========================================================
    def _format_output(self, results):

        if not results:
            return ""

        return "\n".join([f"- {str(r)}" for r in results[:10]])

    # =========================================================
    # 🔹 CONFIDENCE (RESTORED FULL)
    # =========================================================
    def _calculate_confidence(self, question, results):

        if not results:
            return 0.0

        scores = []

        for r in results[:10]:
            sim = difflib.SequenceMatcher(
                None,
                question.lower(),
                str(r).lower()
            ).ratio()

            scores.append(sim)

        avg = sum(scores) / len(scores)

        # volume boost
        boost = min(0.3, len(results) * 0.05)

        return round(min(1.0, avg + boost), 2)

    # =========================================================
    # 🔹 MAIN (STABLE + FULL FEATURE)
    # =========================================================
    def run(self, question, session_id=None):

        logger.info("📊 GraphAgent running...")

        # ---------------- MEMORY ----------------
        memory_context = self._get_memory_context(session_id)

        enhanced_query = f"{memory_context}\nQuestion: {question}"

        # ---------------- QUERY ----------------
        results = self.query_engine.query(enhanced_query)

        if not results:
            return {
                "answer": "No graph results found.",
                "confidence": 0.0,
                "source": "graph"
            }

        # ---------------- FILTER ----------------
        results = self._filter_results(question, results)

        # ---------------- RANK ----------------
        results = self._rank_results(question, results, memory_context)

        # ---------------- FORMAT ----------------
        formatted = self._format_output(results)

        # ---------------- CONFIDENCE ----------------
        confidence = self._calculate_confidence(question, results)

        logger.info(f"📊 Graph Confidence: {confidence}")

        # =====================================================
        # 🔥 FAST MODE (NO LLM)
        # =====================================================
        if (
            Config.ZERO_LLM_MODE
            or not Config.USE_LLM_FOR_GRAPH
            or not self.llm
            or confidence > Config.CONFIDENCE_THRESHOLD
        ):
            return {
                "answer": f"📊 Graph Answer:\n\n{formatted}",
                "confidence": confidence,
                "source": "graph"
            }

        # =====================================================
        # 🔥 CACHE
        # =====================================================
        cache_key = self._get_hash(question + formatted)

        if cache_key in self.llm_cache:
            logger.info("⚡ Using cached graph response")
            response = self.llm_cache[cache_key]

        else:
            prompt = f"""
You are a civil engineering expert.

STRICT RULES:
- Use ONLY graph data
- DO NOT hallucinate

------------------------
Graph Data:
{formatted}

------------------------
Question:
{question}

Return:
- Clear Answer
- Key Facts
- Short Explanation
"""

            try:
                response = self.llm.invoke(prompt).content
                self.llm_cache[cache_key] = response
            except Exception as e:
                logger.error(f"LLM failed: {e}")
                return {
                    "answer": f"📊 Graph Answer:\n\n{formatted}",
                    "confidence": confidence,
                    "source": "graph"
                }

        # ---------------- FINAL ----------------
        return {
            "answer": f"📊 Graph-Based Answer:\n\n{response}",
            "confidence": confidence,
            "source": "graph"
        }
