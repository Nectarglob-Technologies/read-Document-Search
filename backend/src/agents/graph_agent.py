from backend.src.config.config import Config
import difflib


class GraphAgent:

    def __init__(self, llm, query_engine):

        self.llm = llm
        self.query_engine = query_engine

    # ---------------- FORMAT ----------------
    def _format_graph_output(self, results):

        if not results:
            return ""

        if isinstance(results, list):
            return "\n".join([f"- {str(r)}" for r in results])

        return str(results)

    # ---------------- CONFIDENCE ----------------
    def _calculate_confidence(self, question, results):

        if not results:
            return 0.0

        scores = []

        for r in results:
            similarity = difflib.SequenceMatcher(
                None, question.lower(), str(r).lower()
            ).ratio()

            scores.append(similarity)

        # average similarity
        avg_score = sum(scores) / len(scores)

        # boost if multiple results
        volume_boost = min(0.3, len(results) * 0.05)

        return min(1.0, avg_score + volume_boost)

    # ---------------- MAIN ----------------
    def run(self, question):

        print("📊 Querying graph...")

        results = self.query_engine.query(question)

        if not results:
            return "No graph results found.\nConfidence: 0.0"

        formatted = self._format_graph_output(results)
        confidence = self._calculate_confidence(question, results)

        print(f"Graph confidence: {confidence:.2f}")

        # ---------------- ZERO LLM MODE ----------------
        if (
            Config.ZERO_LLM_MODE
            or not Config.USE_LLM_FOR_GRAPH
            or not self.llm
            or confidence > Config.CONFIDENCE_THRESHOLD
        ):

            return f"""
📊 Graph Answer (Zero-LLM):

{formatted}

Confidence: {confidence:.2f}
"""

        # ---------------- LLM ENHANCED ----------------
        prompt = f"""
You are a civil engineering expert.

STRICT RULES:
- Use ONLY the provided graph data
- DO NOT add external knowledge
- DO NOT hallucinate

------------------------

Graph Data:
{formatted}

------------------------

Question:
{question}

------------------------

Return:
1. Clear Answer
2. Key Facts (bullet points)
3. Short Explanation
"""

        try:
            response = self.llm.invoke(prompt).content
        except Exception as e:
            print("LLM failed, fallback to raw output:", e)
            return f"""
📊 Graph Answer (Fallback):

{formatted}

Confidence: {confidence:.2f}
"""

        return f"""
📊 Graph-Based Answer:

{response}

Confidence: {confidence:.2f}
"""
