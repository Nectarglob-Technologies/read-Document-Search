from backend.src.retrieval.query_rewriter import QueryRewriter
from backend.src.retrieval.hybrid_retriever import HybridRetriever
from backend.src.retrieval.reranker import Reranker
from backend.src.config.config import Config
import difflib
import re


class EngineeringAgent:

    def __init__(
        self,
        llm,
        retriever,
        graph_agent=None,
        clause_store=None
    ):

        self.llm = llm

        # ✅ EXISTING
        self.rewriter = QueryRewriter(llm)
        self.retriever = HybridRetriever(retriever)
        self.reranker = Reranker()

        # ✅ NEW
        self.graph_agent = graph_agent
        self.clause_store = clause_store

    # ---------------- CONFIDENCE ----------------
    def _calculate_confidence(self, query, docs):

        if not docs:
            return 0

        scores = []

        for doc in docs:
            similarity = difflib.SequenceMatcher(
                None, query.lower(), doc.page_content.lower()
            ).ratio()

            scores.append(similarity)

        return sum(scores) / len(scores)

    # ---------------- GRAPH CONFIDENCE PARSER ----------------
    def _extract_graph_confidence(self, text):

        if not text:
            return 0.0

        match = re.search(r"confidence[:\s]+(\d+\.?\d*)", text.lower())

        if match:
            try:
                return float(match.group(1))
            except:
                return 0.6

        return 0.6

    # ---------------- CLAUSE RETRIEVAL ----------------
    def _get_clause_context(self, question):

        if not self.clause_store:
            return []

        return self.clause_store.search(question)

    # ---------------- ZERO LLM ANSWER ----------------
    def _extract_direct_answer(self, docs):

        best_doc = docs[0]

        return f"""
            📄 Direct Answer (Fast Mode):

            {best_doc.page_content[:500]}

            📌 Source: {best_doc.metadata.get("source", "unknown")}

            Confidence: 0.8
            """
    
    def _boost_clause_docs(self, docs, question):

        if not docs:
            return docs

        boosted = []

        for doc in docs:

            score = 0

            # 🔥 Boost if clause metadata exists
            if "clauses" in doc.metadata:
                score += 0.2

            # 🔥 Boost if clause matches query
            for clause in doc.metadata.get("clauses", []):
                if clause in question:
                    score += 0.3

            boosted.append((doc, score))

        # sort by score
        boosted.sort(key=lambda x: x[1], reverse=True)

        return [d[0] for d in boosted]


    # ---------------- MAIN ----------------
    def run(self, question):

        print("⚙️ EngineeringAgent running...")

        # 🔥 STEP 1 — QUERY REWRITE
        if Config.ZERO_LLM_MODE:
            improved_query = question
        else:
            try:
                improved_query = self.rewriter.rewrite(question)
            except:
                improved_query = question

        # 🔥 STEP 2 — RETRIEVE
        docs = self.retriever.retrieve(improved_query)

        # 🔥 STEP 3 — RERANK
        top_docs = self.reranker.rerank(improved_query, docs)

        # 🔥 STEP 3.5 — BOOST CLAUSE DOCS
        top_docs = self._boost_clause_docs(top_docs, question)

        if not top_docs:
            return "No relevant documents found."

        # 🔥 STEP 4 — RAG CONFIDENCE
        rag_conf = self._calculate_confidence(improved_query, top_docs)

        print(f"RAG Confidence: {rag_conf:.2f}")

        # 🔥 STEP 5 — GRAPH
        graph_ans = ""
        graph_conf = 0.0

        if self.graph_agent:
            graph_ans = self.graph_agent.run(question)
            graph_conf = self._extract_graph_confidence(graph_ans)

        print(f"Graph Confidence: {graph_conf:.2f}")

        # 🔥 STEP 6 — CLAUSE CONTEXT
        clause_context = self._get_clause_context(question)

        # =========================================================
        # 🔥 STEP 7 — FAST MODE (NO LLM)
        # =========================================================
        if (
            Config.ZERO_LLM_MODE
            or not Config.USE_LLM_FOR_ANSWER
            or not self.llm
        ):

            # Prefer RAG if confident
            if rag_conf >= Config.CONFIDENCE_THRESHOLD:
                return self._extract_direct_answer(top_docs)

            # Otherwise hybrid fallback
            return f"""
🔀 Hybrid Answer (Fast Mode)

📊 Graph:
{graph_ans}

📄 Top Document:
{top_docs[0].page_content[:300]}

Confidence: {max(rag_conf, graph_conf):.2f}
"""

        # =========================================================
        # 🔥 STEP 8 — HIGH CONFIDENCE → SKIP LLM
        # =========================================================
        if rag_conf >= Config.CONFIDENCE_THRESHOLD:
            return self._extract_direct_answer(top_docs)

        # =========================================================
        # 🔥 STEP 9 — GRAPH DOMINANT
        # =========================================================
        if graph_conf > rag_conf + 0.2:
            return f"""
📊 Graph-Based Answer (High Confidence)

{graph_ans}

Confidence: {graph_conf:.2f}
"""

        # =========================================================
        # 🔥 STEP 10 — PREPARE CONTEXT
        # =========================================================
        context = []
        sources = []

        for i, doc in enumerate(top_docs[:5]):
            source = doc.metadata.get("source", "unknown")

            context.append(f"[{i+1}] {doc.page_content}")
            sources.append(f"[{i+1}] {source}")

        context_text = "\n\n".join(context)

        # =========================================================
        # 🔥 STEP 11 — HYBRID LLM (BEST QUALITY)
        # =========================================================
        prompt = f"""
You are a senior civil engineer.

Use ALL sources below:

------------------------
Graph Data:
{graph_ans}

------------------------
Clause Context:
{clause_context}

------------------------
Documents:
{context_text}

------------------------
Question:
{question}

------------------------

Return:
1. Final Answer
2. Technical Explanation
3. Referenced Clauses (if any)
4. Sources
5. Confidence: <0-1>
"""

        try:
            response = self.llm.invoke(prompt).content
        except Exception as e:
            print("LLM failed, fallback:", e)
            return self._extract_direct_answer(top_docs)

        return f"""
{response}

📊 RAG Confidence: {rag_conf:.2f}
📊 Graph Confidence: {graph_conf:.2f}

📚 Sources:
{chr(10).join(sources)}
"""
