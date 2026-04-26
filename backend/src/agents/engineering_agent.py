import difflib
import hashlib
import os
import re

from backend.src.retrieval.query_rewriter import QueryRewriter
from backend.src.retrieval.hybrid_retriever import HybridRetriever
from backend.src.retrieval.reranker import Reranker
from backend.src.config.config import Config
from backend.src.utils.logger import get_logger

logger = get_logger(__name__)


class EngineeringAgent:

    def __init__(self, llm, retriever, graph_agent=None, clause_store=None, memory=None):
        self.llm = llm
        self.rewriter = QueryRewriter(llm)
        self.retriever = HybridRetriever(retriever)
        self.reranker = Reranker()

        self.graph_agent = graph_agent
        self.clause_store = clause_store
        self.memory = memory

        self.llm_cache = {}

    # =========================================================
    # 🔹 HASH
    # =========================================================
    def _get_hash(self, text):
        return hashlib.md5(text.encode()).hexdigest()

    # =========================================================
    # 🔹 REAL PAGE FIX (🔥 CRITICAL)
    # =========================================================
    def _extract_real_page(self, text, fallback):
        #logger.info(f"Extraction text '{text}'")
        match = re.search(r"\b(\d{3})\b", text)
        #logger.info(f"Extracting Match from text: '{match}'")
        logger.info(f"Extracted page from text: {match.group(1) if match else 'None'} | Fallback: {fallback}")  
        return match.group(1) if match else fallback

    # =========================================================
    # 🔹 CONFIDENCE (HYBRID 🔥)
    # =========================================================
    def _calculate_confidence(self, query, docs):

        if not docs:
            return 0.0

        keyword_scores = []
        difflib_scores = []

        keywords = [w.lower() for w in query.split() if len(w) > 3]

        for doc in docs[:5]:
            text = doc.page_content.lower()

            # keyword score
            hits = sum(1 for k in keywords if k in text)
            keyword_scores.append(hits / max(len(keywords), 1))

            # difflib score
            difflib_scores.append(
                difflib.SequenceMatcher(None, query.lower(), text).ratio()
            )

        final_score = (
            (sum(keyword_scores) / len(keyword_scores)) * 0.7 +
            (sum(difflib_scores) / len(difflib_scores)) * 0.3
        )

        return round(final_score, 2)

    # =========================================================
    # 🔹 MEMORY CONTEXT (RESTORED)
    # =========================================================
    def _get_memory_context(self, session_id):

        if not self.memory or not session_id:
            return ""

        try:
            summary = self.memory.get_summary(session_id)
            recent = self.memory.get_recent_history(session_id, limit=5)

            text = ""
            for msg in recent:
                role = msg.get("role")
                content = msg.get("content")

                text += f"{role.capitalize()}: {content}\n"

            logger.info(f"Memory summary: {summary}")
            logger.info(f"Memory recent:\n{text}")

            return f"Summary:\n{summary}\n\nRecent:\n{text}"

        except Exception as e:
            logger.warning(f"Memory fallback: {e}")
            return ""

    # =========================================================
    # 🔹 CONTEXT RANKING (IMPROVED 🔥)
    # =========================================================
    def _rank_context(self, question, docs, memory_context):

        keywords = [w.lower() for w in question.split() if len(w) > 3]

        ranked = []

        for doc in docs:
            text = doc.page_content.lower()

            score = 0

            # keyword boost
            score += sum(1 for k in keywords if k in text) * 0.6

            # memory boost
            if memory_context:
                score += difflib.SequenceMatcher(
                    None, memory_context.lower(), text
                ).ratio() * 0.2

            # clause boost
            if "clauses" in doc.metadata:
                score += 0.2

            ranked.append((doc, score))

        ranked.sort(key=lambda x: x[1], reverse=True)

        return [d[0] for d in ranked]

    # =========================================================
    # 🔹 FORMAT SOURCE (FIXED)
    # =========================================================
    def _format_source(self, doc, idx):

        file_name = os.path.basename(doc.metadata.get("source", "unknown"))

        raw_page = doc.metadata.get("page_label") or doc.metadata.get("page")
        real_page = self._extract_real_page(doc.page_content, raw_page)

        return f"[{idx}] {file_name} (Page {real_page})"

    # =========================================================
    # 🔹 MAIN
    # =========================================================
    def run(self, question, session_id=None):

        logger.info("⚙️ EngineeringAgent running")

        # ---------------- QUERY ----------------
        if Config.ZERO_LLM_MODE:
            improved_query = question
        else:
            try:
                improved_query = self.rewriter.rewrite(question)
            except Exception as e:
                logger.warning(f"Rewrite failed: {e}")
                improved_query = question

        improved_query = f"{improved_query} {question}"

        logger.info(f"Query: {improved_query}")

        # ---------------- RETRIEVE ----------------
        docs = self.retriever.retrieve(improved_query)

        # ---------------- RERANK ----------------
        top_docs = self.reranker.rerank(improved_query, docs)

        # 🔥 MEMORY + RANKING
        memory_context = self._get_memory_context(session_id)
        top_docs = self._rank_context(question, top_docs, memory_context)

        if not top_docs:
            return {
                "answer": "No relevant documents found.",
                "docs": [],
                "confidence": 0.0,
                "rag_conf": 0.0,
                "graph_conf": 0.0
            }

       # logger.info(f"Top doc metadata: {top_docs[0].metadata}")

        # 🔥 DEBUG
        logger.info("======== CONTEXT DEBUG ========")
        for i, d in enumerate(top_docs[:3]):
            logger.info(f"-------------------------------------------")
            #logger.info(f"\nDOC {i+1}:\n{d.page_content[:800]}")
        logger.info("================================")

        # ---------------- CONFIDENCE ----------------
        rag_conf = self._calculate_confidence(improved_query, top_docs)

        # ---------------- GRAPH ----------------
        graph_text = ""
        graph_conf = 0.0

        if self.graph_agent:
            try:
                g = self.graph_agent.run(question, session_id)
                if isinstance(g, dict):
                    graph_text = g.get("answer", "")
                    graph_conf = g.get("confidence", 0.0)
                else:
                    graph_text = str(g)
                    graph_conf = 0.5
            except Exception as e:
                logger.warning(f"GraphAgent failed: {e}")

        logger.info(f"RAG: {rag_conf} | GRAPH: {graph_conf}")

        # =====================================================
        # 🔥 CONTEXT BUILD (IMPROVED)
        # =====================================================
        context = []
        for i, doc in enumerate(top_docs[:5]):
            '''
            #the below code is commented  as it search the page no in whole text and it try to match first 3 digit which it match with any numerical value in text 
            # and it return wrong page no. so we are using fallback page no which is extracted from metadata   
            page = self._extract_real_page(
                doc.page_content,
                doc.metadata.get("page_label") or doc.metadata.get("page")
            )
            '''
            
            page = doc.metadata.get("page_label") or doc.metadata.get("page") or "Unknown"
            logger.info(f"\nDOCNO {i+1} \n doc page: {page} \n doc metadata: {doc.metadata}")
            
            context.append(f"""
--- DOCUMENT {i+1} ---
Source: {os.path.basename(doc.metadata.get("source",""))}
Page: {page}

{doc.page_content}
""")

        context_text = "\n\n".join(context)
        
        # =====================================================
        # 🔥 STRONG PROMPT (FINAL)
        # =====================================================
        prompt = f"""
You are a senior civil engineer.

STRICT RULES:
1. Use ONLY the provided documents
2. DO NOT ignore any relevant section
3. If answer exists, DO NOT say "not available"
4. Combine multiple documents if needed
5. ALWAYS cite like (Document 1, Page X)

Documents:
{context_text}

Graph:
{graph_text}

Question:
{question}

Return:
- Final Answer (with citations)
- Technical Explanation
- Confidence (0 to 1)
"""

        logger.info(f"\nPROMPT:\n{prompt[:1500]}")

        # ---------------- LLM ----------------
        try:
            response = self.llm.invoke(prompt).content
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            return {
                "answer": top_docs[0].page_content,
                "docs": top_docs[:5],
                "confidence": rag_conf,
                "rag_conf": rag_conf,
                "graph_conf": graph_conf
            }

        logger.info(f"\nLLM RESPONSE:\n{response}")

        return {
            "answer": response,
            "docs": top_docs[:5],
            "confidence": round(max(rag_conf, graph_conf), 2),
            "rag_conf": rag_conf,
            "graph_conf": graph_conf
        }
