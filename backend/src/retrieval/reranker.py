from sentence_transformers import CrossEncoder
from backend.src.utils.logger import get_logger

logger = get_logger(__name__)


class Reranker:
    """
    🔥 CrossEncoder-based semantic reranker (state-of-the-art)

    Re-ranks retrieved documents using deep semantic understanding
    of (query, document) pairs.
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):

        try:
            self.model = CrossEncoder(model_name)
            logger.info(f"✅ CrossEncoder loaded: {model_name}")
        except Exception as e:
            logger.error(f"❌ Failed to load CrossEncoder: {e}")
            self.model = None

    # =========================================================
    # 🔹 MAIN RERANK
    # =========================================================
    def rerank(self, query, docs, top_k=10):

        if not docs:
            return []

        # 🔥 fallback safety
        if self.model is None:
            logger.warning("⚠️ Reranker fallback (model not loaded)")
            return docs[:top_k]

        try:
            # 🔥 Step 1: limit docs for performance
            docs = docs[:20]   # important optimization

            # 🔥 Step 2: create (query, doc) pairs
            pairs = [
                (query, doc.page_content[:512])  # truncate long docs
                for doc in docs
            ]

            # 🔥 Step 3: predict scores
            scores = self.model.predict(pairs)

            # 🔥 Step 4: attach scores
            doc_scores = list(zip(docs, scores))

            # 🔥 Step 5: sort by score
            ranked = sorted(
                doc_scores,
                key=lambda x: float(x[1]),
                reverse=True
            )

            # 🔥 DEBUG LOGS (very useful)
            logger.info("🔎 CrossEncoder Ranking:")
            for i, (doc, score) in enumerate(ranked[:5]):
                preview = doc.page_content[:120].replace("\n", " ")
                logger.info(f"[{i+1}] Score={round(float(score),3)} | {preview}")

            # 🔥 Step 6: return top_k docs
            return [doc for doc, _ in ranked[:top_k]]

        except Exception as e:
            logger.error(f"❌ Reranker failed: {e}")
            return docs[:top_k]
