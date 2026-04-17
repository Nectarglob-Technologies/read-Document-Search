from sentence_transformers import SentenceTransformer
import numpy as np


class ClauseEmbeddingStore:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.store = []

    def add_from_document(self, doc):

        text = doc.page_content
        clauses = doc.metadata.get("clauses", [])

        if not clauses:
            return

        for clause in clauses:

            # 🔥 extract clause text (simple split fallback)
            clause_text = self._extract_clause_text(text, clause)

            if not clause_text:
                continue

            emb = self.model.encode(clause_text)

            self.store.append({
                "embedding": emb,
                "text": clause_text,
                "metadata": {
                    **doc.metadata,
                    "clause": clause
                }
            })

    def _extract_clause_text(self, text, clause):

        import re

        pattern = rf"(Clause\s+{clause}.*?)(?=Clause\s+\d+|\Z)"

        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)

        return match.group(1).strip() if match else None

    def search(self, query, top_k=5):

        query_emb = self.model.encode(query)

        scored = []

        for item in self.store:

            sim = np.dot(query_emb, item["embedding"]) / (
                np.linalg.norm(query_emb) * np.linalg.norm(item["embedding"])
            )

            scored.append((sim, item))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [item for _, item in scored[:top_k]]
