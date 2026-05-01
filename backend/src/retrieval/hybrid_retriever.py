from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:

    def __init__(self, vectorstore, documents):

        self.vectorstore = vectorstore
        self.documents = documents

        # 🔥 Prepare BM25
        self.tokenized_docs = [
            doc.page_content.lower().split()
            for doc in documents
        ]

        self.bm25 = BM25Okapi(self.tokenized_docs)

        # weights (tunable)
        self.faiss_weight = 0.7
        self.bm25_weight = 0.3

    # =========================================================
    # 🔹 MAIN RETRIEVE
    # =========================================================
    def retrieve(self, query, top_k=10):

        # ---------------- FAISS ----------------
        faiss_docs = self.vectorstore.similarity_search(query, k=top_k)

        # ---------------- BM25 ----------------
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_docs = [self.documents[i] for i in top_bm25_idx]

        # ---------------- MERGE (DEDUP FIX) ----------------
        doc_scores = {}
        doc_map = {}

        def get_doc_key(doc):
            content = doc.page_content.strip()[:500]  # avoid huge text
            source = doc.metadata.get("source", "")
            page = doc.metadata.get("page", "")
            return f"{source}_{page}_{hash(content)}"

        # FAISS scoring
        for rank, doc in enumerate(faiss_docs):
            key = get_doc_key(doc)

            score = 1 / (rank + 1)

            if key not in doc_scores:
                doc_scores[key] = 0
                doc_map[key] = doc

            doc_scores[key] += score * self.faiss_weight

        # BM25 scoring
        for rank, doc in enumerate(bm25_docs):
            key = get_doc_key(doc)

            score = 1 / (rank + 1)

            if key not in doc_scores:
                doc_scores[key] = 0
                doc_map[key] = doc

            doc_scores[key] += score * self.bm25_weight

        # ---------------- SORT ----------------
        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        final_docs = [doc_map[key] for key, _ in ranked[:top_k]]

        return final_docs


    ''''
    def retrieve(self, query, top_k=10):

        # ---------------- FAISS ----------------
        faiss_docs = self.vectorstore.similarity_search(query, k=top_k)

        # ---------------- BM25 ----------------
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)

        top_bm25_idx = np.argsort(bm25_scores)[::-1][:top_k]
        bm25_docs = [self.documents[i] for i in top_bm25_idx]

        # ---------------- MERGE ----------------
        doc_scores = {}

        # FAISS scoring
        for rank, doc in enumerate(faiss_docs):
            score = 1 / (rank + 1)
            doc_scores[id(doc)] = doc_scores.get(id(doc), 0) + score * self.faiss_weight

        # BM25 scoring
        for rank, doc in enumerate(bm25_docs):
            score = 1 / (rank + 1)
            doc_scores[id(doc)] = doc_scores.get(id(doc), 0) + score * self.bm25_weight

        # ---------------- SORT ----------------
        doc_map = {id(doc): doc for doc in (faiss_docs + bm25_docs)}

        ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

        final_docs = [doc_map[i[0]] for i in ranked[:top_k]]

        return final_docs
    '''