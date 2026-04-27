class Reranker:

    def __init__(self):
        pass

    def rerank(self, query, docs, top_k=5):

        if not docs:
            return []

        query_terms = query.lower().split()

        scored_docs = []

        for doc in docs:
            text = doc.page_content.lower()

            # simple keyword scoring
            score = sum(1 for term in query_terms if term in text)

            scored_docs.append((score, doc))

        # sort by score descending
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for _, doc in scored_docs[:top_k]]
