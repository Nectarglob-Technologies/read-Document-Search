class HybridRetriever:

    def __init__(self, vector_retriever, keyword_retriever=None):

        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever

    def retrieve(self, query, k=8):

        vector_docs = self.vector_retriever.invoke(query) or []

        keyword_docs = []
        if self.keyword_retriever:
            keyword_docs = self.keyword_retriever.invoke(query) or []

        # Merge results
        combined = vector_docs + keyword_docs

        # Remove duplicates
        unique_docs = {}
        for doc in combined:
            unique_docs[doc.page_content] = doc

        return list(unique_docs.values())[:k]
