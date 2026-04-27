from backend.src.utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:

    def __init__(self, vector_retriever, keyword_retriever=None):

        self.vector_retriever = vector_retriever
        self.keyword_retriever = keyword_retriever

    def retrieve(self, query, k=8):

        vector_docs = self.vector_retriever.invoke(query) or []
        logger.info(f"Vector Retriever returned {len(vector_docs)} documents")

        keyword_docs = []
        if self.keyword_retriever:
            keyword_docs = self.keyword_retriever.invoke(query) or []
            logger.info(f"Keyword Retriever returned {len(keyword_docs)} documents")  

        # Merge results
        combined = vector_docs + keyword_docs
        logger.info(f"Combined documents: {len(combined)}")

        # Remove duplicates
        unique_docs = {}
        for doc in combined:
            unique_docs[doc.page_content] = doc

        logger.info(f"Unique documents after merging: {len(unique_docs)}")
        return list(unique_docs.values())[:k]
