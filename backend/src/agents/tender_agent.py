from backend.src.tools.tender_tool import TenderTool
from backend.src.retrieval.hybrid_retriever import HybridRetriever
from backend.src.retrieval.reranker import Reranker
from backend.src.config.config import Config


class TenderAgent:

    def __init__(self, llm, retriever):

        self.llm = llm

        # 🔥 Use RAG
        self.retriever = retriever
        self.reranker = Reranker()

        self.tool = TenderTool(llm)

    def run(self, question):

        print(f"Received question: {question} in tender agent.")

        # 🔥 Step 1 — Retrieve documents
        docs = self.retriever.retrieve(question)

        # 🔥 Step 2 — Rerank
        top_docs = self.reranker.rerank(question, docs)

        if not top_docs:
            return "No tender documents found."

        # 🔥 Step 3 — Build context
        context = "\n\n".join(
            [doc.page_content for doc in top_docs[:5]]
        )

        # 🔥 Step 4 — Analyse using tool
        return self.tool.analyse(context, question)
