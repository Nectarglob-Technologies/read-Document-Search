class QueryRewriter:

    def __init__(self, llm):
        self.llm = llm

    def rewrite(self, question: str) -> str:

        prompt = f"""
You are an expert at rewriting search queries for engineering documents.

Rewrite the question to improve document retrieval.

Question:
{question}

Improved Search Query:
"""

        response = self.llm.invoke(prompt)

        return response.content.strip()
