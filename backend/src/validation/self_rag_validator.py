class SelfRAGValidator:

    def __init__(self,llm):

        self.llm=llm

    def validate(self,question,answer,docs):

        context="\n".join([d.page_content for d in docs])

        prompt=f"""
Check if answer is supported by context.

Question:
{question}

Answer:
{answer}

Context:
{context}

Return YES or NO
"""

        result=self.llm.invoke(prompt)

        return result.content
