class RAGDebugger:

    @staticmethod
    def print_retrieval(question, docs):

        print("\n")
        print("=" * 70)
        print("RAG DEBUG INFORMATION")
        print("=" * 70)

        print("\nUser Question:")
        print(question)

        print("\nRetrieved Chunks:")

        for i, doc in enumerate(docs, start=1):

            meta = doc.metadata if hasattr(doc, "metadata") else {}

            source = meta.get("source", "unknown")
            section = meta.get("section", "NA")
            clause = meta.get("clause", "NA")

            text = doc.page_content[:300]

            print("\n----------------------------------")
            print(f"Chunk #{i}")
            print(f"Source : {source}")
            print(f"Section: {section}")
            print(f"Clause : {clause}")
            print("Content Preview:")
            print(text)
            print("----------------------------------")

        print("\nTotal Chunks Retrieved:", len(docs))
        print("=" * 70)

        for doc, score in docs:

            print("Score:", score)
