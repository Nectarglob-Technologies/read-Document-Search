# backend/src/pipeline/chunking.py

from langchain.text_splitter import RecursiveCharacterTextSplitter


# =========================================================
# 🔥 FILTER BAD CHUNKS (REMOVE NOISE)
# =========================================================
def is_valid_chunk(text):
    """
    Remove useless chunks:
    - Too short
    - No meaningful words
    """
    if not text:
        return False

    text = text.strip()

    if len(text) < 100:
        return False

    # Remove garbage like numbers-only
    if len(set(text)) < 10:
        return False

    return True


# =========================================================
# 🔥 SAFE TEXT LIMIT (PREVENT TOKEN ERROR)
# =========================================================
def safe_text(text, max_chars=4000):
    return text[:max_chars]


# =========================================================
# 🔥 SEMANTIC CHUNKING (BETTER THAN FIXED)
# =========================================================
def semantic_chunk_documents(docs):
    """
    Split documents into meaningful chunks
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,      # 🔥 Better than 500
        chunk_overlap=200     # 🔥 Keeps context
    )

    chunks = splitter.split_documents(docs)

    # 🔥 Apply filtering + trimming
    final_chunks = []

    for c in chunks:
        text = safe_text(c.page_content)

        if is_valid_chunk(text):
            c.page_content = text
            final_chunks.append(c)

    print(f"✅ Semantic chunks created: {len(final_chunks)}")

    return final_chunks
