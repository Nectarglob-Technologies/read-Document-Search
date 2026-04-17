"""Streamlit UI for Agentic RAG System - FINAL CLEAN VERSION"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add backend src to path
sys.path.append(str(Path(__file__).parent))

from backend.src.config.config import Config
from backend.src.document_ingestion.document_processor import DocumentProcessor
from backend.src.vectorstore.vectorstore import VectorStore

# Page config
st.set_page_config(
    page_title="🤖 RAG Search",
    page_icon="🔍",
    layout="centered"
)

# Simple CSS
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# =========================================================
# SESSION STATE
# =========================================================
def init_session_state():
    if 'retriever' not in st.session_state:
        st.session_state.retriever = None
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    if 'initialized' not in st.session_state:
        st.session_state.initialized = False
    if 'history' not in st.session_state:
        st.session_state.history = []


# =========================================================
# INITIALIZATION
# =========================================================
@st.cache_resource
def initialize_rag():
    try:
        # 🔹 LLM
        llm = Config.get_llm()

        # 🔹 Document Processing
        processor = DocumentProcessor(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP
        )

        documents = processor.process()

        if not documents:
            return None, None, 0

        # 🔹 Vector Store
        vector_store = VectorStore()
        vector_store.create_vectorstore(documents)

        retriever = vector_store.get_retriever()

        return retriever, llm, len(documents)

    except Exception as e:
        st.error(f"❌ Failed to initialize: {str(e)}")
        return None, None, 0


# =========================================================
# SIMPLE RAG PIPELINE
# =========================================================
def run_rag(question, retriever, llm):

    # 🔹 Retrieve docs
    docs = retriever.get_relevant_documents(question)

    context = "\n\n".join([d.page_content for d in docs[:5]])

    # 🔹 Prompt
    prompt = f"""
    Answer the question using only the context below.

    Context:
    {context}

    Question:
    {question}
    """

    response = llm.invoke(prompt)

    return {
        "answer": response.content if hasattr(response, "content") else str(response),
        "docs": docs
    }


# =========================================================
# MAIN APP
# =========================================================
def main():

    init_session_state()

    st.title("🔍 RAG Document Search")
    st.markdown("Ask questions about your documents")

    # 🔹 Initialize
    if not st.session_state.initialized:
        with st.spinner("🚀 Initializing system..."):

            retriever, llm, num_chunks = initialize_rag()

            if retriever:
                st.session_state.retriever = retriever
                st.session_state.llm = llm
                st.session_state.initialized = True

                st.success(f"✅ System ready! ({num_chunks} chunks loaded)")
            else:
                st.warning("⚠️ No documents found. Please add PDFs to /data folder")

    st.markdown("---")

    # 🔹 Input
    question = st.text_input(
        "Enter your question:",
        placeholder="Ask something about your documents..."
    )

    # 🔹 Search
    if st.button("🔍 Search") and question:

        if not st.session_state.retriever:
            st.error("System not initialized properly")
            return

        with st.spinner("Searching..."):
            start = time.time()

            result = run_rag(
                question,
                st.session_state.retriever,
                st.session_state.llm
            )

            elapsed = time.time() - start

        # 🔹 Store history
        st.session_state.history.append({
            "question": question,
            "answer": result["answer"],
            "time": elapsed
        })

        # 🔹 Answer
        st.markdown("### 💡 Answer")
        st.success(result["answer"])

        # 🔹 Sources
        with st.expander("📄 Source Documents"):
            for i, doc in enumerate(result["docs"], 1):
                st.text_area(
                    f"Document {i}",
                    doc.page_content[:300] + "...",
                    height=100
                )

        st.caption(f"⏱️ Response time: {elapsed:.2f} sec")

    # 🔹 History
    if st.session_state.history:
        st.markdown("---")
        st.markdown("### 📜 Recent Searches")

        for item in reversed(st.session_state.history[-3:]):
            st.markdown(f"**Q:** {item['question']}")
            st.markdown(f"**A:** {item['answer'][:200]}...")
            st.caption(f"Time: {item['time']:.2f}s")
            st.markdown("")


# =========================================================
if __name__ == "__main__":
    main()
