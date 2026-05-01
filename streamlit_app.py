import streamlit as st
import uuid
import time
import os

from transformers import pipeline
from redis import Redis

# ============================
# IMPORTS
# ============================
from backend.src.config.config import Config
from backend.src.vectorstore.faiss_store import FAISSStore
from backend.src.pipeline.ingestion_pipeline import IngestionPipeline

from backend.src.graph_rag.graph_query import GraphQueryEngine
from backend.src.memory.conversation_memory import ConversationMemory

from backend.src.agents.engineering_agent import EngineeringAgent
from backend.src.agents.weather_agent import WeatherAgent
from backend.src.agents.tender_agent import TenderAgent
from backend.src.agents.project_agent import ProjectAgent
from backend.src.agents.graph_agent import GraphAgent
from backend.src.agents.orchestrator_agent import OrchestratorAgent

from backend.src.orchestration.router import QueryRouter
from backend.src.orchestration.conversation_layer import ConversationLayer

from backend.src.utils.logger import get_logger
logger = get_logger(__name__)


# ============================
# REDIS INIT
# ============================
@st.cache_resource
def get_redis():
    try:
        r = Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", 6379)),
            password=os.getenv("REDIS_PASSWORD", None),
            decode_responses=True,
            socket_connect_timeout=2
        )
        r.ping()
        logger.info("✅ Redis connected")
        return r
    except Exception as e:
        logger.warning(f"⚠️ Redis not available: {e}")
        return None


redis = get_redis()
conv = ConversationLayer(redis)


# ============================
# LOGIN
# ============================
def get_user():

    if "user_id" not in st.session_state:

        user_cookie = st.query_params.get("user")

        if user_cookie:
            st.session_state.user_id = user_cookie
            logger.info(f"User restored: {user_cookie}")
        else:
            name = st.text_input("Enter username")

            if st.button("Login"):
                uid = f"user_{uuid.uuid4().hex[:8]}"
                st.session_state.user_id = uid
                st.query_params["user"] = uid
                logger.info(f"New user created: {uid}")
                st.rerun()

    return st.session_state.get("user_id")


# ============================
# SYSTEM INIT
# ============================
@st.cache_resource
def init_system():

    logger.info("🚀 Initializing system")

    llm = Config.get_llm()

    store = FAISSStore()
    faiss_loaded = store.load()

    pipeline = IngestionPipeline(store, "graph.pkl", llm)

    # 🔥 AUTO INGEST
    if not faiss_loaded or pipeline.graph_store is None:

        logger.warning("🔥 Running ingestion (missing FAISS/Graph)")

        from backend.src.document_ingestion.document_processor import DocumentProcessor

        data_path = "data"

        if not os.path.exists(data_path):
            raise ValueError("❌ 'data' folder not found")

        processor = DocumentProcessor()

        raw_docs = processor.load_documents(data_path)
        chunks = processor.split_documents(raw_docs)

        if not chunks:
            raise ValueError("❌ No documents found")

        pipeline.ingest(chunks)

        logger.info("✅ Ingestion completed")

        store.load()
        graph_store = pipeline._load_graph()

    else:
        logger.info("✅ Using existing FAISS + Graph")
        graph_store = pipeline.graph_store

    # GRAPH
    if graph_store:
        graph_engine = GraphQueryEngine(graph_store, redis_client=redis)
        graph_agent = GraphAgent(llm, graph_engine)
    else:
        graph_agent = None

    # AGENTS
    router = QueryRouter(llm, redis_client=redis)
    retriever = store.get_retriever()

    engineering = EngineeringAgent(llm, retriever, graph_agent=graph_agent)
    weather = WeatherAgent(llm)
    tender = TenderAgent(llm, retriever)
    project = ProjectAgent(graph_store)

    memory = ConversationMemory()

    orchestrator = OrchestratorAgent(
        llm,
        engineering,
        weather,
        tender,
        project,
        graph_agent,
        memory,
        router=router
    )

    logger.info("✅ System ready")

    return orchestrator, store, pipeline

# =========================================================
# SESSION SETUP (🔥 IMPORTANT)
# =========================================================
def initialize_session():

    # Unique user session
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Memory per user
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()

    # Chat history UI (optional)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

# ============================
# STREAM TEXT
# ============================
def stream_text(text):
    placeholder = st.empty()
    output = ""

    for line in text.split("\n"):
        for word in line.split():
            output += word + " "
            placeholder.markdown(output.replace("\n", "  \n"))
            time.sleep(0.01)
        output += "\n\n"

    return output

# ============================
# 📤 FILE UPLOAD HANDLER (INCREMENTAL)
# ============================
def handle_file_upload(pipeline):

    st.markdown("## 📤 Upload Document")

    # ============================
    # 📁 FILE SELECT
    # ============================
    uploaded_file = st.file_uploader(
        "Select PDF",
        type=["pdf"],
        key="file_uploader"
    )

    # Store file in session (IMPORTANT)
    if uploaded_file:
        st.session_state.selected_file = uploaded_file

    # ============================
    # ⬆️ UPLOAD BUTTON
    # ============================
    if st.button("🚀 Upload Document"):

        if "selected_file" not in st.session_state:
            st.warning("⚠️ Please select a file first")
            return

        file = st.session_state.selected_file

        save_path = os.path.join("data", file.name)
        logger.info(f"save file path: {save_path}, size: {file.size} bytes") 

        # ============================
        # 🔁 DUPLICATE CHECK
        # ============================
        if os.path.exists(save_path):
            st.warning(f"⚠️ File already exists: {file.name}")
            return

        # ============================
        # 💾 SAVE FILE
        # ============================
        with open(save_path, "wb") as f:
            f.write(file.getbuffer())

        st.success(f"📁 Saved: {file.name}")

        # ============================
        # 🔥 PROCESS FILE
        # ============================
        placeholder = st.empty()

        with st.spinner("Processing & indexing..."):

            try:
                from backend.src.document_ingestion.document_processor import DocumentProcessor

                processor = DocumentProcessor()

                raw_docs  = processor.load_single_document(str(save_path))
                chunks = processor.split_documents(raw_docs)

                logger.info(f"✅ New chunks: {len(chunks)}")

                if not chunks:
                    placeholder.error("❌ No content extracted")
                    return

                # 🔥 BACKGROUND INGESTION
                #run_ingestion_in_background(pipeline, chunks)

                # Run incremental ingestion
                pipeline.ingest(chunks)
                
                #st.info("⚡ Background ingestion started...")
                placeholder.success("✅ Document uploaded & indexed (Incremental)")

                # 🔥 VERY IMPORTANT
                st.cache_resource.clear()

                # clear selected file
                del st.session_state.selected_file

                st.rerun()

            except Exception as e:
                placeholder.error(f"❌ Error: {e}")


# ============================
# RESPONSE
# ============================
def render_response(result):

    import streamlit as st
    import os
    import base64
    import re

    # =============================
    # HELPERS
    # =============================
    '''
    def clean_text(text):
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'(\w)\s+(\w)', r'\1\2', text)
        return text.strip()
    '''
    def get_snippet(text, query, window=200):
        #text = clean_text(text)
        text_lower = text.lower()

        for word in query.lower().split():
            idx = text_lower.find(word)
            if idx != -1:
                start = max(0, idx - window)
                end = min(len(text), idx + window)
                return text[start:end]

        return text[:400]

    def deduplicate_docs(docs):
        seen = set()
        unique = []

        for d in docs:
            key = (
                d.metadata.get("source"),
                d.metadata.get("page")
            )
            if key not in seen:
                seen.add(key)
                unique.append(d)

        return unique

    # =============================
    # VALIDATION
    # =============================
    if not result:
        st.warning("No response generated.")
        return

    if isinstance(result, dict):
        answer = result.get("answer", "")
        docs = result.get("docs", [])
        query = result.get("query", "")
    else:
        answer = str(result)
        docs = []
        query = ""

    docs = deduplicate_docs(docs)

    # =============================
    # ANSWER
    # =============================
    st.markdown("## 💡 Answer")
    answer = answer.replace("###", "\n###")  # 🔥 fix formatting
    st.markdown(answer)   # ✅ KEEP RAW FORMATTING

    #st.markdown("## 💡 Answer")
    #st.write(answer)

    # =============================
    # SOURCES
    # =============================
    st.markdown("### 📚 Sources")

    for i, doc in enumerate(docs):

        file_path = doc.metadata.get("source")
        file_name = os.path.basename(file_path) if file_path else "Unknown"
        page = doc.metadata.get("page_label") or doc.metadata.get("page") or 1

        if not file_path:
            continue

        try:
            # =============================
            # PDF LINK (INLINE)
            # =============================
            with open(file_path, "rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")

            pdf_url = f"data:application/pdf;base64,{base64_pdf}#page={page}"

            st.markdown(
                f"""
                **[{i+1}] {file_name} (Page {page})**  
                <a href="{pdf_url}" target="_blank">🔗 Open Document</a>
                """,
                unsafe_allow_html=True
            )

            # =============================
            # SMART SNIPPET (KEY FIX)
            # =============================
            snippet = get_snippet(doc.page_content, query)

            st.markdown("🔍 Match Preview:")
            st.write(snippet)

        except Exception as e:
            st.warning(f"Could not open {file_name}")


# ============================
# MAIN
# ============================
def main():

    st.set_page_config(layout="wide")
    st.title("🤖 Agentic RAG System")
    st.markdown("Upload documents or ask questions")

    user_id = get_user()
    if not user_id:
        return

    orchestrator,store,pipeline = init_system()

    if not orchestrator:
        st.warning("⚠️ No documents found.")
        return
    
    #upload single file with incremental indexing
    handle_file_upload(pipeline)

    # ======================
    # SESSION STATE
    # ======================
    if "chat_id" not in st.session_state:
        st.session_state.chat_id = None

    chat_id = st.session_state.chat_id

    # ======================
    # SIDEBAR
    # ======================
    st.sidebar.title("💬 Chats")

    all_chats = conv.get_user_chats(user_id) if redis else []

    # 🔍 SEARCH
    search = st.sidebar.text_input("🔍 Search chats")

    chats = [
        cid for cid in all_chats
        if search.lower() in (conv.get_title(cid) or "").lower()
    ] if search else all_chats

    # ➕ NEW CHAT
    if st.sidebar.button("➕ New Chat", use_container_width=True):

        existing = conv.get_messages(chat_id) if (redis and chat_id) else []

        if existing:
            new_id = conv.create_chat(user_id)
            st.session_state.chat_id = new_id
            logger.info("New chat created")
        else:
            logger.info("Skipped empty chat")

        st.rerun()

    st.sidebar.markdown("---")

    # ======================
    # CHAT LIST
    # ======================
    for i, cid in enumerate(chats):

        title = conv.get_title(cid) or f"Chat {i+1}"

        col1, col2, col3 = st.sidebar.columns([6, 1, 1])

        # OPEN
        with col1:
            if st.button(title[:25], key=f"chat_{cid}"):
                st.session_state.chat_id = cid
                st.rerun()

        # RENAME
        with col2:
            if st.button("✏️", key=f"rename_{cid}"):
                st.session_state[f"rename_{cid}"] = True

        # DELETE
        with col3:
            if st.button("🗑", key=f"delete_{cid}"):

                conv.delete_chat(cid)

                if cid == chat_id:
                    st.session_state.chat_id = None

                st.rerun()

        # RENAME INPUT
        if st.session_state.get(f"rename_{cid}"):

            new_title = st.sidebar.text_input(
                "Rename",
                value=title,
                key=f"input_{cid}"
            )

            if st.sidebar.button("Save", key=f"save_{cid}"):
                conv.set_title(cid, new_title.strip())
                st.session_state[f"rename_{cid}"] = False
                st.rerun()

    # ======================
    # CHAT HISTORY
    # ======================
    if chat_id:
        messages = conv.get_messages(chat_id) if redis else []
    else:
        messages = []

    for m in messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])



    # ======================
    # INPUT
    # ======================
    prompt = st.chat_input("Ask something...")

    if prompt:

        # 👉 CREATE CHAT ON FIRST MESSAGE
        if not chat_id:
            chat_id = conv.create_chat(user_id)
            st.session_state.chat_id = chat_id
            logger.info("First chat created")

        conv.add_message(chat_id, "user", prompt)

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                result = orchestrator.answer(prompt)

                # 🔥 ALWAYS EXTRACT ANSWER SAFELY
                if isinstance(result, dict):
                    answer = result.get("answer", "")
                else:
                    answer = str(result)

                # 🔥 RENDER FULL RESULT (not split)
                render_response(result)

                # 🔥 SAVE ONLY TEXT TO CHAT
                conv.add_message(chat_id, "assistant", answer)


        #conv.add_message(chat_id, "assistant", answer)

        # AUTO TITLE
        title = conv.get_title(chat_id)
        if not title or title == "New Chat":
            conv.set_title(chat_id, prompt[:40])


# ============================
if __name__ == "__main__":
    main()
