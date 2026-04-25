import streamlit as st
import uuid
import time
import os
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

    return orchestrator


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
# RESPONSE
# ============================
def render_response(result):

    if not result:
        st.warning("No response generated.")
        return

    # ----------------------------
    # HANDLE BOTH FORMATS
    # ----------------------------
    if isinstance(result, dict):
        answer = result.get("answer", "")
        docs = result.get("docs", [])
        rag_conf = result.get("rag_conf", None)
        graph_conf = result.get("graph_conf", None)
    else:
        answer = result
        docs = []
        rag_conf = None
        graph_conf = None

    # ----------------------------
    # ANSWER
    # ----------------------------
    st.markdown("### 💡 Answer")
    stream_text(answer)

    # ----------------------------
    # SOURCES (FIXED)
    # ----------------------------
    if docs:
        st.markdown("### 📚 Sources")

        seen = set()

        for i, d in enumerate(docs):
            src = d.metadata.get("source", "Unknown")
            page = d.metadata.get("page_label") or d.metadata.get("page", "N/A")

            key = f"{src}-{page}"
            if key in seen:
                continue
            seen.add(key)

            st.markdown(f"**[{i+1}]** {src} (Page {page})")

    # ----------------------------
    # CONFIDENCE (FIXED)
    # ----------------------------
    if rag_conf is not None:
        st.markdown("### 📊 Confidence")
        st.info(f"RAG: {rag_conf} | Graph: {graph_conf}")

    # ----------------------------
    # DEBUG
    # ----------------------------
    with st.expander("⚙️ Debug Info"):
        st.write(result)


# ============================
# MAIN
# ============================
def main():

    st.set_page_config(layout="wide")
    st.title("🤖 Agentic RAG System")

    user_id = get_user()
    if not user_id:
        return

    orchestrator = init_system()

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
