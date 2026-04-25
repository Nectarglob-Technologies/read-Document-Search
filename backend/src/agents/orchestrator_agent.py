from backend.src.utils.logger import Logger
import re


class OrchestratorAgent:

    def __init__(
        self,
        llm,
        engineering_agent,
        weather_agent,
        tender_agent,
        project_agent,
        graph_agent,
        memory,
        router=None
    ):
        self.llm = llm
        self.engineering_agent = engineering_agent
        self.weather_agent = weather_agent
        self.tender_agent = tender_agent
        self.project_agent = project_agent
        self.graph_agent = graph_agent
        self.memory = memory
        self.router = router

        self.logger = Logger()

    # =========================================================
    # 🔥 SAFE RESPONSE PARSER (KEEP FOR LEGACY SUPPORT)
    # =========================================================
    def _parse_response(self, result):
        """
        Handles BOTH:
        - dict response (NEW agents)
        - string response (OLD agents)
        """

        if isinstance(result, dict):
            return (
                result.get("answer", ""),
                result.get("confidence", 0.0)
            )

        if isinstance(result, str):
            return result, self.extract_confidence(result)

        return str(result), 0.0

    # =========================================================
    # 🔥 CONFIDENCE PARSER (STRING FALLBACK)
    # =========================================================
    def extract_confidence(self, text):

        if not text:
            return 0.0

        try:
            match = re.search(r"confidence[:\s]+(\d+\.?\d*)", text.lower())
            if match:
                return float(match.group(1))
        except:
            pass

        return 0.5

    # =========================================================
    # 🔥 CLEAN OUTPUT
    # =========================================================
    def _clean_output(self, text):

        if not text:
            return "No response generated."

        text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
        return text.strip()

    # =========================================================
    # 🔥 HYBRID ANSWER (UPDATED TO RETURN DICT)
    # =========================================================
    def hybrid_answer(self, question):

        print("🔥 Hybrid query detected")

        graph_raw = self.graph_agent.run(question)
        rag_raw = self.engineering_agent.run(question)

        # ✅ Preserve dicts if available
        graph_text, graph_conf = self._parse_response(graph_raw)
        rag_text, rag_conf = self._parse_response(rag_raw)

        print(f"Graph confidence: {graph_conf}")
        print(f"RAG confidence: {rag_conf}")

        # ---------------------------
        # GRAPH DOMINANT
        # ---------------------------
        if graph_conf > rag_conf + 0.1:
            return {
                "answer": self._clean_output(f"""
📊 Graph-Based Answer

{graph_text}
"""),
                "docs": [],
                "rag_conf": rag_conf,
                "graph_conf": graph_conf
            }

        # ---------------------------
        # RAG DOMINANT
        # ---------------------------
        if rag_conf > graph_conf + 0.1:
            if isinstance(rag_raw, dict):
                return rag_raw  # ✅ PRESERVE FULL DATA

            return {
                "answer": self._clean_output(rag_text),
                "docs": [],
                "rag_conf": rag_conf,
                "graph_conf": graph_conf
            }

        # ---------------------------
        # TRUE HYBRID
        # ---------------------------
        return {
            "answer": self._clean_output(f"""
🔀 Combined Answer

### 📊 Graph Insight
{graph_text}

---

### 📄 Engineering Insight
{rag_text}
"""),
            "docs": rag_raw.get("docs", []) if isinstance(rag_raw, dict) else [],
            "rag_conf": rag_conf,
            "graph_conf": graph_conf
        }

    # =========================================================
    # 🔥 FALLBACK ROUTING
    # =========================================================
    def route_fallback(self, question):

        q = question.lower()

        if "weather" in q:
            return "weather"

        if "tender" in q or "bid" in q:
            return "tender"

        if "project" in q:
            return "project"

        if any(k in q for k in ["material", "location", "contractor"]):
            return "graph"

        return "rag"

    # =========================================================
    # 🔥 MAIN ENTRY (CRITICAL FIX HERE)
    # =========================================================
    def answer(self, question):

        print(f"\n🧠 Question: {question}")

        # ---------------- MEMORY ----------------
        try:
            self.memory.add_user_message(question)
        except:
            pass

        # ---------------- ROUTER ----------------
        intent = None

        if self.router:
            try:
                intent = self.router.route(question)
            except:
                intent = None

        if not intent:
            intent = self.route_fallback(question)

        print(f"➡️ Intent: {intent}")

        # ---------------- EXECUTION ----------------
        try:
            if intent == "hybrid":
                result = self.hybrid_answer(question)

            elif intent == "graph":
                result = self.graph_agent.run(question)

            elif intent == "rag":
                result = self.engineering_agent.run(question)

            elif intent == "weather":
                result = self.weather_agent.run(question)

            elif intent == "tender":
                result = self.tender_agent.run(question)

            elif intent == "project":
                result = self.project_agent.run(question)

            else:
                result = self.engineering_agent.run(question)

        except Exception as e:
            print(f"❌ Agent failed: {e}")
            result = {
                "answer": "Something went wrong. Please try again.",
                "docs": [],
                "rag_conf": 0.0,
                "graph_conf": 0.0
            }

        # =====================================================
        # 🔥 CRITICAL FIX: DO NOT DESTROY STRUCTURE
        # =====================================================
        if isinstance(result, dict):
            final_result = result
            text = result.get("answer", "")
        else:
            text, _ = self._parse_response(result)
            text = self._clean_output(text)

            final_result = {
                "answer": text,
                "docs": [],
                "rag_conf": 0.0,
                "graph_conf": 0.0
            }

        # ---------------- MEMORY SAVE ----------------
        try:
            self.memory.add_ai_message(text)
        except:
            pass

        # ---------------- LOGGING ----------------
        try:
            self.logger.log(question, text)
        except:
            pass

        return final_result   # ✅ RETURN FULL STRUCTURE
