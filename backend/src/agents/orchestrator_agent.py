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
        memory
    ):
        self.llm = llm
        self.engineering_agent = engineering_agent
        self.weather_agent = weather_agent
        self.tender_agent = tender_agent
        self.project_agent = project_agent
        self.graph_agent = graph_agent
        self.memory = memory
        self.logger = Logger()

    # ---------------- CONFIDENCE PARSER ----------------
    def extract_confidence(self, text):

        if not text:
            return 0.0

        match = re.search(r"confidence[:\s]+(\d+\.?\d*)", text.lower())

        if match:
            try:
                return float(match.group(1))
            except:
                return 0.5

        return 0.5  # fallback

    # ---------------- HYBRID DETECTION ----------------
    def is_hybrid(self, question):

        q = question.lower()

        structured_keywords = ["project", "material", "location", "contractor"]
        reasoning_keywords = ["why", "how", "impact", "defect", "reason"]
        print(f"🔍 Checking hybrid: structured_keywords={any(k in q for k in structured_keywords)}, reasoning_keywords={any(k in q for k in reasoning_keywords)}")

        return (
            any(k in q for k in structured_keywords) and
            any(k in q for k in reasoning_keywords)
        )

    # ---------------- ROUTING ----------------
    def route(self, question):
        print(f"🔍 It is not hybrid. So Routing question: {question}")
        q = question.lower()

        if "weather" in q:
            return "weather"

        if "tender" in q or "bid" in q:
            return "tender"

        if "project" in q:
            return "project"

        if any(k in q for k in ["material", "location", "contractor"]):
            return "graph"

        return "engineering"

    # ---------------- HYBRID FUSION ----------------
    def hybrid_answer(self, question):

        print("🔥 Hybrid query detected")

        graph_result = self.graph_agent.run(question)
        print(f"📝OrchestratorAgent Graph Result: {graph_result}")
        rag_result = self.engineering_agent.run(question)
        print(f"📝OrchestratorAgent RAG Result: {rag_result}")

        graph_conf = self.extract_confidence(graph_result)
        print(f"📝OrchestratorAgent Graph Confidence: {graph_conf:.2f}")
        rag_conf = self.extract_confidence(rag_result)
        print(f"📝OrchestratorAgent RAG Confidence: {rag_conf:.2f}")

        # 🔥 Smart decision
        if graph_conf > rag_conf + 0.1:
            return f"""
                📊 Graph-Based Answer (High Confidence)

                {graph_result}
            """

        elif rag_conf > graph_conf + 0.1:
            return f"""
📄              Engineering Answer (High Confidence)

                {rag_result}
            """

        else:
            return f"""
                🔀 Combined Answer

                📊 Graph Insight:
                {graph_result}

                📄 Engineering Insight:
                {rag_result}
            """

    # ---------------- MAIN ----------------
    def answer(self, question):

        print(f"\n🧠OrchestratorAgent Question: {question}")

        self.memory.add_user_message(question)
        print(f"📝OrchestratorAgent User Message: {question}")

        # 🔥 STEP 1 — HYBRID FLOW
        if self.is_hybrid(question):
            print("➡️ Detected hybrid question, invoking hybrid_answer()")
            final = self.hybrid_answer(question)
            print(f"📝OrchestratorAgent Hybrid Answer: {final}")

            self.memory.add_ai_message(final)
            self.logger.log(question, final)

            return final

        # 🔽 STEP 2 — ROUTING
        category = self.route(question)

        print(f"➡️ Routing to: {category}")

        if category == "weather":
            result = self.weather_agent.run(question)

        elif category == "tender":
            result = self.tender_agent.run(question)

        elif category == "project":
            result = self.project_agent.run(question)

        elif category == "graph":
            result = self.graph_agent.run(question)

        else:
             result = self.engineering_agent.run(question)

        self.memory.add_ai_message(result)
        self.logger.log(question, result)
        print(f"📝OrchestratorAgent Final Answer:")
        return result
