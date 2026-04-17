import json
import re


class GraphQueryEngine:

    def __init__(self, graph_store, llm):

        self.graph = graph_store
        self.llm = llm


    # ---------------- LLM ENTITY EXTRACTION ----------------
    def extract_entity_llm(self, question):

        prompt = f"""
        Extract entity information from this question.

        Question:
        {question}

        Return ONLY valid JSON:
        {{
            "material": "",
            "location": "",
            "contractor": ""
        }}
        """

        try:
            result = self.llm.invoke(prompt).content

            # Convert string → dict
            return json.loads(result)

        except Exception:
            return {}


    # ---------------- RULE-BASED EXTRACTION (Fallback) ----------------
    def extract_entity_rule(self, question):

        material = None
        location = None

        # Material pattern
        mat_match = re.search(r"M\d+\s*Concrete", question, re.IGNORECASE)
        if mat_match:
            material = mat_match.group(0)

        # Location pattern
        loc_match = re.search(
            r"(Maharashtra|Mumbai|Delhi|Pune|Bangalore)",
            question,
            re.IGNORECASE
        )
        if loc_match:
            location = loc_match.group(0)

        return {
            "material": material,
            "location": location
        }


    # ---------------- MAIN QUERY ----------------
    def query(self, question):

        # 🔹 Step 1: Try RULE-based first (FAST + FREE)
        entity_info = self.extract_entity_rule(question)

        material = entity_info.get("material")
        location = entity_info.get("location")

        # 🔹 Step 2: If rule-based fails → use LLM
        if not material and not location:

            print("⚠️ Rule-based failed → using LLM")

            entity_info = self.extract_entity_llm(question)

            material = entity_info.get("material")
            location = entity_info.get("location")

        # 🔹 Step 3: Query graph
        if material:
            return self.graph.get_projects_using_material(material)

        if location:
            return self.graph.get_projects_by_location(location)

        return []
