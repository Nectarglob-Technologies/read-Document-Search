import re


class TenderTool:

    def __init__(self, llm=None):
        self.llm = llm

    # ---------------- SECTION EXTRACTION ----------------
    def extract_sections(self, text):

        sections = {
            "scope": [],
            "technical": [],
            "deadline": [],
            "penalty": []
        }

        lines = text.split("\n")

        for line in lines:

            l = line.lower()

            if any(k in l for k in ["scope", "work description"]):
                sections["scope"].append(line)

            elif any(k in l for k in ["technical", "specification"]):
                sections["technical"].append(line)

            elif any(k in l for k in ["deadline", "completion", "timeline"]):
                sections["deadline"].append(line)

            elif any(k in l for k in ["penalty", "liquidated damages"]):
                sections["penalty"].append(line)

        return sections

    # ---------------- CONFIDENCE ----------------
    def calculate_confidence(self, text, sections):

        score = 0

        for key, value in sections.items():
            if value:
                score += 1

        return round(score / 4, 2)  # max 1.0

    # ---------------- MAIN ANALYSIS ----------------
    def analyse(self, context, question):

        # 🔥 Step 1 — Rule-based extraction
        sections = self.extract_sections(context)

        confidence = self.calculate_confidence(context, sections)

        # 🔥 Step 2 — ZERO LLM MODE
        if not self.llm:
            return f"""
📄 Tender Analysis (Zero-LLM Mode)

🔹 Scope:
{chr(10).join(sections['scope'][:5]) or "Not found"}

🔹 Technical:
{chr(10).join(sections['technical'][:5]) or "Not found"}

🔹 Deadline:
{chr(10).join(sections['deadline'][:5]) or "Not found"}

🔹 Penalty:
{chr(10).join(sections['penalty'][:5]) or "Not found"}

📊 Confidence: {confidence}
"""

        # 🔥 Step 3 — LLM (STRICT GROUNDED)
        prompt = f"""
You are a tender analysis expert.

STRICT RULES:
- Use ONLY provided context
- DO NOT add external knowledge
- Extract exact facts
- If not found → say "Not available"

Context:
{context[:3000]}

Question:
{question}

Return:

1. Scope of Work
2. Technical Requirements
3. Deadline
4. Penalty Clauses
5. Key Risks

Also provide:
- Citations like [1], [2]
- Confidence score (0 to 1)
"""

        response = self.llm.invoke(prompt)

        return f"""
📄 Tender Analysis

{response.content}

📊 Confidence: {confidence}
"""
