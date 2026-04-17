# backend/src/graph_rag/graph_extractor.py

import json


class GraphExtractor:

    def __init__(self, llm):
        self.llm = llm

    def extract(self, text):

        if not self.llm:
            return {}

        # 🔥 Trim text (avoid token overload)
        text = text[:3000]

        prompt = f"""
You are an expert information extractor for construction documents.

Extract structured data in STRICT JSON format.

ONLY return valid JSON. No explanation.

----------------------------------------
EXAMPLE:

Text:
ABC Bridge Project uses M30 Concrete in Mumbai built by XYZ Constructions.

Output:
{{
  "projects": ["ABC Bridge Project"],
  "contractors": ["XYZ Constructions"],
  "materials": ["M30 Concrete"],
  "locations": ["Mumbai"],
  "relations": [
    {{"project":"ABC Bridge Project","relation":"USES","entity":"M30 Concrete"}},
    {{"project":"ABC Bridge Project","relation":"LOCATED_IN","entity":"Mumbai"}},
    {{"project":"ABC Bridge Project","relation":"BUILT_BY","entity":"XYZ Constructions"}}
  ]
}}

----------------------------------------

Now extract from below text:

Text:
{text}

Output:
"""

        try:
            result = self.llm.invoke(prompt)

            content = result.content.strip()

            # 🔥 HARD FIX: remove markdown if LLM adds it
            if content.startswith("```"):
                content = content.replace("```json", "").replace("```", "").strip()

            data = json.loads(content)

            return data

        except Exception as e:
            print("❌ LLM parsing failed:", e)
            return {}
