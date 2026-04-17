import re
from backend.src.tools.weather_tool import WeatherTool
from backend.src.config.config import Config


class WeatherAgent:

    def __init__(self, llm):
        self.llm = llm
        self.tool = WeatherTool()

    # ---------------- SMART CITY EXTRACTION ----------------
    def extract_city(self, question):

        q = question.lower()

        patterns = [
            r"in ([a-zA-Z\s]+)",
            r"weather in ([a-zA-Z\s]+)",
            r"weather ([a-zA-Z\s]+)",
            r"([a-zA-Z\s]+) weather"
        ]

        for pattern in patterns:
            match = re.search(pattern, q)
            if match:
                return match.group(1).strip().title()

        return "Mumbai"  # fallback

    # ---------------- CONFIDENCE ----------------
    def calculate_confidence(self, weather_data):

        if "not available" in weather_data.lower():
            return 0.3

        return 0.9

    # ---------------- ZERO LLM RESPONSE ----------------
    def zero_llm_response(self, weather):

        return f"""
🌤 Weather Summary:

{weather}

⚠️ Construction Risks:
- High humidity → material curing issues
- Rain (if present) → delays & safety risks
- Extreme heat → worker fatigue

📊 Confidence: 0.8
"""

    # ---------------- MAIN ----------------
    def run(self, question):

        city = self.extract_city(question)

        print(f"City detected: {city}")

        weather = self.tool.get_weather(city)

        confidence = self.calculate_confidence(weather)

        # 🔥 ZERO LLM MODE
        if Config.ZERO_LLM_MODE:
            return self.zero_llm_response(weather)

        # 🔥 LLM MODE (STRICT GROUNDED)
        prompt = f"""
You are a construction risk analyst.

STRICT RULES:
- Use ONLY provided weather data
- DO NOT use your own knowledge
- If data missing → say "No data available"

Weather Data:
{weather}

Question:
{question}

Return:
1. Current Weather Summary
2. Construction Risks

Also include:
Confidence: {confidence}
"""

        result = self.llm.invoke(prompt).content

        return result
