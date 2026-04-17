import json
import os

CONFIG_FILE = "logs/thresholds.json"


class ThresholdOptimizer:

    def __init__(self):

        self.data = {
            "project_threshold": 300,
            "relation_threshold": 800,
            "llm_success": 0,
            "llm_calls": 0
        }

        self._load()

    # =====================================================
    def _load(self):

        if os.path.exists(CONFIG_FILE):
            try:
                self.data = json.load(open(CONFIG_FILE))
            except:
                pass

    # =====================================================
    def _save(self):

        os.makedirs("logs", exist_ok=True)

        with open(CONFIG_FILE, "w") as f:
            json.dump(self.data, f, indent=2)

    # =====================================================
    # 🔥 RECORD LLM RESULT
    # =====================================================
    def record_llm_result(self, success):

        self.data["llm_calls"] += 1

        if success:
            self.data["llm_success"] += 1

        # Auto adjust every 20 calls
        if self.data["llm_calls"] % 20 == 0:
            self._adjust()

        self._save()

    # =====================================================
    # 🔥 AUTO ADJUST LOGIC
    # =====================================================
    def _adjust(self):

        success_rate = self.data["llm_success"] / max(1, self.data["llm_calls"])

        print(f"📊 LLM success rate: {success_rate:.2f}")

        # 🔥 If LLM is NOT useful → reduce calls
        if success_rate < 0.3:
            self.data["project_threshold"] += 100
            self.data["relation_threshold"] += 200

        # 🔥 If LLM is VERY useful → call more often
        elif success_rate > 0.7:
            self.data["project_threshold"] = max(200, self.data["project_threshold"] - 50)
            self.data["relation_threshold"] = max(500, self.data["relation_threshold"] - 100)

        # Reset counters
        self.data["llm_calls"] = 0
        self.data["llm_success"] = 0

    # =====================================================
    def get_project_threshold(self):
        return self.data["project_threshold"]

    def get_relation_threshold(self):
        return self.data["relation_threshold"]
