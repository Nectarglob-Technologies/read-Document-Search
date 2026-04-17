# backend/src/utils/logger.py

import json
from datetime import datetime
import re
import uuid
import logging
import os

# =========================================================
# 🔥 SYSTEM LOGGING (APP LOGS)
# =========================================================
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")


def get_logger(name="app"):

    logger = logging.getLogger(name)

    # ✅ Prevent duplicate handlers (VERY IMPORTANT)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    # ✅ FILE HANDLER (UTF-8 FIX for emoji)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    # ✅ CONSOLE HANDLER
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


# =========================================================
# 🔥 BUSINESS LOGGING (Q&A TRACKING)
# =========================================================
class Logger:

    def __init__(self, file="logs/qa_logs.json"):
        self.file = file

    # =====================================================
    # 🔹 Extract confidence
    # =====================================================
    def _extract_confidence(self, text):

        match = re.search(r"confidence[:\s]+(\d+\.?\d*)", text.lower())

        if match:
            try:
                return float(match.group(1))
            except:
                return 0.5

        return 0.5

    # =====================================================
    # 🔹 Extract source
    # =====================================================
    def _extract_source(self, text):

        if "source:" in text.lower():
            return "document"

        if "graph" in text.lower():
            return "graph"

        return "hybrid"

    # =====================================================
    # 🔹 MAIN LOG FUNCTION
    # =====================================================
    def log(self, question, answer, source=None, confidence=None):

        if confidence is None:
            confidence = self._extract_confidence(answer)

        if source is None:
            source = self._extract_source(answer)

        entry = {
            "request_id": str(uuid.uuid4()),
            "time": str(datetime.now()),
            "question": question,
            "answer": answer[:300],
            "source": source,
            "confidence": confidence
        }

        # ✅ SAFE FILE READ
        try:
            if os.path.exists(self.file):
                with open(self.file, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = []
        except:
            data = []

        data.append(entry)

        # ✅ SAFE FILE WRITE
        with open(self.file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
