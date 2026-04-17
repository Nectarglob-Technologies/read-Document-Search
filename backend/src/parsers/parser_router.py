import json
from pathlib import Path


class ParserRouter:

    def __init__(self, memory_file="parser_memory.json"):

        self.memory_file = memory_file
        self.memory = self._load_memory()

    # =========================================================
    # 🔹 LOAD MEMORY
    # =========================================================
    def _load_memory(self):

        try:
            with open(self.memory_file, "r") as f:
                return json.load(f)
        except:
            return {}

    # =========================================================
    # 🔹 SAVE MEMORY
    # =========================================================
    def _save_memory(self):

        with open(self.memory_file, "w") as f:
            json.dump(self.memory, f, indent=2)

    # =========================================================
    # 🔹 DOCUMENT TYPE DETECTION
    # =========================================================
    def _get_doc_type(self, file_path):

        name = file_path.lower()

        if any(k in name for k in ["contract", "agreement", "tender"]):
            return "contract"

        if any(k in name for k in ["boq", "quantity", "schedule"]):
            return "table"

        if any(k in name for k in ["scan", "drawing"]):
            return "scanned"

        return "general"

    # =========================================================
    # 🔥 MAIN METHOD (FINAL)
    # =========================================================
    def get_parser_order(self, file_path):

        doc_type = self._get_doc_type(file_path)
        print(f"📄 Document type detected: {doc_type} for file {file_path}")

        size = Path(file_path).stat().st_size
        print(f"📏 Document size: {size} bytes for file {file_path} ")

        # 🔥 1. Learned behavior (highest priority)
        if doc_type in self.memory:
            print(f"🧠 Using learned best parser: {self.memory[doc_type]} for document type: {doc_type} ")
            best = self.memory[doc_type]

            # Ensure fallback chain is always safe
            fallback_chain = ["azure", "unstructured", "pypdf"]
            print(f"🔁 Fallback chain: {fallback_chain} for document type: {doc_type} ")

            ordered = [best] + [p for p in fallback_chain if p != best]
            print(f"✅ Final parser order: {ordered} for document type: {doc_type} ")

            return ordered

        # 🔥 2. Smart defaults (no learning yet)

        # Contracts → accuracy first
        if doc_type == "contract":
            print(f"📑 Defaulting to contract parser order for file {file_path} ")
            return ["azure", "unstructured", "pypdf"]

        # Tables → structure extraction
        if doc_type == "table":
            print(f"📑 Defaulting to table parser order for file {file_path} ")
            return ["azure", "unstructured", "pypdf"]

        # Scanned → OCR needed
        if doc_type == "scanned" or size > 3_000_000:
            print(f"📑 Defaulting to scanned parser order for file {file_path} ")
            return ["unstructured", "azure", "pypdf"]

        # General docs → fast first
        print(f"📑 Defaulting to general parser order for file {file_path} ")
        return ["pypdf", "unstructured", "azure"]

    # =========================================================
    # 🔥 LEARNING STEP (AUTO-OPTIMIZATION)
    # =========================================================
    def update_learning(self, file_path, best_parser):

        doc_type = self._get_doc_type(file_path)
        print(f"📚 Updating learning for document type: {doc_type} with best parser: {best_parser} for file {file_path} ")

        # Only store meaningful parsers
        if best_parser not in ["azure", "unstructured", "pypdf"]:
            print(f"❌ Ignoring learned parser: {best_parser} for document type: {doc_type} ")
            return

        self.memory[doc_type] = best_parser
        print(f"✅ Learned best parser: {best_parser} for document type: {doc_type} ")
        self._save_memory()

        
