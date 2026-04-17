from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from langchain.schema import Document

from backend.src.config.config import Config


class AzureDocIntelligence:

    def __init__(self):

        # 🔹 Feature toggle (VERY IMPORTANT for cost control)
        if not getattr(Config, "USE_AZURE_DOC_INTELLIGENCE", False):
            self.client = None
            return

        try:
            self.client = DocumentAnalysisClient(
                endpoint=Config.AZURE_DOC_INTELLIGENCE_ENDPOINT,
                credential=AzureKeyCredential(Config.AZURE_DOC_INTELLIGENCE_KEY)
            )

            # Default model
            self.model = getattr(
                Config,
                "AZURE_DOC_INTELLIGENCE_MODEL",
                "prebuilt-layout"
            )

        except Exception as e:
            print("❌ Azure client init failed:", e)
            self.client = None

    # =========================================================
    # 🔥 MAIN PARSE METHOD
    # =========================================================
    def parse(self, file_path):

        if not self.client:
            print("⚠️ Azure disabled or not configured")
            return None

        try:
            with open(file_path, "rb") as f:
                poller = self.client.begin_analyze_document(
                    self.model,
                    document=f
                )

            result = poller.result()

            docs = []

            # =====================================================
            # 🔹 TEXT EXTRACTION (PARAGRAPH LEVEL - BEST FOR RAG)
            # =====================================================
            if hasattr(result, "paragraphs") and result.paragraphs:

                for p in result.paragraphs:

                    content = p.content.strip()

                    if not content or len(content) < 20:
                        continue

                    docs.append(
                        Document(
                            page_content=content,
                            metadata={
                                "source": file_path,
                                "type": "text",
                                "parser": "azure",
                                "role": getattr(p, "role", None)
                            }
                        )
                    )

            else:
                # 🔥 Fallback if paragraphs not available
                text = ""
                for page in result.pages:
                    for line in page.lines:
                        text += line.content + "\n"

                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": file_path,
                                "type": "text",
                                "parser": "azure"
                            }
                        )
                    )

            # =====================================================
            # 🔹 TABLE EXTRACTION (STRUCTURED)
            # =====================================================
            for table in getattr(result, "tables", []):

                rows = []
                current_row = []

                for cell in table.cells:

                    current_row.append(cell.content)

                    # New row detection
                    if len(current_row) == table.column_count:
                        rows.append(" | ".join(current_row))
                        current_row = []

                if rows:
                    table_text = "\n".join(rows)

                    docs.append(
                        Document(
                            page_content=table_text,
                            metadata={
                                "source": file_path,
                                "type": "table",
                                "parser": "azure",
                                "rows": len(rows),
                                "columns": table.column_count
                            }
                        )
                    )

            # =====================================================
            # 🔹 SAFETY CHECK
            # =====================================================
            if not docs:
                print("⚠️ Azure returned empty result")
                return None

            print(f"✅ Azure parsed {len(docs)} chunks")

            return docs

        except Exception as e:
            print(f"❌ Azure parsing failed for {file_path}: {e}")
            return None
