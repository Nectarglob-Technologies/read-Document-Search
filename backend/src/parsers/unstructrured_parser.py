from unstructured.partition.pdf import partition_pdf
from langchain.schema import Document


class UnstructuredParser:

    def parse(self, file_path):

        elements = partition_pdf(
            filename=file_path,
            infer_table_structure=True,
            strategy="hi_res"
        )

        docs = []

        for el in elements:

            text = el.text.strip() if hasattr(el, "text") else ""

            if not text:
                continue

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path,
                        "type": el.category,
                        "parser": "unstructured"
                    }
                )
            )

        return docs
