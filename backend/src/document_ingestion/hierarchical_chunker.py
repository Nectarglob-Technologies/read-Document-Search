import re
from langchain.schema import Document


class HierarchicalChunker:

    def __init__(self, max_chunk_size=500, overlap=2):

        self.section_pattern = re.compile(r"Section\s+(\d+)", re.IGNORECASE)
        self.clause_pattern = re.compile(r"Clause\s+(\d+(\.\d+)?)", re.IGNORECASE)

        self.numbered_pattern = re.compile(r"^\s*(\d+(\.\d+)*)")
        self.heading_pattern = re.compile(r"^[A-Z][A-Z\s]{5,}$")

        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def detect_structure(self, para):

        if self.section_pattern.search(para):
            return "section"

        if self.clause_pattern.search(para):
            return "clause"

        if self.numbered_pattern.match(para):
            return "numbered"

        if self.heading_pattern.match(para):
            return "heading"

        return None

    def process(self, text, source):

        docs = []

        current_section = None
        current_clause = None

        detected_clauses = set()

        paragraphs = re.split(r"\n\s*\n", text)

        buffer_paragraphs = []
        buffer_length = 0

        for para in paragraphs:

            para = para.strip()
            if not para:
                continue

            structure_type = self.detect_structure(para)

            # 🔥 FORCE SPLIT on new structure
            if structure_type in ["section", "clause"] and buffer_paragraphs:

                docs.append(
                    Document(
                        page_content="\n\n".join(buffer_paragraphs),
                        metadata={
                            "source": source,
                            "section": current_section,
                            "clause": current_clause,
                            "clauses": list(detected_clauses)
                        }
                    )
                )

                # overlap handling
                buffer_paragraphs = buffer_paragraphs[-self.overlap:]
                buffer_length = sum(len(p) for p in buffer_paragraphs)

                # 🔥 rebuild clause tracking from overlap
                detected_clauses = set()
                for p in buffer_paragraphs:
                    match = self.clause_pattern.search(p)
                    if match:
                        detected_clauses.add(match.group(1))

            # 🔹 Update structure
            if structure_type == "section":
                current_section = self.section_pattern.search(para).group(1)

            elif structure_type == "clause":
                current_clause = self.clause_pattern.search(para).group(1)
                detected_clauses.add(current_clause)

            elif structure_type == "numbered":
                current_clause = para.split()[0]
                detected_clauses.add(current_clause)

            elif structure_type == "heading":
                current_section = para[:50]

            buffer_paragraphs.append(para)
            buffer_length += len(para)

            # 🔥 SIZE-BASED SPLIT
            if buffer_length >= self.max_chunk_size:

                docs.append(
                    Document(
                        page_content="\n\n".join(buffer_paragraphs),
                        metadata={
                            "source": source,
                            "section": current_section,
                            "clause": current_clause,
                            "clauses": list(detected_clauses)
                        }
                    )
                )

                buffer_paragraphs = buffer_paragraphs[-self.overlap:]
                buffer_length = sum(len(p) for p in buffer_paragraphs)

                # rebuild clauses after overlap
                detected_clauses = set()
                for p in buffer_paragraphs:
                    match = self.clause_pattern.search(p)
                    if match:
                        detected_clauses.add(match.group(1))

        # 🔹 Final chunk
        if buffer_paragraphs:
            docs.append(
                Document(
                    page_content="\n\n".join(buffer_paragraphs),
                    metadata={
                        "source": source,
                        "section": current_section,
                        "clause": current_clause,
                        "clauses": list(detected_clauses)
                    }
                )
            )

        return docs
