import re


class TableGraphExtractor:
    """
    Convert table rows into graph relationships
    """

    def extract(self, text):

        relations = []

        # Example:
        # Project | Contractor | Material
        rows = text.split("|")

        if len(rows) < 3:
            return []

        # Try extracting triplets
        for i in range(0, len(rows) - 2, 3):

            project = rows[i].strip()
            contractor = rows[i + 1].strip()
            material = rows[i + 2].strip()

            if project and contractor:
                relations.append({
                    "project": project,
                    "relation": "BUILT_BY",
                    "entity": contractor
                })

            if project and material:
                relations.append({
                    "project": project,
                    "relation": "USES",
                    "entity": material
                })

        return relations
