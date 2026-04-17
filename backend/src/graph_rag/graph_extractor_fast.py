import re


class FastGraphExtractor:

    def extract(self, text):

        data = {
            "projects": [],
            "contractors": [],
            "materials": [],
            "locations": [],
            "relations": []
        }

        # 🔹 MATERIALS
        materials = re.findall(r"(M\d+\s*Concrete)", text, re.IGNORECASE)

        # 🔹 LOCATIONS
        locations = re.findall(
            r"\b(Maharashtra|Mumbai|Delhi|Pune|Bangalore)\b",
            text,
            re.IGNORECASE
        )

        # 🔹 PROJECTS
        projects = re.findall(
            r"\b([A-Z][a-zA-Z]+\s+(Bridge|Project|Road|Highway))\b",
            text
        )
        projects = [p[0] for p in projects]

        # 🔹 CONTRACTORS
        contractors = re.findall(
            r"\b([A-Z][a-zA-Z]+\s+(Constructions|Builders|Infra))\b",
            text
        )
        contractors = [c[0] for c in contractors]

        # Deduplicate
        data["projects"] = list(set(projects))
        data["materials"] = list(set(materials))
        data["locations"] = list(set(locations))
        data["contractors"] = list(set(contractors))

        # 🔥 SMART RELATION (line-based)
        lines = text.split("\n")

        for line in lines:

            found_projects = [
                p for p in data["projects"] if p in line
            ]

            found_materials = [
                m for m in data["materials"] if m in line
            ]

            found_locations = [
                l for l in data["locations"] if l in line
            ]

            for p in found_projects:

                for m in found_materials:
                    data["relations"].append({
                        "project": p,
                        "relation": "USES",
                        "entity": m
                    })

                for l in found_locations:
                    data["relations"].append({
                        "project": p,
                        "relation": "LOCATED_IN",
                        "entity": l
                    })

        return data
