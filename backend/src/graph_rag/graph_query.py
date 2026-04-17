import re
import json
import os
try:
    from sentence_transformers import SentenceTransformer, util
    EMBEDDING_AVAILABLE = True
except:
    EMBEDDING_AVAILABLE = False



class GraphQueryEngine:

    def __init__(self, graph_store):

        self.graph = graph_store

        # 🔥 Embedding model (FREE)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # 🔥 Dynamic pattern store
        self.pattern_file = "patterns.json"
        self.patterns = self.load_patterns()

        # 🔥 Cache graph entities
        self.materials = self._get_entities("material")
        self.locations = self._get_entities("location")

        # Precompute embeddings
        self.material_embeddings = self.model.encode(self.materials, convert_to_tensor=True)
        self.location_embeddings = self.model.encode(self.locations, convert_to_tensor=True)


    # ---------------- LOAD/SAVE PATTERNS ----------------
    def load_patterns(self):
        if os.path.exists(self.pattern_file):
            return json.load(open(self.pattern_file))
        return {"materials": [], "locations": []}

    def save_patterns(self):
        json.dump(self.patterns, open(self.pattern_file, "w"), indent=2)


    # ---------------- GRAPH ENTITY EXTRACTION ----------------
    def _get_entities(self, entity_type):
        return [
            node for node, data in self.graph.graph.nodes(data=True)
            if data.get("type") == entity_type
        ]


    # ---------------- 1️⃣ RULE-BASED ----------------
    def extract_entity_rule(self, question):

        material = None
        location = None

        # Static regex
        mat_match = re.search(r"M\d+\s*Concrete", question, re.IGNORECASE)
        if mat_match:
            material = mat_match.group(0)

        loc_match = re.search(
            r"(Maharashtra|Mumbai|Delhi|Pune|Bangalore)",
            question,
            re.IGNORECASE
        )
        if loc_match:
            location = loc_match.group(0)

        return material, location


    # ---------------- 2️⃣ DYNAMIC PATTERN ----------------
    def extract_entity_dynamic(self, question):

        material = None
        location = None

        for pattern in self.patterns["materials"]:
            if pattern.lower() in question.lower():
                material = pattern

        for pattern in self.patterns["locations"]:
            if pattern.lower() in question.lower():
                location = pattern

        return material, location


    def learn_pattern(self, material, location):
        updated = False

        if material and material not in self.patterns["materials"]:
            self.patterns["materials"].append(material)
            updated = True

        if location and location not in self.patterns["locations"]:
            self.patterns["locations"].append(location)
            updated = True

        if updated:
            self.save_patterns()


    # ---------------- 3️⃣ EMBEDDING MATCHING ----------------
    def extract_entity_embedding(self, question):

        query_embedding = self.model.encode(question, convert_to_tensor=True)

        material = None
        location = None

        # Material match
        if len(self.materials) > 0:
            scores = util.cos_sim(query_embedding, self.material_embeddings)[0]
            best_idx = scores.argmax().item()

            if scores[best_idx] > 0.6:
                material = self.materials[best_idx]

        # Location match
        if len(self.locations) > 0:
            scores = util.cos_sim(query_embedding, self.location_embeddings)[0]
            best_idx = scores.argmax().item()

            if scores[best_idx] > 0.6:
                location = self.locations[best_idx]

        return material, location


    # ---------------- MAIN QUERY ----------------
    def query(self, question):

        print("\n🔍 Step 1: Rule-based extraction")
        material, location = self.extract_entity_rule(question)

        if material or location:
            print("✅ Rule-based success")

        else:
            print("⚠️ Rule failed → trying dynamic patterns")
            material, location = self.extract_entity_dynamic(question)

        if not material and not location:
            print("⚠️ Dynamic failed → using embeddings")
            material, location = self.extract_entity_embedding(question)

        # 🔥 Learn patterns automatically
        self.learn_pattern(material, location)

        print(f"📌 Extracted → Material: {material}, Location: {location}")

        # Query graph
        if material:
            return self.graph.get_projects_using_material(material)

        if location:
            return self.graph.get_projects_by_location(location)

        return ["No matching data found."]
