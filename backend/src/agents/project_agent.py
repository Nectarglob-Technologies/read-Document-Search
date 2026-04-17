class ProjectAgent:

    def __init__(self, graph_store):
        self.graph = graph_store

    def run(self, question):

        print("ProjectAgent using Graph (no hardcoding)")

        # Extract project name (simple version)
        words = question.lower().split()

        for node in self.graph.graph.nodes(data=True):
            name, data = node

            if data.get("type") == "project":
                if name.lower() in question.lower():

                    return f"""
🏗 Project Info (from Graph):

Project: {name}

Connected Data:
{list(self.graph.graph.neighbors(name))}

Confidence: 0.8
"""

        return "Project not found.\nConfidence: 0.4"
